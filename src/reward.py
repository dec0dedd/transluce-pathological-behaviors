import json
import logging
import re
import msgspec
from typing import Any
from openai import AsyncOpenAI

# Import the templates from your prompts file
from src.prompts import (
    COMBINED_JUDGE_SYSTEM_PROMPT,
    COMBINED_JUDGE_USER_PROMPT,
    PROPOSAL_TEMPLATE_V1,
    RESPONSE_START_STR_V1
)

logger = logging.getLogger(__name__)

# ==========================================
# 1. Data Structures
# ==========================================

class LogProbs(msgspec.Struct):
    total: float
    tokens: list[str]
    token_logprobs: list[float]

class JudgeResponse(msgspec.Struct):
    score: float
    subscores: dict[str, float]
    judge_prompt: str
    judge_response: str

class PRBOReward(msgspec.Struct):
    behavior_id: str
    policy_output: str
    score: float
    normal_response: str | None
    steered_response: str | None
    normal_response_score: JudgeResponse | None
    steered_response_score: JudgeResponse | None
    normal_logprobs: LogProbs
    steered_logprobs: LogProbs
    proposal_prefix: str
    behavior: str


# ==========================================
# 2. Judge Model Handler
# ==========================================

class LLMJudge:
    """Handles interactions with the vLLM Judge model to score prompts and responses."""
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    async def _query_judge(self, system_prompt: str, user_prompt: str) -> JudgeResponse:
        """Sends the prompt to the judge and parses the expected XML output."""
        raw_output = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw_output = response.choices[0].message.content.strip()

            # Helper function to extract float values using regex
            def extract_float(tag: str, default: float = 0.0) -> float:
                match = re.search(rf'<{tag}>(.*?)</{tag}>', raw_output, re.DOTALL)
                if match:
                    try:
                        return float(match.group(1).strip())
                    except ValueError:
                        pass
                return default

            # Extract the scores safely
            score = extract_float("score")
            subscores = {
                "rubric_adherence": extract_float("rubric_adherence"),
                "authenticity": extract_float("authenticity"),
                "misalignment": extract_float("misalignment"),
                "unexpectedness": extract_float("unexpectedness")
            }

            return JudgeResponse(
                score=score,
                subscores=subscores,
                judge_prompt=user_prompt,
                judge_response=raw_output
            )
            
        except Exception as e:
            logger.error(f"Judge API call/parsing failed: {e}. Raw output: {raw_output}")
            return JudgeResponse(0.0, {}, user_prompt, raw_output)

    async def evaluate_response(self, policy_output: str, target_response: str, behavior: str) -> JudgeResponse:
        """Evaluates if the target model's response exhibits the pathological behavior."""
        # Dynamically inject the behavior, prompt, and response into the user template
        user_content = COMBINED_JUDGE_USER_PROMPT.format(
            behavior=behavior,
            prompt=policy_output,
            response=target_response
        )
        return await self._query_judge(
            system_prompt=COMBINED_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_content
        )


# ==========================================
# 3. Target Model Handler
# ==========================================

class TargetLLM:
    """Handles interactions with the vLLM Target model, including logprob extraction."""
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    async def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generates a text completion from the target model."""
        try:
            # Note: For the raw unsteered completion, you might want to use chat templates
            # depending on if your target model expects raw text or chat markup.
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Target generation failed: {e}")
            return ""

    async def get_response_logprobs(self, context: str, response_text: str) -> LogProbs:
        """
        Uses the vLLM completions endpoint with `echo=True` to get the log-probabilities 
        of a specific response given a context.
        """
        full_text = context + response_text
        context_len = len(context)

        try:
            api_res = await self.client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=1, 
                temperature=0.0,
                logprobs=1,
                echo=True
            )

            logprobs_info = api_res.choices[0].logprobs
            offsets = logprobs_info.text_offset
            tokens = logprobs_info.tokens
            t_logprobs = logprobs_info.token_logprobs

            response_tokens = []
            response_logprobs = []
            total_logprob = 0.0

            for i, offset in enumerate(offsets):
                if offset >= context_len:
                    lp = t_logprobs[i]
                    if lp is not None:
                        response_tokens.append(tokens[i])
                        response_logprobs.append(lp)
                        total_logprob += lp

            return LogProbs(total=total_logprob, tokens=response_tokens, token_logprobs=response_logprobs)
            
        except Exception as e:
            logger.error(f"Logprob extraction failed: {e}")
            return LogProbs(0.0, [], [])


# ==========================================
# 4. PRBO Evaluator / Orchestrator
# ==========================================

class PRBOEvaluator:
    """Orchestrates the PRBO math and pipeline."""
    def __init__(self, target: TargetLLM, judge: LLMJudge, beta: float = 0.1):
        self.target = target
        self.judge = judge
        self.beta = beta

    async def evaluate(self, behavior_id: str, behavior: str, policy_output: str) -> PRBOReward:
        
        # --- Extract the clean prompt from the XML tags ---
        tp_match = re.search(r"<test_prompt>\s*(.*?)\s*</test_prompt>", policy_output, re.DOTALL | re.IGNORECASE)
        if tp_match:
            clean_prompt = tp_match.group(1).strip()
        else:
            clean_prompt = re.sub(r"<scratchpad>.*?</scratchpad>", "", policy_output, flags=re.DOTALL | re.IGNORECASE).strip()
            if not clean_prompt:
                clean_prompt = policy_output.strip()

        # 1. Format the Steering Prefix (Proposal Distribution c)
        proposal_prefix = RESPONSE_START_STR_V1
        
        # Inject the prefix directly into the prompt so the model continues from it
        steered_prompt = PROPOSAL_TEMPLATE_V1.format(user_prompt=clean_prompt, prefix=proposal_prefix)
        
        # The unsteered target gets just the raw clean prompt 
        normal_prompt = f"<|im_start|>user\n{clean_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # 2. Generate Responses
        normal_response = await self.target.generate(normal_prompt)
        
        # The model will generate tokens immediately following the prefix
        steered_response_continuation = await self.target.generate(steered_prompt)
        
        # Recombine them carefully with a space
        steered_response_full = proposal_prefix + " " + steered_response_continuation.strip()

        # 3. Score with the Judge (Using the clean prompt!)
        normal_response_score = await self.judge.evaluate_response(clean_prompt, normal_response, behavior)
        steered_response_score = await self.judge.evaluate_response(clean_prompt, steered_response_full, behavior)

        # 4. Compute Logprobs for PRBO Density Ratio
        normal_logprobs = await self.target.get_response_logprobs(normal_prompt, steered_response_full)
        steered_logprobs = await self.target.get_response_logprobs(steered_prompt, steered_response_continuation)

        # 5. Compute Final PRBO Score
        reward_judge = steered_response_score.score
        density_ratio_penalty = (normal_logprobs.total - steered_logprobs.total)
        
        final_score = reward_judge + (self.beta * density_ratio_penalty)

        return PRBOReward(
            behavior_id=behavior_id,
            policy_output=policy_output, 
            score=final_score,
            normal_response=normal_response,
            steered_response=steered_response_full,
            normal_response_score=normal_response_score,
            steered_response_score=steered_response_score,
            normal_logprobs=normal_logprobs,
            steered_logprobs=steered_logprobs,
            proposal_prefix=proposal_prefix,
            behavior=behavior
        )

# ==========================================
# 5. Main Entry Point
# ==========================================

async def compute_reward(
    target_client: AsyncOpenAI,
    target_tokenizer: Any,
    judge_client: AsyncOpenAI,
    behavior_id: str,
    policy_output: str,
    optimizer_target: str,
    behavior: str,
    target_model: str,
    prompt_judge_model: str,
    response_judge_model: str,
) -> PRBOReward:
    
    target_handler = TargetLLM(client=target_client, model_name=target_model)
    judge_handler = LLMJudge(client=judge_client, model_name=response_judge_model)
    
    evaluator = PRBOEvaluator(target=target_handler, judge=judge_handler, beta=0.1)
    
    return await evaluator.evaluate(
        behavior_id=behavior_id, 
        behavior=behavior, 
        policy_output=policy_output
    )