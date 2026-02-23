import asyncio
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# Import the original complex logic from where you saved it
# (Assuming you saved the code you provided as 'transluce_logic.py' in the same folder)
from .reward import compute_reward

# 1. Define your model names globally (Must match exactly what you pass to vLLM)
TARGET_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. Initialize the tokenizer globally so it doesn't reload on every single prompt
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)

def compute_score(solution_str: str, ground_truth: dict) -> float:
    behavior = ground_truth.get('behavior', '')
    behavior_id = ground_truth.get('behavior_id', 'unknown')
    optimizer_target = ground_truth.get('optimizer_target', 'target_model')

    # Define an internal async function to handle the API calls
    async def _run_async_reward():
        # A. Initialize clients INSIDE the async function. 
        # This prevents asyncio loop crashes across Ray workers.
        vllm_target_client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_retries=3
        )
        vllm_judge_client = AsyncOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            max_retries=3
        )

        # B. Call the paper's heavy logic
        prbo_result = await compute_reward(
            gpt_oss_client=vllm_target_client,
            gpt_oss_tokenizer=tokenizer,
            behavior_id=behavior_id,
            policy_output=solution_str,
            optimizer_target=optimizer_target,
            behavior=behavior,
            openai_client=vllm_judge_client,
            target_model=TARGET_MODEL_NAME,
            prompt_judge_model=JUDGE_MODEL_NAME,
            response_judge_model=JUDGE_MODEL_NAME,
        )
        
        return prbo_result

    # 3. Bridge the async gap using asyncio.run
    try:
        prbo_struct = asyncio.run(_run_async_reward())
        
        # Optional: Print the beautiful Rich table to your terminal for debugging
        # pretty_print_reward(prbo_struct)
        
        # 4. Return ONLY the float that verl needs for the PPO update
        return float(prbo_struct.score)
        
    except Exception as e:
        print(f"Reward computation failed for behavior {behavior_id}: {e}")
        return -100.0