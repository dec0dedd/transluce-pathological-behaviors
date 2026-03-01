import asyncio
import json
import os
import traceback
import logging
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.reward import compute_reward

TARGET_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"

os.makedirs("reward_logs", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
logger = logging.getLogger(__name__)

def _safe_judge_extract(judge_obj) -> tuple[float | None, str, str]:
    if not judge_obj:
        return None, "N/A", "N/A"
    return judge_obj.score, judge_obj.judge_prompt, judge_obj.judge_response

def compute_score(data_source: Any, solution_str: str, ground_truth: dict, extra_info: Any = None) -> float:
    behavior = ground_truth.get('behavior', '')
    behavior_id = ground_truth.get('behavior_id', 'unknown')
    optimizer_target = ground_truth.get('optimizer_target', 'target_model')

    async def _run_async_reward():
        async with AsyncOpenAI(
            base_url="http://localhost:8000/v1", api_key="EMPTY", max_retries=3
        ) as vllm_target_client, AsyncOpenAI(
            base_url="http://localhost:8080/v1", api_key="EMPTY", max_retries=3
        ) as vllm_judge_client:

            return await compute_reward(
                target_client=vllm_target_client,
                target_tokenizer=tokenizer,
                judge_client=vllm_judge_client,
                behavior_id=behavior_id,
                policy_output=solution_str,
                optimizer_target=optimizer_target,
                behavior=behavior,
                target_model=TARGET_MODEL_NAME,
                prompt_judge_model=JUDGE_MODEL_NAME,
                response_judge_model=JUDGE_MODEL_NAME,
            )

    try:
        prbo_struct = asyncio.run(_run_async_reward())

        p_score, p_prompt, p_resp = _safe_judge_extract(prbo_struct.prompt_score)
        n_score, n_prompt, n_resp = _safe_judge_extract(prbo_struct.normal_response_score)
        s_score, s_prompt, s_resp = _safe_judge_extract(prbo_struct.steered_response_score)

        log_data = {
            "behavior_id": prbo_struct.behavior_id,
            "behavior": prbo_struct.behavior,
            "policy_output": prbo_struct.policy_output,
            "score": float(prbo_struct.score),
            "proposal_prefix": prbo_struct.proposal_prefix,

            "normal_response": prbo_struct.normal_response or "FAILED_TO_GENERATE",
            "steered_response": prbo_struct.steered_response or "FAILED_TO_GENERATE",

            "prompt_score_value": p_score,
            "prompt_judge_prompt": p_prompt,
            "prompt_judge_response": p_resp,

            "normal_response_score_value": n_score,
            "normal_judge_prompt": n_prompt,
            "normal_judge_response": n_resp,

            "steered_response_score_value": s_score,
            "steered_judge_prompt": s_prompt,
            "steered_judge_response": s_resp,
        }

        worker_pid = os.getpid()
        with open(f"reward_logs/{worker_pid}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
        
        return float(prbo_struct.score)

    except Exception as e:
        logger.error(f"Reward computation failed for behavior {behavior_id}: {e}")
        logger.error(traceback.format_exc())
        return -100.0