import logging
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tenacity import retry, before_sleep_log, stop_after_attempt, wait_random_exponential
from transformers import PreTrainedTokenizerBase

from src.logprobs import LogProbs

logger = logging.getLogger(__name__)

async def get_token_logprobs(
    client: AsyncOpenAI,
    tokenizer: PreTrainedTokenizerBase,
    model: str,
    output_text: str,
    input_token_ids: list[int] | None = None,
    input_messages: list[ChatCompletionMessageParam] | None = None,
) -> LogProbs:    
    if (input_token_ids is None) == (input_messages is None):
        raise ValueError("Must provide exactly one of 'input_token_ids' or 'input_messages'.")

    if input_messages is not None:
        conversation_tokens = cast(list[int], tokenizer.apply_chat_template(
            input_messages, add_generation_prompt=True
        ))
    else:
        conversation_tokens = cast(list[int], input_token_ids)

    full_text = tokenizer.decode(conversation_tokens) + output_text
    suffix_idx = len(conversation_tokens)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(50),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _fetch_logprobs() -> LogProbs:
        output = await client.completions.create(
            model=model,
            prompt=full_text,
            max_tokens=1,
            logprobs=1,
            echo=True,
        )

        logprobs_obj = output.choices[0].logprobs
        
        if logprobs_obj is None or logprobs_obj.tokens is None or logprobs_obj.token_logprobs is None:
            raise ValueError("Failed to get valid logprobs from the model response.")

        token_strs = logprobs_obj.tokens[:-1]
        token_logprobs = logprobs_obj.token_logprobs[:-1]

        return LogProbs(
            prompt_token_strs=token_strs[:suffix_idx],
            prompt_token_logprobs=cast(list[float], token_logprobs[:suffix_idx]),
            response_token_strs=token_strs[suffix_idx:],
            response_token_logprobs=cast(list[float], token_logprobs[suffix_idx:]),
        )

    return await _fetch_logprobs()