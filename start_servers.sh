#!/bin/bash


# ports 8000 and 8080
uv run vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.25

echo "vLLM servers starting on ports 8000 and 8001..."