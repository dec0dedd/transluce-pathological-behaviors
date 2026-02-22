#!/bin/bash


# Target and Steered models both use Qwen 7B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.25 &


# Judge model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.35 &

echo "vLLM servers starting on ports 8000 and 8001..."