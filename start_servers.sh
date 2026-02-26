#!/bin/bash


uv run vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15 \
    --enforce-eager \
    --max-num-seqs 16

echo "vLLM Target spawned on 8000"