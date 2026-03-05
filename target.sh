#!/bin/bash


uv run vllm serve Qwen/Qwen2.5-1.5B-Instruct \
	--port 8000 \
	--max-model-len 4096 \
	--gpu-memory-utilization 0.15 \
	--enforce-eager \
	--max-num-seqs 64

echo "vLLM Target starting on ports 8000"