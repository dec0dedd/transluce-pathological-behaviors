#!/bin/bash


uv run vllm serve Qwen/Qwen2.5-0.5B-Instruct \
	--port 8000 \
	--max-model-len 4096 \
	--gpu-memory-utilization 0.1 \
	--enforce-eager \
	--max-num-seqs 16

echo "vLLM servers starting on ports 8000 and 8001..."