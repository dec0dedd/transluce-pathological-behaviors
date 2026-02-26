#!/bin/bash


uv run vllm serve Qwen/Qwen2.5-0.5B-Instruct \
	--port 8080 \
	--max-model-len 4096 \
	--gpu-memory-utilization 0.1 \
	--enforce-eager \
	--max-num-seqs 16

echo "vLLM Judge spawned on 8080"