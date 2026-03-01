#!/bin/bash

uv run vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
	--port 8080 \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.5 \
	--enforce-eager \
	--max-num-seqs 16

echo "vLLM Judge spawned on 8080"