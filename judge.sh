#!/bin/bash

uv run vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
	--port 8080 \
	--max-model-len 4096 \
	--gpu-memory-utilization 0.35 \
	--enforce-eager \
	--max-num-seqs 64 \
	--quantization awq_marlin

echo "vLLM Judge spawned on 8080"