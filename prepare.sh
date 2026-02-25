#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-0}
export MAX_JOBS=32

echo "0. Installing uv via curl..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "0.5 Setting up Python 3.12 Environment..."
# Create the environment explicitly
uv venv .venv --python 3.12 --allow-existing

echo "1. install inference frameworks and pytorch they need"
uv pip install --python .venv "vllm==0.11.0"

echo "2. install basic packages"
uv pip install --python .venv "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

uv pip install --python .venv "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashAttention and FlashInfer"
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install --python .venv flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install --python .venv flashinfer-python==0.3.1

echo "5. Fix opencv"
uv pip install --python .venv opencv-python opencv-fixer
.venv/bin/python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "7. Install your project (transluce-pathological-behaviors)"
uv pip install --python .venv -e .

echo "Successfully installed all packages FAST in Python 3.12!"
uv lock