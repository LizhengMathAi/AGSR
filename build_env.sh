#!/usr/bin/env bash
set -e

# 1) Create and activate conda env
conda create -n sam3 python=3.12 -y
conda activate sam3

# 2) Install PyTorch with CUDA
# Official repo currently recommends PyTorch 2.10 + CUDA 12.8 wheels
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) Install Hugging Face tools
pip install -U huggingface_hub transformers accelerate pillow requests

