#!/bin/bash
# PRISM Stack Installation Script for Perlmutter

echo "Loading modules..."
module load PrgEnv-gnu cudatoolkit/12.9.lua python/3.12

echo "Configuring cache endpoints to SCRATCH..."
export UV_PYTHON_INSTALL_DIR=$SCRATCH/PRISM_env/.cache/uv
export UV_CACHE_DIR=$SCRATCH/PRISM_env/.cache/uv
export VLLM_CACHE_ROOT=$SCRATCH/PRISM_env/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/PRISM_env/.cache/torchinductor
export HF_HOME=$SCRATCH/PRISM_env/.cache/hf_home
export TRANSFORMERS_HOME=$SCRATCH/PRISM_env/.cache/transformers
export HF_DATASETS_CACHE=$SCRATCH/PRISM_env/.cache/hf_datasets

# Perlmutter A100 architecture
export TORCH_CUDA_ARCH_LIST="8.0"

echo "Installing uv package manager..."
pip install uv


echo "Creating and activating virtual environment..."
python -m venv $SCRATCH/PRISM_env/venvs/vllm-env
source $SCRATCH/PRISM_env/venvs/vllm-env/bin/activate

echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

echo "Cloning and building vLLM from source..."
cd $SCRATCH/PRISM_env
git clone --branch v0.17.0 https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt

# Bypassing FlashInfer due to Perlmutter compatibility issues
uv pip install flash-attn --no-build-isolation
uv pip uninstall flashinfer-python apache-tvm-ffi

echo "Compiling vLLM backend (This will take time)..."
MAX_JOBS=16 CC=$(which gcc) CXX=$(which g++) uv pip install --no-build-isolation -e . -v --pre

echo "Installing Dynamo Orchestrator and Networking Libraries..."
# Critical: Install without the [vllm] flag so it does not overwrite the custom build above
uv pip install ai-dynamo msgpack aiperf
uv pip install nixl kvbm

echo "PRISM Stack Installation Complete!"
