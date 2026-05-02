#!/bin/bash
set -euo pipefail

# LS6-ready PRISM stack setup for a 1-node A100 smoke path.
# Run this INSIDE an interactive LS6 GPU allocation, e.g.:
#   idev -p gpu-a100-dev -N 1 -n 1 -t 02:00:00 -A <ACCOUNT> -- --cpus-per-task=32 --exclusive

module reset
module load gcc/13.2.0 cuda/12.8 python/3.12.11

PRISM_ROOT="${PRISM_ROOT:-$SCRATCH/PRISM_env}"
VENV_DIR="${VENV_DIR:-$PRISM_ROOT/venvs/prism-env}"
SRC_DIR="${SRC_DIR:-$PRISM_ROOT/src}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$SRC_DIR/vllm}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100
MAX_JOBS="${MAX_JOBS:-16}"

mkdir -p "$PRISM_ROOT" "$PRISM_ROOT/venvs" "$PRISM_ROOT/.cache" "$SRC_DIR"

export UV_PYTHON_INSTALL_DIR="$PRISM_ROOT/.cache/uv"
export UV_CACHE_DIR="$PRISM_ROOT/.cache/uv"
export VLLM_CACHE_ROOT="$PRISM_ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PRISM_ROOT/.cache/torchinductor"
export HF_HOME="$PRISM_ROOT/.cache/hf_home"
export TRANSFORMERS_HOME="$PRISM_ROOT/.cache/transformers"
export HF_DATASETS_CACHE="$PRISM_ROOT/.cache/hf_datasets"
export TORCH_CUDA_ARCH_LIST

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install uv

# Match your working LS6 recipe: use cu128 wheels, not the Perlmutter cu129 setup.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

export TORCH_LIB="$VENV_DIR/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

cd "$SRC_DIR"
if [ ! -d "$VLLM_SRC_DIR/.git" ]; then
  git clone --branch v0.17.0 https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
fi
cd "$VLLM_SRC_DIR"

echo "Using existing torch build..."
python3 use_existing_torch.py
uv pip install -r requirements/build.txt

# PRISM's Perlmutter script installs flash-attn and removes flashinfer/tvm.
# Keep the same shape, but don't fail the whole setup if flash-attn is flaky.
if ! uv pip install flash-attn --no-build-isolation; then
  echo "WARNING: flash-attn install failed; continuing for smoke-test purposes."
fi
uv pip uninstall -y flashinfer-python apache-tvm-ffi || true

echo "Building vLLM from source..."
MAX_JOBS="$MAX_JOBS" CC="$(which gcc)" CXX="$(which g++)" \
  uv pip install --no-build-isolation -e . -v --prerelease allow

echo "Installing Dynamo + profiling dependencies..."
uv pip install ai-dynamo msgpack aiperf
uv pip install nixl kvbm

echo
echo "PRISM LS6 setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "vLLM source at: $VLLM_SRC_DIR"
echo "HF cache at: $HF_HOME"
