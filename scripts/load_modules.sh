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

echo "Activating virtual environment..."
source $SCRATCH/PRISM_env/venvs/vllm-env/bin/activate
