#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 7B 1-node aggregated native vLLM launch script
#
# Layout:
#   NODE0 GPUs 0,1,2,3 -> Aggregated vLLM TP=4
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"

VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

AGG_CUDA_VISIBLE_DEVICES="${AGG_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
AGG_TP_SIZE="${AGG_TP_SIZE:-4}"
AGG_PP_SIZE="${AGG_PP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
MAX_VIDEOS_PER_PROMPT="${MAX_VIDEOS_PER_PROMPT:-0}"
LIMIT_MM_PER_PROMPT="{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}"

LOG_DIR="${LOG_DIR:-./artifacts/p4_7b/aggregated_1node_logs}"
mkdir -p "$LOG_DIR"

CLUSTER_ENV_FILE="${LOG_DIR}/cluster_env.sh"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  log "ERROR: This script expects to run inside a Slurm allocation." >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
CLIENT_NODE="${NODES[0]}"
CLIENT_IP="$(ssh -q "$CLIENT_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"

: "${VLLM_ROOT:?VLLM_ROOT is empty}"
: "${VENV_ACTIVATE:?VENV_ACTIVATE is empty}"

log "=== Phase 4 7B 1-node aggregated launch ==="
log "MODEL=$MODEL"
log "PORT=$PORT"
log "CLIENT_NODE=$CLIENT_NODE"
log "CLIENT_IP=$CLIENT_IP"
log "VLLM_ROOT=$VLLM_ROOT"
log "VENV_ACTIVATE=$VENV_ACTIVATE"
log "AGG_CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES"
log "AGG_TP_SIZE=$AGG_TP_SIZE"
log "AGG_PP_SIZE=$AGG_PP_SIZE"
log "MAX_MODEL_LEN=$MAX_MODEL_LEN"
log "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
log "LIMIT_MM_PER_PROMPT=$LIMIT_MM_PER_PROMPT"
log "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
  echo
  echo "Layout:"
  echo "  $CLIENT_NODE GPUs $AGG_CUDA_VISIBLE_DEVICES -> Aggregated TP=$AGG_TP_SIZE PP=$AGG_PP_SIZE"
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export MODE="aggregated_1node"
export CLIENT_NODE="$CLIENT_NODE"
export CLIENT_IP="$CLIENT_IP"
export PORT="$PORT"
export MODEL="$MODEL"
export VLLM_ROOT="$VLLM_ROOT"
export AGG_CUDA_VISIBLE_DEVICES="$AGG_CUDA_VISIBLE_DEVICES"
export AGG_TP_SIZE="$AGG_TP_SIZE"
export AGG_PP_SIZE="$AGG_PP_SIZE"
export LIMIT_MM_PER_PROMPT='$LIMIT_MM_PER_PROMPT'
EOF

log "Wrote cluster env to $CLUSTER_ENV_FILE"

cleanup() {
  set +e
  log "Stopping aggregated 1-node vLLM stack..."

  ssh -q "$CLIENT_NODE" "
    source '$VENV_ACTIVATE' >/dev/null 2>&1 || true
    pkill -f 'vllm serve' >/dev/null 2>&1 || true
    pkill -f 'vllm.entrypoints.openai.api_server' >/dev/null 2>&1 || true
    pkill -f 'VLLM::EngineCore' >/dev/null 2>&1 || true
    pkill -f 'RayWorkerWrapper' >/dev/null 2>&1 || true
    ray stop --force >/dev/null 2>&1 || true
  " 2>/dev/null || true
}

trap cleanup EXIT

log "Cleaning existing processes..."
cleanup
sleep 5

log "Starting aggregated vLLM on $CLIENT_NODE GPUs $AGG_CUDA_VISIBLE_DEVICES"

ssh -q "$CLIENT_NODE" "
set -euo pipefail
source '$VENV_ACTIVATE'
cd '$VLLM_ROOT'

export HF_HOME=\"\${SCRATCH:-\$HOME}/.cache/huggingface\"
export HF_TRANSFORMERS_CACHE=\"\$HF_HOME\"
export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\"
export TORCHINDUCTOR_CACHE_DIR=\"\${SCRATCH:-\$HOME}/.cache/torch_inductor\"
export NCCL_DEBUG=\"\${NCCL_DEBUG:-WARN}\"

CUDA_VISIBLE_DEVICES='$AGG_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$PORT' \
  --tensor-parallel-size '$AGG_TP_SIZE' \
  --pipeline-parallel-size '$AGG_PP_SIZE' \
  --enable-request-id-headers \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT'
" > "$LOG_DIR/vllm_serve.log" 2>&1 &

VLLM_PID=$!

{
  echo "VLLM_SHELL_PID=$VLLM_PID"
  echo "CLIENT_NODE=$CLIENT_NODE"
  echo "CLIENT_IP=$CLIENT_IP"
  echo "PORT=$PORT"
  echo "AGG_CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES"
  echo "AGG_TP_SIZE=$AGG_TP_SIZE"
  echo "AGG_PP_SIZE=$AGG_PP_SIZE"
} > "$LOG_DIR/launcher_pids.txt"

log "Aggregated 1-node vLLM launched. Waiting on child process..."
wait "$VLLM_PID"
