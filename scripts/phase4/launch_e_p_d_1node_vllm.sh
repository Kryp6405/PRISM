#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 7B 1-node E/P/D native vLLM launch script — NO RAY
#
# Layout:
#   NODE0 GPU 0     -> Encoder TP=1
#   NODE0 GPU 1     -> Prefill TP=1
#   NODE0 GPUs 2,3  -> Decode TP=2
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"

VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache_epd_1node_7b}"

PREFILL_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5559}"
DECODE_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_SIDE_CHANNEL_PORT:-6000}"

ENCODER_CUDA_VISIBLE_DEVICES="${ENCODER_CUDA_VISIBLE_DEVICES:-0}"
ENCODER_TP_SIZE="${ENCODER_TP_SIZE:-1}"

PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-1}"

DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-2,3}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-2}"
DECODE_PP_SIZE="${DECODE_PP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
MAX_VIDEOS_PER_PROMPT="${MAX_VIDEOS_PER_PROMPT:-0}"
LIMIT_MM_PER_PROMPT="{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}"

LOG_DIR="${LOG_DIR:-./artifacts/p4_7b/e_p_d_1node_logs}"
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

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

log "=== Phase 4 7B 1-node E/P/D launch ==="
log "MODEL=$MODEL"
log "PORT=$PORT"
log "ENCODE_PORT=$ENCODE_PORT"
log "PREFILL_PORT=$PREFILL_PORT"
log "DECODE_PORT=$DECODE_PORT"
log "CLIENT_NODE=$CLIENT_NODE"
log "CLIENT_IP=$CLIENT_IP"
log "VLLM_ROOT=$VLLM_ROOT"
log "VENV_ACTIVATE=$VENV_ACTIVATE"
log "EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
log "PREFILL_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_SIDE_CHANNEL_PORT"
log "DECODE_NIXL_SIDE_CHANNEL_PORT=$DECODE_NIXL_SIDE_CHANNEL_PORT"
log "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
log "ENCODER_TP_SIZE=$ENCODER_TP_SIZE"
log "PREFILL_CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES"
log "PREFILL_TP_SIZE=$PREFILL_TP_SIZE"
log "PREFILL_PP_SIZE=$PREFILL_PP_SIZE"
log "DECODE_CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES"
log "DECODE_TP_SIZE=$DECODE_TP_SIZE"
log "DECODE_PP_SIZE=$DECODE_PP_SIZE"
log "MAX_MODEL_LEN=$MAX_MODEL_LEN"
log "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
log "LIMIT_MM_PER_PROMPT=$LIMIT_MM_PER_PROMPT"
log "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
  echo
  echo "Layout:"
  echo "  $CLIENT_NODE GPU $ENCODER_CUDA_VISIBLE_DEVICES -> Encoder TP=$ENCODER_TP_SIZE"
  echo "  $CLIENT_NODE GPU $PREFILL_CUDA_VISIBLE_DEVICES -> Prefill TP=$PREFILL_TP_SIZE PP=$PREFILL_PP_SIZE"
  echo "  $CLIENT_NODE GPUs $DECODE_CUDA_VISIBLE_DEVICES -> Decode TP=$DECODE_TP_SIZE PP=$DECODE_PP_SIZE"
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export MODE="e_p_d_1node"
export CLIENT_NODE="$CLIENT_NODE"
export CLIENT_IP="$CLIENT_IP"
export PORT="$PORT"
export ENCODE_PORT="$ENCODE_PORT"
export PREFILL_PORT="$PREFILL_PORT"
export DECODE_PORT="$DECODE_PORT"
export MODEL="$MODEL"
export VLLM_ROOT="$VLLM_ROOT"
export EC_SHARED_STORAGE_PATH="$EC_SHARED_STORAGE_PATH"
export PREFILL_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT"
export DECODE_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT"
export ENCODER_CUDA_VISIBLE_DEVICES="$ENCODER_CUDA_VISIBLE_DEVICES"
export ENCODER_TP_SIZE="$ENCODER_TP_SIZE"
export PREFILL_CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_VISIBLE_DEVICES"
export PREFILL_TP_SIZE="$PREFILL_TP_SIZE"
export PREFILL_PP_SIZE="$PREFILL_PP_SIZE"
export DECODE_CUDA_VISIBLE_DEVICES="$DECODE_CUDA_VISIBLE_DEVICES"
export DECODE_TP_SIZE="$DECODE_TP_SIZE"
export DECODE_PP_SIZE="$DECODE_PP_SIZE"
export LIMIT_MM_PER_PROMPT='$LIMIT_MM_PER_PROMPT'
EOF

log "Wrote cluster env to $CLUSTER_ENV_FILE"

remote_env_prefix() {
  cat <<EOF
set -euo pipefail
source "$VENV_ACTIVATE"
cd "$VLLM_ROOT"

export HF_HOME="\${SCRATCH:-\$HOME}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="\$HF_HOME"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="\${SCRATCH:-\$HOME}/.cache/torch_inductor"
export NCCL_DEBUG="\${NCCL_DEBUG:-WARN}"
export UCX_TLS="\${UCX_TLS:-all}"
export UCX_NET_DEVICES="\${UCX_NET_DEVICES:-all}"
EOF
}

cleanup() {
  set +e
  log "Stopping E/P/D 1-node vLLM stack..."

  ssh -q "$CLIENT_NODE" "
    source '$VENV_ACTIVATE' >/dev/null 2>&1 || true
    pkill -f 'disagg_epd_proxy.py' >/dev/null 2>&1 || true
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

log "Starting encoder vLLM producer on $CLIENT_NODE GPU $ENCODER_CUDA_VISIBLE_DEVICES"

ssh -q "$CLIENT_NODE" "
$(remote_env_prefix)

CUDA_VISIBLE_DEVICES='$ENCODER_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$ENCODE_PORT' \
  --tensor-parallel-size '$ENCODER_TP_SIZE' \
  --enforce-eager \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --max-num-batched-tokens 114688 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT' \
  --ec-transfer-config '{
    \"ec_connector\":\"ECExampleConnector\",
    \"ec_role\":\"ec_producer\",
    \"ec_connector_extra_config\":{
      \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
    }
  }'
" > "$LOG_DIR/encoder.log" 2>&1 &

ENCODER_PID=$!
sleep 10

log "Starting prefill vLLM service on $CLIENT_NODE GPU $PREFILL_CUDA_VISIBLE_DEVICES"

ssh -q "$CLIENT_NODE" "
$(remote_env_prefix)

export VLLM_NIXL_SIDE_CHANNEL_HOST='$CLIENT_IP'
export VLLM_NIXL_SIDE_CHANNEL_PORT='$PREFILL_NIXL_SIDE_CHANNEL_PORT'

CUDA_VISIBLE_DEVICES='$PREFILL_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$PREFILL_PORT' \
  --tensor-parallel-size '$PREFILL_TP_SIZE' \
  --pipeline-parallel-size '$PREFILL_PP_SIZE' \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-request-id-headers \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT' \
  --ec-transfer-config '{
    \"ec_connector\":\"ECExampleConnector\",
    \"ec_role\":\"ec_consumer\",
    \"ec_connector_extra_config\":{
      \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
    }
  }' \
  --kv-transfer-config '{
    \"kv_connector\":\"NixlConnector\",
    \"kv_role\":\"kv_producer\"
  }'
" > "$LOG_DIR/prefill.log" 2>&1 &

PREFILL_PID=$!
sleep 10

log "Starting decode vLLM service on $CLIENT_NODE GPUs $DECODE_CUDA_VISIBLE_DEVICES"

ssh -q "$CLIENT_NODE" "
$(remote_env_prefix)

export VLLM_NIXL_SIDE_CHANNEL_HOST='$CLIENT_IP'
export VLLM_NIXL_SIDE_CHANNEL_PORT='$DECODE_NIXL_SIDE_CHANNEL_PORT'

CUDA_VISIBLE_DEVICES='$DECODE_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$DECODE_PORT' \
  --tensor-parallel-size '$DECODE_TP_SIZE' \
  --pipeline-parallel-size '$DECODE_PP_SIZE' \
  --enforce-eager \
  --enable-request-id-headers \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT' \
  --kv-transfer-config '{
    \"kv_connector\":\"NixlConnector\",
    \"kv_role\":\"kv_consumer\"
  }'
" > "$LOG_DIR/decode.log" 2>&1 &

DECODE_PID=$!
sleep 10

log "Starting E/P/D proxy on $CLIENT_NODE port $PORT"

ssh -q "$CLIENT_NODE" "
set -euo pipefail
source '$VENV_ACTIVATE'
cd '$VLLM_ROOT/examples/online_serving/disaggregated_encoder'

python3 disagg_epd_proxy.py \
  --host 0.0.0.0 \
  --port '$PORT' \
  --encode-servers-urls 'http://localhost:$ENCODE_PORT' \
  --prefill-servers-urls 'http://localhost:$PREFILL_PORT' \
  --decode-servers-urls 'http://localhost:$DECODE_PORT'
" > "$LOG_DIR/proxy.log" 2>&1 &

PROXY_PID=$!

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PREFILL_SHELL_PID=$PREFILL_PID"
  echo "DECODE_SHELL_PID=$DECODE_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "CLIENT_NODE=$CLIENT_NODE"
  echo "CLIENT_IP=$CLIENT_IP"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PREFILL_PORT=$PREFILL_PORT"
  echo "DECODE_PORT=$DECODE_PORT"
  echo "PORT=$PORT"
} > "$LOG_DIR/launcher_pids.txt"

log "E/P/D 1-node native vLLM stack launched. Waiting on child processes..."
wait "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID"
