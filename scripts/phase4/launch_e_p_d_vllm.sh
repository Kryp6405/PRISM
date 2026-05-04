#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 E/P/D native vLLM launch script — NO RAY, NO PP
#
# Layout:
#   NODE0 GPUs 0,1     -> Encoder vLLM       (TP=2, port ENCODE_PORT)
#   NODE0 GPUs 2,3     -> Prefill vLLM       (TP=2, port PREFILL_PORT)
#   NODE1 GPUs 0,1,2,3 -> Decode vLLM        (TP=4, port DECODE_PORT)
#
# Proxy:
#   Runs on NODE0 port 8000 so AIPerf can use http://localhost:8000.
#
# Transfer:
#   Encoder -> Prefill: ECExampleConnector via shared filesystem
#   Prefill -> Decode:  NixlConnector
#
# Each role is single-node intra-NVLink. No Ray, no PP, no compiled DAG.
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"

VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache_epd}"

# NIXL side-channel ports (used by NixlConnector for KV transfer handshake).
PREFILL_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5559}"
DECODE_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_SIDE_CHANNEL_PORT:-6000}"

###############################################################################
# GPU placement and parallelism
###############################################################################

# Encoder: 2 GPUs on Node 0, TP=2.
ENCODER_CUDA_VISIBLE_DEVICES="${ENCODER_CUDA_VISIBLE_DEVICES:-0,1}"
ENCODER_TP_SIZE="${ENCODER_TP_SIZE:-2}"

# Prefill: 2 GPUs on Node 0, TP=2 (intra-node NVLink, no PP).
PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-2,3}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-1}"

# Decode: 4 GPUs on Node 1, TP=4 (intra-node NVLink, no PP).
DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-4}"
DECODE_PP_SIZE="${DECODE_PP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
MAX_VIDEOS_PER_PROMPT="${MAX_VIDEOS_PER_PROMPT:-0}"
LIMIT_MM_PER_PROMPT="{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}"

LOG_DIR="${LOG_DIR:-./artifacts/p4/e_p_d_native_vllm_logs}"
mkdir -p "$LOG_DIR"

CLUSTER_ENV_FILE="${LOG_DIR}/cluster_env.sh"

SERVICE_READY_TIMEOUT_SEC="${SERVICE_READY_TIMEOUT_SEC:-1200}"

###############################################################################
# Helpers
###############################################################################

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

wait_for_local_http() {
  local name="$1"
  local port="$2"
  local pid="$3"

  log "Waiting for $name /v1/models on localhost:$port (timeout=${SERVICE_READY_TIMEOUT_SEC}s)..."

  local ready=0
  for _ in $(seq 1 "$SERVICE_READY_TIMEOUT_SEC"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR: $name process exited early. Check logs in $LOG_DIR" >&2
      return 1
    fi

    if curl -sf --max-time 3 "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi

    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    log "ERROR: $name did not become reachable on localhost:$port." >&2
    return 1
  fi

  log "$name /v1/models is reachable."
}

wait_for_remote_http() {
  local name="$1"
  local host="$2"
  local port="$3"
  local pid="$4"

  log "Waiting for $name /v1/models on ${host}:${port} (timeout=${SERVICE_READY_TIMEOUT_SEC}s)..."

  local ready=0
  for _ in $(seq 1 "$SERVICE_READY_TIMEOUT_SEC"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR: $name process exited early. Check logs in $LOG_DIR" >&2
      return 1
    fi

    if curl -sf --max-time 3 "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi

    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    log "ERROR: $name did not become reachable on ${host}:${port}." >&2
    return 1
  fi

  log "$name /v1/models is reachable."
}

###############################################################################
# Slurm node resolution
###############################################################################

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  log "ERROR: This script expects to run inside a Slurm allocation." >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ "${#NODES[@]}" -lt 2 ]]; then
  log "ERROR: Need at least 2 nodes. Found: ${#NODES[@]}" >&2
  printf '%s\n' "${NODES[@]}" >&2
  exit 1
fi

CLIENT_NODE="${NODES[0]}"
ENCODER_NODE="${NODES[0]}"
PREFILL_NODE="${NODES[0]}"
DECODE_NODE="${NODES[1]}"

CLIENT_IP="$(ssh -q "$CLIENT_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"
DECODE_IP="$(ssh -q "$DECODE_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"
PREFILL_IP="$CLIENT_IP"   # prefill is on the same node as the client

if [[ -z "$CLIENT_IP" || -z "$DECODE_IP" ]]; then
  log "ERROR: Could not resolve CLIENT_IP or DECODE_IP." >&2
  log "CLIENT_NODE=$CLIENT_NODE CLIENT_IP=$CLIENT_IP" >&2
  log "DECODE_NODE=$DECODE_NODE DECODE_IP=$DECODE_IP" >&2
  exit 1
fi

: "${VLLM_ROOT:?VLLM_ROOT is empty}"
: "${VENV_ACTIVATE:?VENV_ACTIVATE is empty}"

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

###############################################################################
# Logging config
###############################################################################

log "=== Phase 4 E/P/D native vLLM launch ==="
log "MODEL=$MODEL"
log "PORT=$PORT"
log "ENCODE_PORT=$ENCODE_PORT"
log "PREFILL_PORT=$PREFILL_PORT"
log "DECODE_PORT=$DECODE_PORT"
log "CLIENT_NODE=$CLIENT_NODE  CLIENT_IP=$CLIENT_IP"
log "ENCODER_NODE=$ENCODER_NODE  GPUs $ENCODER_CUDA_VISIBLE_DEVICES  TP=$ENCODER_TP_SIZE"
log "PREFILL_NODE=$PREFILL_NODE  GPUs $PREFILL_CUDA_VISIBLE_DEVICES  TP=$PREFILL_TP_SIZE PP=$PREFILL_PP_SIZE"
log "DECODE_NODE=$DECODE_NODE  DECODE_IP=$DECODE_IP  GPUs $DECODE_CUDA_VISIBLE_DEVICES  TP=$DECODE_TP_SIZE PP=$DECODE_PP_SIZE"
log "EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
log "PREFILL_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP  PORT=$PREFILL_NIXL_SIDE_CHANNEL_PORT"
log "DECODE_NIXL_SIDE_CHANNEL_HOST=$DECODE_IP  PORT=$DECODE_NIXL_SIDE_CHANNEL_PORT"
log "MAX_MODEL_LEN=$MAX_MODEL_LEN  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
log "LIMIT_MM_PER_PROMPT=$LIMIT_MM_PER_PROMPT"
log "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
  echo
  echo "Layout:"
  echo "  $ENCODER_NODE  GPUs $ENCODER_CUDA_VISIBLE_DEVICES  -> Encoder  (TP=$ENCODER_TP_SIZE)"
  echo "  $PREFILL_NODE  GPUs $PREFILL_CUDA_VISIBLE_DEVICES  -> Prefill  (TP=$PREFILL_TP_SIZE PP=$PREFILL_PP_SIZE)"
  echo "  $DECODE_NODE  GPUs $DECODE_CUDA_VISIBLE_DEVICES    -> Decode   (TP=$DECODE_TP_SIZE PP=$DECODE_PP_SIZE)"
  echo
  echo "No Ray. Each role is a single-node vLLM service. NixlConnector for KV transfer."
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export MODE="e_p_d"
export CLIENT_NODE="$CLIENT_NODE"
export ENCODER_NODE="$ENCODER_NODE"
export PREFILL_NODE="$PREFILL_NODE"
export DECODE_NODE="$DECODE_NODE"
export CLIENT_IP="$CLIENT_IP"
export PREFILL_IP="$PREFILL_IP"
export DECODE_IP="$DECODE_IP"
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

###############################################################################
# Remote env helper
###############################################################################

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

###############################################################################
# Cleanup
###############################################################################

cleanup() {
  set +e
  log "Stopping E/P/D native vLLM stack..."

  for node in "$CLIENT_NODE" "$DECODE_NODE"; do
    ssh -q "$node" "
      source '$VENV_ACTIVATE' >/dev/null 2>&1 || true

      pkill -f 'disagg_epd_proxy.py'                >/dev/null 2>&1 || true
      pkill -f 'vllm serve'                         >/dev/null 2>&1 || true
      pkill -f 'vllm.entrypoints.openai.api_server' >/dev/null 2>&1 || true
      pkill -f 'VLLM::EngineCore'                   >/dev/null 2>&1 || true
      pkill -f 'RayWorkerWrapper'                   >/dev/null 2>&1 || true
      ray stop --force                              >/dev/null 2>&1 || true
    " 2>/dev/null || true
  done
}

trap cleanup EXIT

log "Cleaning existing processes..."
cleanup
sleep 5

###############################################################################
# Step 1 - Encoder on Node 0 GPUs 0,1  (TP=2)
###############################################################################

log "=== Step 1: Encoder vLLM on $ENCODER_NODE GPUs $ENCODER_CUDA_VISIBLE_DEVICES (TP=$ENCODER_TP_SIZE) ==="

ssh -q "$ENCODER_NODE" "
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
log "Encoder PID=$ENCODER_PID"

wait_for_local_http "Encoder backend" "$ENCODE_PORT" "$ENCODER_PID"

###############################################################################
# Step 2 - Prefill on Node 0 GPUs 2,3  (TP=2, KV producer)
###############################################################################

log "=== Step 2: Prefill vLLM on $PREFILL_NODE GPUs $PREFILL_CUDA_VISIBLE_DEVICES (TP=$PREFILL_TP_SIZE PP=$PREFILL_PP_SIZE) ==="

ssh -q "$PREFILL_NODE" "
$(remote_env_prefix)

# NIXL side-channel: prefill advertises its own host:port so decode can reach it.
export VLLM_NIXL_SIDE_CHANNEL_HOST='$PREFILL_IP'
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
log "Prefill PID=$PREFILL_PID"

wait_for_local_http "Prefill backend" "$PREFILL_PORT" "$PREFILL_PID"

###############################################################################
# Step 3 - Decode on Node 1 GPUs 0,1,2,3  (TP=4, KV consumer)
###############################################################################

log "=== Step 3: Decode vLLM on $DECODE_NODE GPUs $DECODE_CUDA_VISIBLE_DEVICES (TP=$DECODE_TP_SIZE PP=$DECODE_PP_SIZE) ==="

ssh -q "$DECODE_NODE" "
$(remote_env_prefix)

# NIXL side-channel: decode advertises its own host:port too.
export VLLM_NIXL_SIDE_CHANNEL_HOST='$DECODE_IP'
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
log "Decode PID=$DECODE_PID"

wait_for_remote_http "Decode backend" "$DECODE_IP" "$DECODE_PORT" "$DECODE_PID"

###############################################################################
# Step 4 - Proxy on Node 0 port PORT
###############################################################################

log "=== Step 4: E/P/D proxy on $CLIENT_NODE port $PORT ==="

ssh -q "$CLIENT_NODE" "
set -euo pipefail
source '$VENV_ACTIVATE'
cd '$VLLM_ROOT/examples/online_serving/disaggregated_encoder'

python3 disagg_epd_proxy.py \
  --host 0.0.0.0 \
  --port '$PORT' \
  --encode-servers-urls 'http://localhost:$ENCODE_PORT' \
  --prefill-servers-urls 'http://localhost:$PREFILL_PORT' \
  --decode-servers-urls 'http://$DECODE_IP:$DECODE_PORT'
" > "$LOG_DIR/proxy.log" 2>&1 &

PROXY_PID=$!
log "Proxy PID=$PROXY_PID"

wait_for_local_http "E/P/D proxy" "$PORT" "$PROXY_PID"

###############################################################################
# Save PIDs / metadata
###############################################################################

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PREFILL_SHELL_PID=$PREFILL_PID"
  echo "DECODE_SHELL_PID=$DECODE_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "CLIENT_NODE=$CLIENT_NODE"
  echo "ENCODER_NODE=$ENCODER_NODE"
  echo "PREFILL_NODE=$PREFILL_NODE"
  echo "DECODE_NODE=$DECODE_NODE"
  echo "CLIENT_IP=$CLIENT_IP"
  echo "PREFILL_IP=$PREFILL_IP"
  echo "DECODE_IP=$DECODE_IP"
  echo "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
  echo "ENCODER_TP_SIZE=$ENCODER_TP_SIZE"
  echo "PREFILL_CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES"
  echo "PREFILL_TP_SIZE=$PREFILL_TP_SIZE"
  echo "PREFILL_PP_SIZE=$PREFILL_PP_SIZE"
  echo "DECODE_CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES"
  echo "DECODE_TP_SIZE=$DECODE_TP_SIZE"
  echo "DECODE_PP_SIZE=$DECODE_PP_SIZE"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PREFILL_PORT=$PREFILL_PORT"
  echo "DECODE_PORT=$DECODE_PORT"
  echo "PORT=$PORT"
  echo "PREFILL_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_SIDE_CHANNEL_PORT"
  echo "DECODE_NIXL_SIDE_CHANNEL_PORT=$DECODE_NIXL_SIDE_CHANNEL_PORT"
} > "$LOG_DIR/launcher_pids.txt"

log "=== E/P/D native vLLM stack launched. Tailing children. ==="
wait "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID"
