#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Perlmutter defaults — edit these once if your paths differ
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"

VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"

# GPU placement
# Full E/P/D scale-out:
#   GPU_E -> encoder
#   GPU_P -> prefill
#   GPU_D -> decode
GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

LOG_DIR="${LOG_DIR:-./artifacts/e_p_d_logs}"

###############################################################################
mkdir -p "$LOG_DIR"
rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

: "${VLLM_ROOT:?VLLM_ROOT is empty}"
: "${VENV_ACTIVATE:?VENV_ACTIVATE is empty}"

capture_gpu_audit() {
  local tag="$1"
  local out_file="$LOG_DIR/gpu_audit_${tag}.txt"

  {
    echo "=== GPU audit: $tag ==="
    echo "timestamp=$(date)"
    echo "hostname=$(hostname)"
    echo "USER=${USER:-unknown}"
    echo "OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo

    echo "=== Intended script GPU mapping ==="
    echo "GPU_E=${GPU_E:-unset}"
    echo "GPU_PD=${GPU_PD:-unset}"
    echo "GPU_P=${GPU_P:-unset}"
    echo "GPU_D=${GPU_D:-unset}"
    echo

    echo "=== nvidia-smi full table ==="
    nvidia-smi || true
    echo

    echo "=== GPU index / UUID map ==="
    nvidia-smi --query-gpu=index,uuid,name,bus_id,memory.used,power.draw,utilization.gpu \
      --format=csv || true
    echo

    echo "=== Compute apps by GPU UUID/PID ==="
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv || true
    echo

    echo "=== User vLLM/Dynamo/disagg processes ==="
    ps -u "$USER" -f | grep -E "vllm|dynamo|disagg|EngineCore" | grep -v grep || true
  } > "$out_file" 2>&1
}

echo "Launching full E/P/D with:"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  ENCODE_PORT=$ENCODE_PORT"
echo "  PREFILL_PORT=$PREFILL_PORT"
echo "  DECODE_PORT=$DECODE_PORT"
echo "  GPU_E=$GPU_E"
echo "  GPU_P=$GPU_P"
echo "  GPU_D=$GPU_D"
echo "  LOG_DIR=$LOG_DIR"
echo "  EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
echo "  OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

capture_gpu_audit "before_launch"

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"
  export UCX_TLS=all
  export UCX_NET_DEVICES=all

  echo "Starting encoder worker on physical GPU $GPU_E"
  CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config "{
      \"ec_connector\":\"ECExampleConnector\",
      \"ec_role\":\"ec_producer\",
      \"ec_connector_extra_config\":{
        \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
      }
    }"
) > "$LOG_DIR/encoder.log" 2>&1 &
ENCODER_PID=$!

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"
  export UCX_TLS=all
  export UCX_NET_DEVICES=all
  export VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5559}"

  echo "Starting prefill worker on physical GPU $GPU_P"
  CUDA_VISIBLE_DEVICES="$GPU_P" vllm serve "$MODEL" \
    --port "$PREFILL_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config "{
      \"ec_connector\":\"ECExampleConnector\",
      \"ec_role\":\"ec_consumer\",
      \"ec_connector_extra_config\":{
        \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
      }
    }" \
    --kv-transfer-config '{
      "kv_connector":"NixlConnector",
      "kv_role":"kv_producer"
    }'
) > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"
  export UCX_TLS=all
  export UCX_NET_DEVICES=all
  export VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_SIDE_CHANNEL_PORT:-6000}"

  echo "Starting decode worker on physical GPU $GPU_D"
  CUDA_VISIBLE_DEVICES="$GPU_D" vllm serve "$MODEL" \
    --port "$DECODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --kv-transfer-config '{
      "kv_connector":"NixlConnector",
      "kv_role":"kv_consumer"
    }'
) > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT/examples/online_serving/disaggregated_encoder"

  echo "Starting E/P/D proxy on port $PORT"
  python3 disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT"
) > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PREFILL_SHELL_PID=$PREFILL_PID"
  echo "DECODE_SHELL_PID=$DECODE_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "GPU_E=$GPU_E"
  echo "GPU_P=$GPU_P"
  echo "GPU_D=$GPU_D"
  echo "OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PREFILL_PORT=$PREFILL_PORT"
  echo "DECODE_PORT=$DECODE_PORT"
  echo "PORT=$PORT"
  echo "PREFILL_NIXL_SIDE_CHANNEL_PORT=${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5559}"
  echo "DECODE_NIXL_SIDE_CHANNEL_PORT=${DECODE_NIXL_SIDE_CHANNEL_PORT:-6000}"
} > "$LOG_DIR/launcher_pids.txt"

sleep 10
capture_gpu_audit "after_launch"

cleanup() {
  set +e
  kill "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID" 2>/dev/null || true
  wait "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID"
