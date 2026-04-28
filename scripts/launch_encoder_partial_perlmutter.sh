#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Perlmutter defaults — edit these once if your paths differ
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"

# Your built vLLM checkout from setup_prism_stack.sh
VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"

# Your venv from setup_prism_stack.sh
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

# Shared EC cache path
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache}"

# Ports
ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"

# GPU placement
# E/PD scale-out:
#   GPU_E  -> encoder
#   GPU_PD -> combined prefill/decode
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

# Logs
LOG_DIR="${LOG_DIR:-./artifacts/p0}"

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

echo "Launching E/PD with:"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  ENCODE_PORT=$ENCODE_PORT"
echo "  PD_PORT=$PD_PORT"
echo "  GPU_E=$GPU_E"
echo "  GPU_PD=$GPU_PD"
echo "  LOG_DIR=$LOG_DIR"
echo "  EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
echo "  OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

capture_gpu_audit "before_launch"

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"

  echo "Starting encoder worker on physical GPU $GPU_E"
  CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.01 \
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

  echo "Starting P/D worker on physical GPU $GPU_PD"
  CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PD_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config "{
      \"ec_connector\":\"ECExampleConnector\",
      \"ec_role\":\"ec_consumer\",
      \"ec_connector_extra_config\":{
        \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
      }
    }"
) > "$LOG_DIR/decode.log" 2>&1 &

PD_PID=$!

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT/examples/online_serving/disaggregated_encoder"

  echo "Starting E/PD proxy on port $PORT"
  python3 disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PD_PORT"
) > "$LOG_DIR/proxy.log" 2>&1 &

PROXY_PID=$!

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PD_SHELL_PID=$PD_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "GPU_E=$GPU_E"
  echo "GPU_PD=$GPU_PD"
  echo "OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PD_PORT=$PD_PORT"
  echo "PORT=$PORT"
} > "$LOG_DIR/launcher_pids.txt"

sleep 10
capture_gpu_audit "after_launch"

cleanup() {
  set +e
  kill "$ENCODER_PID" "$PD_PID" "$PROXY_PID" 2>/dev/null || true
  wait "$ENCODER_PID" "$PD_PID" "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait "$ENCODER_PID" "$PD_PID" "$PROXY_PID"
