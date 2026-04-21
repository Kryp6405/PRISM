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

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"
  CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
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
  CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" \
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

DECODE_PID=$!

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT/examples/online_serving/disaggregated_encoder"
  python disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PD_PORT"
) > "$LOG_DIR/proxy.log" 2>&1 &

PROXY_PID=$!

wait "$ENCODER_PID" "$DECODE_PID" "$PROXY_PID"
