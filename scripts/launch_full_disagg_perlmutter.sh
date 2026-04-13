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

(
  source "$VENV_ACTIVATE"
  cd "$VLLM_ROOT"
  export UCX_TLS=all
  export UCX_NET_DEVICES=all

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
  export UCX_TLS=all
  export UCX_NET_DEVICES=all
  export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

  CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_PORT" \
    --enforce-eager \
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
  export VLLM_NIXL_SIDE_CHANNEL_PORT=6000

  CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$DECODE_PORT" \
    --enforce-eager \
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
  python disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT"
) > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!

wait "$ENCODER_PID" "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID"
