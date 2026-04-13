#!/usr/bin/env bash
set -euo pipefail

: "${LOG_DIR:?LOG_DIR must be set}"
: "${MODEL:=Qwen/Qwen2-VL-2B-Instruct}"
: "${VLLM_ROOT:?Set VLLM_ROOT, e.g. $SCRATCH/PRISM_env/vllm}"
: "${VENV_ACTIVATE:?Set VENV_ACTIVATE, e.g. $SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"
: "${EC_SHARED_STORAGE_PATH:?Set EC_SHARED_STORAGE_PATH}"
: "${PORT:=8000}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"

mkdir -p "$LOG_DIR" "$EC_SHARED_STORAGE_PATH"

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
