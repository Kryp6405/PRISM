#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 E/P/D native vLLM run script
#
# Expected launch layout:
#   NODE0 GPUs 0,1     -> Encoder
#   NODE0 GPUs 2,3     -> Prefill
#   NODE1 GPUs 0,1,2,3 -> Decode
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

PHASE="${PHASE:-p4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-src/phase4/workloads/simple_baseline.json}"

E_P_D_LAUNCH_CMD="${E_P_D_LAUNCH_CMD:-bash scripts/phase4/launch_e_p_d_vllm.sh}"

WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-1}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"

GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-none}"
CLUSTER_GPU_TELEMETRY="${CLUSTER_GPU_TELEMETRY:-true}"
CLUSTER_GPU_TELEMETRY_INTERVAL_SEC="${CLUSTER_GPU_TELEMETRY_INTERVAL_SEC:-2}"

ENABLE_STREAMING="${ENABLE_STREAMING:-false}"
USE_LEGACY_MAX_TOKENS="${USE_LEGACY_MAX_TOKENS:-false}"
AIPERF_EXTRA_INPUTS="${AIPERF_EXTRA_INPUTS:-ignore_eos:true}"

PROXY_READY_TIMEOUT_SEC="${PROXY_READY_TIMEOUT_SEC:-1200}"

ENABLE_EPD_MULTIMODAL_WARMUP="${ENABLE_EPD_MULTIMODAL_WARMUP:-false}"
WARMUP_COMPLETION_MAX_ATTEMPTS="${WARMUP_COMPLETION_MAX_ATTEMPTS:-3}"
WARMUP_CURL_MAX_TIME="${WARMUP_CURL_MAX_TIME:-45}"
WARMUP_RETRY_SLEEP_SEC="${WARMUP_RETRY_SLEEP_SEC:-5}"

###############################################################################
# Helpers
###############################################################################

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

read_workload_field() {
  python3 - "$WORKLOAD_CONFIG" "$1" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)

if key == "concurrency_values":
    print(" ".join(map(str, data.get("concurrency_values", []))))
else:
    value = data.get(key, "")
    print(str(value).lower() if isinstance(value, bool) else value)
PY
}

###############################################################################
# Load workload
###############################################################################

if [[ ! -f "$WORKLOAD_CONFIG" ]]; then
  log "ERROR: Missing workload config: $WORKLOAD_CONFIG" >&2
  exit 1
fi

WORKLOAD_NAME="$(read_workload_field name)"
WORKLOAD_TYPE="$(read_workload_field workload_type)"
IMAGE_SOURCE="$(read_workload_field image_source)"
IMAGE_PATH="$(read_workload_field image_path)"
IMAGE_WIDTH_MEAN="$(read_workload_field image_width_mean)"
IMAGE_HEIGHT_MEAN="$(read_workload_field image_height_mean)"
REQUEST_COUNT="$(read_workload_field request_count)"
CONCURRENCY_VALUES="$(read_workload_field concurrency_values)"
PROMPT="$(read_workload_field prompt)"
NUM_IMAGES="$(read_workload_field num_images)"
ENDPOINT_TYPE="$(read_workload_field endpoint_type)"
USE_SERVER_TOKEN_COUNT="$(read_workload_field use_server_token_count)"
OUTPUT_TOKENS_MEAN="$(read_workload_field output_tokens_mean)"
OUTPUT_TOKENS_STDDEV="$(read_workload_field output_tokens_stddev)"

: "${WORKLOAD_NAME:?Missing name in $WORKLOAD_CONFIG}"
: "${WORKLOAD_TYPE:?Missing workload_type in $WORKLOAD_CONFIG}"
: "${IMAGE_SOURCE:?Missing image_source in $WORKLOAD_CONFIG}"
: "${REQUEST_COUNT:?Missing request_count in $WORKLOAD_CONFIG}"
: "${CONCURRENCY_VALUES:?Missing concurrency_values in $WORKLOAD_CONFIG}"
: "${PROMPT:?Missing prompt in $WORKLOAD_CONFIG}"
: "${NUM_IMAGES:?Missing num_images in $WORKLOAD_CONFIG}"
: "${ENDPOINT_TYPE:?Missing endpoint_type in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_MEAN:?Missing output_tokens_mean in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_STDDEV:?Missing output_tokens_stddev in $WORKLOAD_CONFIG}"

if [[ "$IMAGE_SOURCE" == "custom" ]]; then
  if [[ -z "$IMAGE_PATH" || ! -f "$IMAGE_PATH" ]]; then
    log "ERROR: Custom workload requires valid image_path, got: '$IMAGE_PATH'" >&2
    exit 1
  fi

  IMAGE_PATH="$(python3 - "$IMAGE_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
fi

###############################################################################
# Dirs
###############################################################################

RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-e-p-d-native-vllm}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

WORKLOAD_ARTIFACT_ROOT="${ARTIFACT_ROOT}/${PHASE}/${WORKLOAD_TYPE}"
RUN_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$WORKLOAD_ARTIFACT_ROOT" "$RUN_DIR" "$LOG_DIR" "$SUMMARY_DIR"

###############################################################################
# Custom JSONL helper
###############################################################################

write_custom_inputs_jsonl() {
  local input_file="$1"
  local request_count="$2"
  local prompt="$3"
  local image_path="$4"
  local output_length="$5"
  local num_images="$6"

  python3 - "$input_file" "$request_count" "$prompt" "$image_path" "$output_length" "$num_images" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
request_count = int(sys.argv[2])
prompt = sys.argv[3]
image_path = sys.argv[4]
output_length = int(sys.argv[5])
num_images = int(sys.argv[6])

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w") as f:
    for i in range(request_count):
        row = {
            "texts": [f"{prompt}\n\nRequest ID: {i}"],
            "images": [image_path for _ in range(num_images)],
            "output_length": output_length,
        }
        f.write(json.dumps(row) + "\n")
PY
}

###############################################################################
# Cleanup
###############################################################################

cleanup() {
  set +e
  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

###############################################################################
# Save configs
###############################################################################

cp "$WORKLOAD_CONFIG" "$RUN_DIR/workload.json"

cat > "$RUN_DIR/run_config.json" <<EOF
{
  "phase": "$PHASE",
  "mode": "e_p_d",
  "model": "$MODEL",
  "port": $PORT,
  "workload_config": "$WORKLOAD_CONFIG",
  "workload_name": "$WORKLOAD_NAME",
  "workload_type": "$WORKLOAD_TYPE",
  "image_source": "$IMAGE_SOURCE",
  "image_path": "$IMAGE_PATH",
  "request_count": $REQUEST_COUNT,
  "concurrency_values": "$CONCURRENCY_VALUES",
  "output_tokens_mean": $OUTPUT_TOKENS_MEAN,
  "output_tokens_stddev": $OUTPUT_TOKENS_STDDEV,
  "warmup_request_count": $WARMUP_REQUEST_COUNT,
  "warmup_concurrency": $WARMUP_CONCURRENCY,
  "gpu_telemetry_mode": "$GPU_TELEMETRY_MODE",
  "cluster_gpu_telemetry": "$CLUSTER_GPU_TELEMETRY",
  "cluster_gpu_telemetry_interval_sec": $CLUSTER_GPU_TELEMETRY_INTERVAL_SEC,
  "enable_streaming": "$ENABLE_STREAMING",
  "aiperf_extra_inputs": "$AIPERF_EXTRA_INPUTS",
  "enable_epd_multimodal_warmup": "$ENABLE_EPD_MULTIMODAL_WARMUP",
  "launch_cmd": "$E_P_D_LAUNCH_CMD"
}
EOF

###############################################################################
# Print config
###############################################################################

log "=== Phase 4 E/P/D native vLLM run ==="
log "MODEL=$MODEL"
log "WORKLOAD_CONFIG=$WORKLOAD_CONFIG"
log "WORKLOAD_TYPE=$WORKLOAD_TYPE"
log "IMAGE_SOURCE=$IMAGE_SOURCE"
log "REQUEST_COUNT=$REQUEST_COUNT"
log "CONCURRENCY_VALUES=[$CONCURRENCY_VALUES]"
log "OUTPUT_TOKENS_MEAN=$OUTPUT_TOKENS_MEAN"
log "OUTPUT_TOKENS_STDDEV=$OUTPUT_TOKENS_STDDEV"
log "RUN_DIR=$RUN_DIR"
log "LOG_DIR=$LOG_DIR"
log "GPU_TELEMETRY_MODE=$GPU_TELEMETRY_MODE"
log "ENABLE_STREAMING=$ENABLE_STREAMING"
log "AIPERF_EXTRA_INPUTS=$AIPERF_EXTRA_INPUTS"

###############################################################################
# Launch stack
###############################################################################

LOG_DIR="$LOG_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
MAX_IMAGES_PER_PROMPT="$NUM_IMAGES" \
bash -lc "$E_P_D_LAUNCH_CMD" > "$LOG_DIR/launcher.log" 2>&1 &

LAUNCHER_PID=$!
log "Launcher PID=$LAUNCHER_PID"

###############################################################################
# Wait for proxy
###############################################################################

log "Waiting for E/P/D proxy /v1/models on port $PORT, timeout=${PROXY_READY_TIMEOUT_SEC}s..."

READY=0
for _ in $(seq 1 "$PROXY_READY_TIMEOUT_SEC"); do
  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    log "ERROR: Launcher exited early. Check $LOG_DIR/launcher.log" >&2
    exit 1
  fi

  if curl -sf --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  log "ERROR: Proxy did not become reachable within ${PROXY_READY_TIMEOUT_SEC}s." >&2
  log "  Launcher log: $LOG_DIR/launcher.log" >&2
  log "  Proxy log:    $LOG_DIR/proxy.log" >&2
  exit 1
fi

log "Proxy /v1/models is reachable."

###############################################################################
# Source cluster env + backend checks
###############################################################################

CLUSTER_ENV_FILE="$LOG_DIR/cluster_env.sh"
log "Waiting for cluster_env.sh to appear..."

for _ in $(seq 1 30); do
  [[ -f "$CLUSTER_ENV_FILE" ]] && break
  sleep 2
done

if [[ -f "$CLUSTER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CLUSTER_ENV_FILE"

  log "Sourced cluster env:"
  log "  ENCODE_PORT=${ENCODE_PORT:-unset}"
  log "  PREFILL_PORT=${PREFILL_PORT:-unset}"
  log "  DECODE_IP=${DECODE_IP:-unset}"
  log "  DECODE_PORT=${DECODE_PORT:-unset}"

  log "Checking Encoder backend at http://localhost:${ENCODE_PORT}/v1/models ..."
  curl -sf --max-time 10 "http://localhost:${ENCODE_PORT}/v1/models" >/dev/null 2>&1 \
    && log "Encoder backend: OK" \
    || log "WARNING: Encoder backend not directly reachable."

  log "Checking Prefill backend at http://localhost:${PREFILL_PORT}/v1/models ..."
  curl -sf --max-time 10 "http://localhost:${PREFILL_PORT}/v1/models" >/dev/null 2>&1 \
    && log "Prefill backend: OK" \
    || log "WARNING: Prefill backend not directly reachable."

  log "Checking Decode backend at http://${DECODE_IP}:${DECODE_PORT}/v1/models ..."
  curl -sf --max-time 10 "http://${DECODE_IP}:${DECODE_PORT}/v1/models" >/dev/null 2>&1 \
    && log "Decode backend: OK" \
    || log "WARNING: Decode backend not directly reachable."
else
  log "WARNING: cluster_env.sh not found; skipping backend checks."
fi

###############################################################################
# Optional multimodal warmup
###############################################################################

if [[ "$ENABLE_EPD_MULTIMODAL_WARMUP" == "true" ]]; then
  log "Running optional multimodal warmup through proxy..."

  WARMUP_IMAGE_B64="$(python3 - <<'PY'
import base64, io
from PIL import Image
img = Image.new("RGB", (64, 64), color=(128, 128, 128))
buf = io.BytesIO()
img.save(buf, format="PNG")
print(base64.b64encode(buf.getvalue()).decode())
PY
)"

  if [[ "$IMAGE_SOURCE" == "custom" && -f "$IMAGE_PATH" ]]; then
    WARMUP_IMAGE_B64="$(base64 -w 0 "$IMAGE_PATH")"
  fi

  WARMUP_PAYLOAD="$(python3 - "$MODEL" "$WARMUP_IMAGE_B64" <<'PY'
import json, sys
model, b64 = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }],
    "max_tokens": 16
}
print(json.dumps(payload))
PY
)"

  WARMUP_OK=0
  for attempt in $(seq 1 "$WARMUP_COMPLETION_MAX_ATTEMPTS"); do
    HTTP_STATUS="$(curl -so "$LOG_DIR/epd_warmup_completion.json" \
      --max-time "$WARMUP_CURL_MAX_TIME" \
      -w "%{http_code}" \
      -H "Content-Type: application/json" \
      -d "$WARMUP_PAYLOAD" \
      "http://localhost:${PORT}/v1/chat/completions" \
      2>"$LOG_DIR/epd_warmup_completion.err" || true)"

    if [[ "$HTTP_STATUS" == "200" ]] && grep -q '"choices"' "$LOG_DIR/epd_warmup_completion.json" 2>/dev/null; then
      WARMUP_OK=1
      log "Multimodal warmup completion OK."
      break
    fi

    log "Warmup attempt ${attempt}/${WARMUP_COMPLETION_MAX_ATTEMPTS} failed with HTTP=$HTTP_STATUS."
    sleep "$WARMUP_RETRY_SLEEP_SEC"
  done

  if [[ "$WARMUP_OK" -ne 1 ]]; then
    log "ERROR: Multimodal warmup failed." >&2
    exit 1
  fi
else
  log "Skipping optional multimodal warmup."
fi

###############################################################################
# Pre-AIPerf GPU snapshot
###############################################################################

log "GPU snapshot before AIPerf sweep..."

{
  for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    ssh -q "$node" "
      host=\$(hostname)
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
      | awk -v host=\"\$host\" 'BEGIN{OFS=\",\"}{print host,\$0}'
    " 2>/dev/null || true
  done
} | tee "$LOG_DIR/live_gpu_usage_before_aiperf.csv"

###############################################################################
# Cluster GPU sampler
###############################################################################

start_cluster_gpu_sampler() {
  local out_file="$1"
  local stop_file="$2"
  local interval_sec="${3:-2}"

  rm -f "$stop_file"

  {
    while [[ ! -f "$stop_file" ]]; do
      for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        ssh -q "$node" "
          host=\$(hostname)
          nvidia-smi \
            --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
            --format=csv,noheader,nounits \
          | awk -v host=\"\$host\" 'BEGIN{OFS=\",\"}{print host,\$0}'
        " 2>/dev/null || true
      done
      sleep "$interval_sec"
    done
  } > "$out_file" 2>"${out_file}.err" &

  echo $!
}

###############################################################################
# AIPerf sweep
###############################################################################

AIPERF_OUT_DIRS=()

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_concurrency${CONCURRENCY}_${TIMESTAMP}"
  mkdir -p "$OUT_DIR"

  AIPERF_OUT_DIRS+=("$OUT_DIR")
  cp "$WORKLOAD_CONFIG" "$OUT_DIR/workload.json"

  log "=== AIPerf concurrency=$CONCURRENCY -> $OUT_DIR ==="

  if [[ "$IMAGE_SOURCE" == "custom" ]]; then
    INPUT_FILE="$OUT_DIR/inputs.jsonl"

    write_custom_inputs_jsonl \
      "$INPUT_FILE" \
      "$REQUEST_COUNT" \
      "$PROMPT" \
      "$IMAGE_PATH" \
      "$OUTPUT_TOKENS_MEAN" \
      "$NUM_IMAGES"

    AIPERF_CMD=(
      aiperf profile
      --model "$MODEL"
      --url "http://localhost:${PORT}"
      --endpoint-type "$ENDPOINT_TYPE"
      --input-file "$INPUT_FILE"
      --custom-dataset-type single_turn
      --concurrency "$CONCURRENCY"
      --request-count "$REQUEST_COUNT"
      --warmup-request-count "$WARMUP_REQUEST_COUNT"
      --warmup-concurrency "$WARMUP_CONCURRENCY"
      --use-server-token-count
      --server-metrics "http://localhost:${PORT}/metrics"
      --output-artifact-dir "$OUT_DIR"
    )
  else
    AIPERF_CMD=(
      aiperf profile
      --model "$MODEL"
      --url "http://localhost:${PORT}"
      --endpoint-type "$ENDPOINT_TYPE"
      --image-width-mean "$IMAGE_WIDTH_MEAN"
      --image-height-mean "$IMAGE_HEIGHT_MEAN"
      --concurrency "$CONCURRENCY"
      --request-count "$REQUEST_COUNT"
      --warmup-request-count "$WARMUP_REQUEST_COUNT"
      --warmup-concurrency "$WARMUP_CONCURRENCY"
      --use-server-token-count
      --prompt-output-tokens-mean "$OUTPUT_TOKENS_MEAN"
      --prompt-output-tokens-stddev "$OUTPUT_TOKENS_STDDEV"
      --server-metrics "http://localhost:${PORT}/metrics"
      --output-artifact-dir "$OUT_DIR"
    )
  fi

  if [[ -n "$AIPERF_EXTRA_INPUTS" ]]; then
    AIPERF_CMD+=(--extra-inputs "$AIPERF_EXTRA_INPUTS")
  fi

  if [[ "$GPU_TELEMETRY_MODE" == "none" ]]; then
    AIPERF_CMD+=(--no-gpu-telemetry)
  else
    AIPERF_CMD+=(--gpu-telemetry "$GPU_TELEMETRY_MODE")
  fi

  [[ "$USE_LEGACY_MAX_TOKENS" == "true" ]] && AIPERF_CMD+=(--use-legacy-max-tokens)
  [[ "$ENABLE_STREAMING" == "true" ]] && AIPERF_CMD+=(--streaming)

  printf "%q " "${AIPERF_CMD[@]}" > "$OUT_DIR/aiperf_command.txt"
  echo >> "$OUT_DIR/aiperf_command.txt"

  GPU_SAMPLER_OUT="$OUT_DIR/gpu_telemetry_all_nodes.csv"
  GPU_SAMPLER_TMP="$OUT_DIR/gpu_telemetry_all_nodes.tmp"
  GPU_SAMPLER_STOP="$OUT_DIR/gpu_sampler.stop"

  echo "host,timestamp,gpu_index,uuid,name,memory_used_mb,memory_total_mb,gpu_util_percent,power_watts,temp_c" > "$GPU_SAMPLER_OUT"

  if [[ "$CLUSTER_GPU_TELEMETRY" == "true" ]]; then
    SAMPLER_PID="$(start_cluster_gpu_sampler "$GPU_SAMPLER_TMP" "$GPU_SAMPLER_STOP" "$CLUSTER_GPU_TELEMETRY_INTERVAL_SEC")"
  else
    SAMPLER_PID=""
  fi

  set +e
  "${AIPERF_CMD[@]}" > "$LOG_DIR/aiperf_${WORKLOAD_TYPE}_c${CONCURRENCY}.log" 2>&1
  AIPERF_RC=$?
  set -e

  if [[ -n "${SAMPLER_PID:-}" ]]; then
    touch "$GPU_SAMPLER_STOP"
    wait "$SAMPLER_PID" 2>/dev/null || true
    cat "$GPU_SAMPLER_TMP" >> "$GPU_SAMPLER_OUT" || true
    rm -f "$GPU_SAMPLER_TMP"
  fi

  {
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
      ssh -q "$node" "
        host=\$(hostname)
        nvidia-smi \
          --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
          --format=csv,noheader,nounits \
        | awk -v host=\"\$host\" 'BEGIN{OFS=\",\"}{print host,\$0}'
      " 2>/dev/null || true
    done
  } | tee "$OUT_DIR/gpu_after_aiperf.csv"

  if [[ "$CLUSTER_GPU_TELEMETRY" == "true" ]]; then
    {
      echo "=== Unique GPU telemetry hosts for concurrency=$CONCURRENCY ==="
      cut -d, -f1 "$GPU_SAMPLER_OUT" | tail -n +2 | sort | uniq -c || true
      echo
      echo "=== First telemetry rows ==="
      head -20 "$GPU_SAMPLER_OUT" || true
    } | tee "$OUT_DIR/gpu_telemetry_summary.txt"
  fi

  if [[ "$AIPERF_RC" -ne 0 ]]; then
    log "ERROR: AIPerf failed for concurrency=$CONCURRENCY, rc=$AIPERF_RC." >&2
    log "  Log: $LOG_DIR/aiperf_${WORKLOAD_TYPE}_c${CONCURRENCY}.log" >&2
    exit "$AIPERF_RC"
  fi

  log "AIPerf concurrency=$CONCURRENCY complete."
  rm -f "$OUT_DIR/inputs.json"
done

###############################################################################
# Run info JSON
###############################################################################

python3 - "$RUN_DIR/run_info.json" "${AIPERF_OUT_DIRS[@]}" <<PY
import json
import sys

out = {
    "phase": "$PHASE",
    "mode": "e_p_d",
    "model": "$MODEL",
    "run_prefix": "$RUN_PREFIX",
    "timestamp": "$TIMESTAMP",
    "run_dir": "$RUN_DIR",
    "log_dir": "$LOG_DIR",
    "workload_config": "$WORKLOAD_CONFIG",
    "workload_name": "$WORKLOAD_NAME",
    "workload_type": "$WORKLOAD_TYPE",
    "image_source": "$IMAGE_SOURCE",
    "image_path": "$IMAGE_PATH",
    "request_count": int("$REQUEST_COUNT"),
    "concurrency_values": [int(x) for x in "$CONCURRENCY_VALUES".split()],
    "output_tokens_mean": int("$OUTPUT_TOKENS_MEAN"),
    "output_tokens_stddev": float("$OUTPUT_TOKENS_STDDEV"),
    "warmup_request_count": int("$WARMUP_REQUEST_COUNT"),
    "warmup_concurrency": int("$WARMUP_CONCURRENCY"),
    "gpu_telemetry_mode": "$GPU_TELEMETRY_MODE",
    "cluster_gpu_telemetry": "$CLUSTER_GPU_TELEMETRY",
    "cluster_gpu_telemetry_interval_sec": int("$CLUSTER_GPU_TELEMETRY_INTERVAL_SEC"),
    "enable_streaming": "$ENABLE_STREAMING",
    "aiperf_extra_inputs": "$AIPERF_EXTRA_INPUTS",
    "enable_epd_multimodal_warmup": "$ENABLE_EPD_MULTIMODAL_WARMUP",
    "aiperf_artifact_dirs": sys.argv[2:],
}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

###############################################################################
# Summary
###############################################################################

{
  echo "mode=e_p_d"
  echo "phase=$PHASE"
  echo "model=$MODEL"
  echo "workload_type=$WORKLOAD_TYPE"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "run_dir=$RUN_DIR"
  echo "log_dir=$LOG_DIR"
  echo "request_count=$REQUEST_COUNT"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "output_tokens_mean=$OUTPUT_TOKENS_MEAN"
  echo "warmup_request_count=$WARMUP_REQUEST_COUNT"
  echo "warmup_concurrency=$WARMUP_CONCURRENCY"
  echo "gpu_telemetry_mode=$GPU_TELEMETRY_MODE"
  echo "cluster_gpu_telemetry=$CLUSTER_GPU_TELEMETRY"
  echo "enable_streaming=$ENABLE_STREAMING"
  echo "aiperf_extra_inputs=$AIPERF_EXTRA_INPUTS"
  echo "enable_epd_multimodal_warmup=$ENABLE_EPD_MULTIMODAL_WARMUP"
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

log "=== Phase 4 E/P/D run complete ==="
log "RUN_DIR=$RUN_DIR"
