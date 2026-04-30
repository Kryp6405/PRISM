#!/usr/bin/env bash
set -euo pipefail

# Phase 3 E/PD runner.
#
# E/PD:
#   GPU_E  -> Encoder
#   GPU_PD -> Prefill + Decode
#
# Artifacts:
#   artifacts/p3/<workload_type>/<run dirs>

ENCODER_ONLY_LAUNCH_CMD="${ENCODER_ONLY_LAUNCH_CMD:-bash scripts/launch_encoder_partial_perlmutter.sh}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"

WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-src/phase3/workloads/simple_baseline.json}"

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
    if isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)
PY
}

if [[ ! -f "$WORKLOAD_CONFIG" ]]; then
  echo "Missing workload config: $WORKLOAD_CONFIG" >&2
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
: "${IMAGE_WIDTH_MEAN:?Missing image_width_mean in $WORKLOAD_CONFIG}"
: "${IMAGE_HEIGHT_MEAN:?Missing image_height_mean in $WORKLOAD_CONFIG}"
: "${REQUEST_COUNT:?Missing request_count in $WORKLOAD_CONFIG}"
: "${CONCURRENCY_VALUES:?Missing concurrency_values in $WORKLOAD_CONFIG}"
: "${PROMPT:?Missing prompt in $WORKLOAD_CONFIG}"
: "${NUM_IMAGES:?Missing num_images in $WORKLOAD_CONFIG}"
: "${ENDPOINT_TYPE:?Missing endpoint_type in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_MEAN:?Missing output_tokens_mean in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_STDDEV:?Missing output_tokens_stddev in $WORKLOAD_CONFIG}"

USE_LEGACY_MAX_TOKENS="${USE_LEGACY_MAX_TOKENS:-false}"
GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-pynvml}"
ENABLE_STREAMING="${ENABLE_STREAMING:-false}"

PHASE="${PHASE:-p3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
COMPARISON_FRAMING="${COMPARISON_FRAMING:-stage_disaggregated_scaleout}"
GPU_BUDGET="${GPU_BUDGET:-2}"

GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-encoder-only-openai-chat}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORKLOAD_ARTIFACT_ROOT="${ARTIFACT_ROOT}/${PHASE}/${WORKLOAD_TYPE}"
RUN_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$WORKLOAD_ARTIFACT_ROOT" "$RUN_DIR" "$LOG_DIR" "$ENV_DIR" "$SUMMARY_DIR"

cleanup() {
  set +e

  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi

  pkill -P $$ python 2>/dev/null || true
  pkill -P $$ python3 2>/dev/null || true
}
trap cleanup EXIT

capture_gpu_audit() {
  local tag="$1"
  local out_file="$RUN_DIR/gpu_process_mapping_${tag}.txt"

  {
    echo "=== GPU audit: $tag ==="
    echo "timestamp=$(date)"
    echo "hostname=$(hostname)"
    echo "USER=${USER:-unknown}"
    echo "OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "GPU_E=${GPU_E}"
    echo "GPU_PD=${GPU_PD}"
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
  } > "$out_file" 2>&1 || true
}

write_custom_inputs_jsonl() {
  local input_file="$1"
  local request_count="$2"
  local prompt="$3"
  local image_path="$4"
  local output_length="$5"

  python3 - "$input_file" "$request_count" "$prompt" "$image_path" "$output_length" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
request_count = int(sys.argv[2])
prompt = sys.argv[3]
image_path = sys.argv[4]
output_length = int(sys.argv[5])

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w") as f:
    for i in range(request_count):
        # Include a tiny request index so requests are not byte-identical.
        # This helps avoid accidental cache artifacts while keeping the task identical.
        row = {
            "texts": [f"{prompt}\n\nRequest ID: {i}"],
            "images": [image_path],
            "output_length": output_length
        }
        f.write(json.dumps(row) + "\n")
PY
}

[[ -f src/phase3/run_manifest_template.json ]] && \
  cp src/phase3/run_manifest_template.json "$RUN_DIR/run_manifest.json"

cp "$WORKLOAD_CONFIG" "$RUN_DIR/workload.json"

if [[ -n "$IMAGE_PATH" && -f "$IMAGE_PATH" ]]; then
  mkdir -p "$RUN_DIR/input_assets"
  cp "$IMAGE_PATH" "$RUN_DIR/input_assets/" || true
fi

if [[ -x scripts/capture_env.sh ]]; then
  bash scripts/capture_env.sh "$ENV_DIR" > "$LOG_DIR/capture_env.log" 2>&1 || true
else
  echo "scripts/capture_env.sh not found or not executable; skipping env capture" \
    > "$LOG_DIR/capture_env.log"
fi

export PHASE COMPARISON_FRAMING GPU_BUDGET GPU_E GPU_PD
export MODEL PORT ENCODER_ONLY_LAUNCH_CMD
export WORKLOAD_CONFIG WORKLOAD_NAME WORKLOAD_TYPE IMAGE_SOURCE IMAGE_PATH
export IMAGE_WIDTH_MEAN IMAGE_HEIGHT_MEAN PROMPT NUM_IMAGES

python3 - "$RUN_DIR/launch_config.json" <<'PY'
import json
import os
import sys

out = {
    "mode": "encoder_only",
    "phase": os.environ["PHASE"],
    "comparison_framing": os.environ["COMPARISON_FRAMING"],
    "gpu_budget": int(os.environ["GPU_BUDGET"]),
    "gpu_mapping": {
        f"gpu{os.environ['GPU_E']}": "encoder",
        f"gpu{os.environ['GPU_PD']}": "prefill_decode"
    },
    "model": os.environ["MODEL"],
    "port": int(os.environ["PORT"]),
    "launcher": os.environ["ENCODER_ONLY_LAUNCH_CMD"],
    "workload_config": os.environ["WORKLOAD_CONFIG"],
    "workload_name": os.environ["WORKLOAD_NAME"],
    "workload_type": os.environ["WORKLOAD_TYPE"],
    "image_source": os.environ["IMAGE_SOURCE"],
    "image_path": os.environ["IMAGE_PATH"],
    "image_width_mean": int(os.environ["IMAGE_WIDTH_MEAN"]),
    "image_height_mean": int(os.environ["IMAGE_HEIGHT_MEAN"]),
    "prompt": os.environ["PROMPT"],
    "enable_multimodal": True,
    "limit_mm_per_prompt": {
        "image": int(os.environ["NUM_IMAGES"])
    }
}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

echo "Loaded workload:"
echo "  WORKLOAD_CONFIG=$WORKLOAD_CONFIG"
echo "  WORKLOAD_NAME=$WORKLOAD_NAME"
echo "  WORKLOAD_TYPE=$WORKLOAD_TYPE"
echo "  IMAGE_SOURCE=$IMAGE_SOURCE"
echo "  IMAGE_PATH=$IMAGE_PATH"
echo "  IMAGE_WIDTH_MEAN=$IMAGE_WIDTH_MEAN"
echo "  IMAGE_HEIGHT_MEAN=$IMAGE_HEIGHT_MEAN"
echo "  REQUEST_COUNT=$REQUEST_COUNT"
echo "  CONCURRENCY_VALUES=$CONCURRENCY_VALUES"
echo "  OUTPUT_TOKENS_MEAN=$OUTPUT_TOKENS_MEAN"
echo "  OUTPUT_TOKENS_STDDEV=$OUTPUT_TOKENS_STDDEV"
echo "  NUM_IMAGES=$NUM_IMAGES"
echo "  ENDPOINT_TYPE=$ENDPOINT_TYPE"
echo
echo "Launching encoder-disaggregated E/PD stack:"
echo "  RUN_DIR=$RUN_DIR"
echo "  LOG_DIR=$LOG_DIR"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  GPU_E=$GPU_E"
echo "  GPU_PD=$GPU_PD"
echo "  GPU_BUDGET=$GPU_BUDGET"
echo "  OUTER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

capture_gpu_audit "before_launch"

LOG_DIR="$LOG_DIR" \
PORT="$PORT" \
MODEL="$MODEL" \
GPU_E="$GPU_E" \
GPU_PD="$GPU_PD" \
bash -lc "$ENCODER_ONLY_LAUNCH_CMD" \
  > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Launcher PID: $LAUNCHER_PID"
echo "Waiting for encoder/decode/proxy readiness..."

READY=0
for i in $(seq 1 240); do
  if (( i % 10 == 0 )); then
    echo "still waiting... $(date)"
  fi

  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "Launcher exited early; inspect $LOG_DIR/launcher.log" | tee "$RUN_DIR/summary.txt"
    exit 1
  fi

  encoder_ready=0
  decode_ready=0
  proxy_ready=0

  if grep -q "Application startup complete" "$LOG_DIR/encoder.log" 2>/dev/null || \
     grep -q "Uvicorn running on" "$LOG_DIR/encoder.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/encoder.log" 2>/dev/null; then
    encoder_ready=1
  fi

  if grep -q "Application startup complete" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "Registered endpoint" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/decode.log" 2>/dev/null; then
    decode_ready=1
  fi

  if grep -q "Uvicorn running on" "$LOG_DIR/proxy.log" 2>/dev/null || \
     grep -q "Application startup complete" "$LOG_DIR/proxy.log" 2>/dev/null || \
     grep -q "Listening on" "$LOG_DIR/proxy.log" 2>/dev/null; then
    proxy_ready=1
  fi

  if [[ "$encoder_ready" -eq 1 && "$decode_ready" -eq 1 && "$proxy_ready" -eq 1 ]]; then
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
  fi

  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  cat > "$RUN_DIR/summary.txt" <<EOF
Encoder-only E/PD stack failed to become ready.

Inspect:
  $LOG_DIR/launcher.log
  $LOG_DIR/encoder.log
  $LOG_DIR/decode.log
  $LOG_DIR/proxy.log
EOF
  exit 1
fi

sleep 10
capture_gpu_audit "after_ready"

echo "Encoder-only E/PD stack is ready. Starting AIPerf sweep..."

AIPERF_OUT_DIRS=()

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_concurrency${CONCURRENCY}_${TIMESTAMP}"
  AIPERF_OUT_DIRS+=("$OUT_DIR")
  mkdir -p "$OUT_DIR"

  cp "$WORKLOAD_CONFIG" "$OUT_DIR/workload.json"

  echo "Running AIPerf:"
  echo "  workload=$WORKLOAD_TYPE"
  echo "  output_tokens_mean=$OUTPUT_TOKENS_MEAN"
  echo "  concurrency=$CONCURRENCY"
  echo "  out_dir=$OUT_DIR"

  if [[ "$IMAGE_SOURCE" == "custom" ]]; then
    INPUT_FILE="$OUT_DIR/inputs.jsonl"

    write_custom_inputs_jsonl \
      "$INPUT_FILE" \
      "$REQUEST_COUNT" \
      "$PROMPT" \
      "$IMAGE_PATH" \
      "$OUTPUT_TOKENS_MEAN"

    AIPERF_CMD=(
      aiperf profile
      --model "$MODEL"
      --url "http://localhost:${PORT}"
      --endpoint-type "$ENDPOINT_TYPE"
      --input-file "$INPUT_FILE"
      --custom-dataset-type single_turn
      --concurrency "$CONCURRENCY"
      --request-count "$REQUEST_COUNT"
      --use-server-token-count
      --gpu-telemetry "$GPU_TELEMETRY_MODE"
      --output-artifact-dir "$OUT_DIR"
      --server-metrics "http://localhost:${PORT}/metrics"
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
      --use-server-token-count
      --prompt-output-tokens-mean "$OUTPUT_TOKENS_MEAN"
      --prompt-output-tokens-stddev "$OUTPUT_TOKENS_STDDEV"
      --gpu-telemetry "$GPU_TELEMETRY_MODE"
      --output-artifact-dir "$OUT_DIR"
      --server-metrics "http://localhost:${PORT}/metrics"
    )
  fi

  if [[ "$USE_LEGACY_MAX_TOKENS" == "true" ]]; then
    AIPERF_CMD+=(--use-legacy-max-tokens)
  fi

  if [[ "$ENABLE_STREAMING" == "true" ]]; then
    AIPERF_CMD+=(--streaming)
  fi

  printf "%q " "${AIPERF_CMD[@]}" > "$OUT_DIR/aiperf_command.txt"
  echo >> "$OUT_DIR/aiperf_command.txt"

  "${AIPERF_CMD[@]}" > "$LOG_DIR/aiperf_${WORKLOAD_TYPE}_c${CONCURRENCY}.log" 2>&1

  # AIPerf writes inputs.json with base64-expanded image payloads.
  # It is huge and not needed for our analysis because we keep inputs.jsonl,
  # workload.json, aiperf_command.txt, and the source image path.
  rm -f "$OUT_DIR/inputs.json"
done

export RUN_PREFIX TIMESTAMP REQUEST_COUNT CONCURRENCY_VALUES
export OUTPUT_TOKENS_MEAN OUTPUT_TOKENS_STDDEV GPU_TELEMETRY_MODE ENABLE_STREAMING ENDPOINT_TYPE
export USE_SERVER_TOKEN_COUNT RUN_DIR LOG_DIR

python3 - "$RUN_DIR/run_info.json" "${AIPERF_OUT_DIRS[@]}" <<'PY'
import json
import os
import sys

out = {
    "mode": "encoder_only",
    "phase": os.environ["PHASE"],
    "comparison_framing": os.environ["COMPARISON_FRAMING"],
    "gpu_budget": int(os.environ["GPU_BUDGET"]),
    "gpu_mapping": {
        f"gpu{os.environ['GPU_E']}": "encoder",
        f"gpu{os.environ['GPU_PD']}": "prefill_decode"
    },
    "model": os.environ["MODEL"],
    "workload_config": os.environ["WORKLOAD_CONFIG"],
    "workload_name": os.environ["WORKLOAD_NAME"],
    "workload_type": os.environ["WORKLOAD_TYPE"],
    "image_source": os.environ["IMAGE_SOURCE"],
    "image_path": os.environ["IMAGE_PATH"],
    "image_width_mean": int(os.environ["IMAGE_WIDTH_MEAN"]),
    "image_height_mean": int(os.environ["IMAGE_HEIGHT_MEAN"]),
    "prompt": os.environ["PROMPT"],
    "num_images": int(os.environ["NUM_IMAGES"]),
    "custom_dataset_type": "single_turn" if os.environ["IMAGE_SOURCE"] == "custom" else None,
    "input_file_mode": "custom_jsonl" if os.environ["IMAGE_SOURCE"] == "custom" else "synthetic_aiperf",
    "run_prefix": os.environ["RUN_PREFIX"],
    "timestamp": os.environ["TIMESTAMP"],
    "run_dir": os.environ["RUN_DIR"],
    "log_dir": os.environ["LOG_DIR"],
    "port": int(os.environ["PORT"]),
    "request_count": int(os.environ["REQUEST_COUNT"]),
    "concurrency_values": [int(x) for x in os.environ["CONCURRENCY_VALUES"].split()],
    "output_tokens_mean": int(os.environ["OUTPUT_TOKENS_MEAN"]),
    "output_tokens_stddev": float(os.environ["OUTPUT_TOKENS_STDDEV"]),
    "gpu_telemetry_mode": os.environ["GPU_TELEMETRY_MODE"],
    "enable_streaming": os.environ["ENABLE_STREAMING"] == "true",
    "endpoint_type": os.environ["ENDPOINT_TYPE"],
    "use_server_token_count": os.environ["USE_SERVER_TOKEN_COUNT"] == "true",
    "launcher": os.environ["ENCODER_ONLY_LAUNCH_CMD"],
    "encoder_gpu": os.environ["GPU_E"],
    "pd_gpu": os.environ["GPU_PD"],
    "aiperf_artifact_dirs": sys.argv[2:]
}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

capture_gpu_audit "after_benchmark"

if [[ -f src/phase3/summarize_run.py ]]; then
  if python3 src/phase3/summarize_run.py \
      --artifact-root "$WORKLOAD_ARTIFACT_ROOT" \
      --run-prefix "$RUN_PREFIX" \
      > "$SUMMARY_DIR/summary.json" 2> "$LOG_DIR/summarize_run.err"; then
    echo "Wrote summary to $SUMMARY_DIR/summary.json"
  else
    echo "summarize_run.py failed; see $LOG_DIR/summarize_run.err" | tee -a "$RUN_DIR/summary.txt"
  fi
else
  echo "src/phase3/summarize_run.py not found" | tee -a "$RUN_DIR/summary.txt"
fi

{
  echo "mode=encoder_only"
  echo "phase=$PHASE"
  echo "comparison_framing=$COMPARISON_FRAMING"
  echo "gpu_budget=$GPU_BUDGET"
  echo "gpu_mapping=gpu${GPU_E}:encoder,gpu${GPU_PD}:prefill_decode"
  echo "model=$MODEL"
  echo "workload_config=$WORKLOAD_CONFIG"
  echo "workload_name=$WORKLOAD_NAME"
  echo "workload_type=$WORKLOAD_TYPE"
  echo "image_source=$IMAGE_SOURCE"
  echo "image_path=$IMAGE_PATH"
  echo "image_width_mean=$IMAGE_WIDTH_MEAN"
  echo "image_height_mean=$IMAGE_HEIGHT_MEAN"
  echo "prompt=$PROMPT"
  echo "num_images=$NUM_IMAGES"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "launcher_pid=$LAUNCHER_PID"
  echo "port=$PORT"
  echo "request_count=$REQUEST_COUNT"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "output_tokens_mean=$OUTPUT_TOKENS_MEAN"
  echo "output_tokens_stddev=$OUTPUT_TOKENS_STDDEV"
  echo "gpu_telemetry_mode=$GPU_TELEMETRY_MODE"
  echo "enable_streaming=$ENABLE_STREAMING"
  echo "endpoint_type=$ENDPOINT_TYPE"
  echo "use_server_token_count=$USE_SERVER_TOKEN_COUNT"
  echo "run_dir=$RUN_DIR"
  echo "log_dir=$LOG_DIR"
  echo "summary_dir=$SUMMARY_DIR"
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

echo "Phase 3 encoder-only E/PD run complete."
echo "RUN_DIR=$RUN_DIR"
