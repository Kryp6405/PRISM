#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 aggregated native vLLM run script.
#
# Runs:
#   native vLLM aggregated baseline
#   TP=4, PP=2 across 2 nodes
#   AIPerf workload sweep with warmup
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

PHASE="${PHASE:-p4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-src/phase4/workloads/simple_baseline.json}"

AGG_LAUNCH_CMD="${AGG_LAUNCH_CMD:-bash scripts/phase4/launch_aggregated_vllm.sh}"

WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-10}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"

GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-pynvml}"
CLUSTER_GPU_TELEMETRY="${CLUSTER_GPU_TELEMETRY:-true}"
CLUSTER_GPU_TELEMETRY_INTERVAL_SEC="${CLUSTER_GPU_TELEMETRY_INTERVAL_SEC:-2}"

ENABLE_STREAMING="${ENABLE_STREAMING:-true}"
USE_LEGACY_MAX_TOKENS="${USE_LEGACY_MAX_TOKENS:-false}"

REQUEST_TIMEOUT_READY_SEC="${REQUEST_TIMEOUT_READY_SEC:-900}"

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
: "${REQUEST_COUNT:?Missing request_count in $WORKLOAD_CONFIG}"
: "${CONCURRENCY_VALUES:?Missing concurrency_values in $WORKLOAD_CONFIG}"
: "${PROMPT:?Missing prompt in $WORKLOAD_CONFIG}"
: "${NUM_IMAGES:?Missing num_images in $WORKLOAD_CONFIG}"
: "${ENDPOINT_TYPE:?Missing endpoint_type in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_MEAN:?Missing output_tokens_mean in $WORKLOAD_CONFIG}"
: "${OUTPUT_TOKENS_STDDEV:?Missing output_tokens_stddev in $WORKLOAD_CONFIG}"

if [[ "$IMAGE_SOURCE" == "custom" ]]; then
  if [[ -z "$IMAGE_PATH" || ! -f "$IMAGE_PATH" ]]; then
    echo "Custom workload requires valid image_path, got: $IMAGE_PATH" >&2
    exit 1
  fi

  IMAGE_PATH="$(python3 - "$IMAGE_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
fi

RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-aggregated-native-vllm}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

WORKLOAD_ARTIFACT_ROOT="${ARTIFACT_ROOT}/${PHASE}/${WORKLOAD_TYPE}"
RUN_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$WORKLOAD_ARTIFACT_ROOT" "$RUN_DIR" "$LOG_DIR" "$SUMMARY_DIR"

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
        row = {
            "texts": [f"{prompt}\n\nRequest ID: {i}"],
            "images": [image_path],
            "output_length": output_length,
        }
        f.write(json.dumps(row) + "\n")
PY
}

cleanup() {
  set +e
  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cp "$WORKLOAD_CONFIG" "$RUN_DIR/workload.json"

cat > "$RUN_DIR/run_config.json" <<EOF
{
  "phase": "$PHASE",
  "mode": "aggregated_native_vllm",
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
  "launch_cmd": "$AGG_LAUNCH_CMD"
}
EOF

echo "Phase 4 aggregated native vLLM run"
echo "MODEL=$MODEL"
echo "WORKLOAD_CONFIG=$WORKLOAD_CONFIG"
echo "WORKLOAD_TYPE=$WORKLOAD_TYPE"
echo "IMAGE_SOURCE=$IMAGE_SOURCE"
echo "IMAGE_PATH=$IMAGE_PATH"
echo "REQUEST_COUNT=$REQUEST_COUNT"
echo "CONCURRENCY_VALUES=$CONCURRENCY_VALUES"
echo "OUTPUT_TOKENS_MEAN=$OUTPUT_TOKENS_MEAN"
echo "WARMUP_REQUEST_COUNT=$WARMUP_REQUEST_COUNT"
echo "WARMUP_CONCURRENCY=$WARMUP_CONCURRENCY"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "GPU_TELEMETRY_MODE=$GPU_TELEMETRY_MODE"
echo "CLUSTER_GPU_TELEMETRY=$CLUSTER_GPU_TELEMETRY"
echo "CLUSTER_GPU_TELEMETRY_INTERVAL_SEC=$CLUSTER_GPU_TELEMETRY_INTERVAL_SEC"

if [[ "$GPU_TELEMETRY_MODE" == "pynvml" ]]; then
  echo "WARNING: AIPerf pynvml telemetry only sees GPUs local to the AIPerf process."
  echo "WARNING: For 2-node runs, use gpu_telemetry_all_nodes.csv for 8-GPU telemetry."
fi

LOG_DIR="$LOG_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
bash -lc "$AGG_LAUNCH_CMD" > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Launcher PID: $LAUNCHER_PID"
echo "Waiting for vLLM readiness on port $PORT..."

READY=0
for _ in $(seq 1 "$REQUEST_TIMEOUT_READY_SEC"); do
  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "Launcher exited early. Inspect $LOG_DIR/launcher.log and $LOG_DIR/vllm_serve.log" >&2
    exit 1
  fi

  if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "vLLM did not become ready. Inspect logs in $LOG_DIR" >&2
  exit 1
fi

echo "vLLM is ready. Starting AIPerf sweep."

if [[ -f "$LOG_DIR/cluster_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$LOG_DIR/cluster_env.sh"
fi

echo "Checking Ray 8-GPU validation output..."
if [[ -f "$LOG_DIR/ray_cluster_resources.txt" ]]; then
  cat "$LOG_DIR/ray_cluster_resources.txt"
else
  echo "WARNING: Missing $LOG_DIR/ray_cluster_resources.txt"
fi

echo "Checking live GPU usage on all nodes before AIPerf..."
{
  for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    ssh -q "$node" "
      host=\$(hostname)
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
      | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
    " 2>/dev/null || true
  done
} | tee "$LOG_DIR/live_gpu_usage_before_aiperf.csv"

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
          | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
        " 2>/dev/null || true
      done
      sleep "$interval_sec"
    done
  } > "$out_file" 2>"${out_file}.err" &

  echo $!
}

AIPERF_OUT_DIRS=()

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="${WORKLOAD_ARTIFACT_ROOT}/${RUN_PREFIX}_concurrency${CONCURRENCY}_${TIMESTAMP}"
  mkdir -p "$OUT_DIR"
  AIPERF_OUT_DIRS+=("$OUT_DIR")

  cp "$WORKLOAD_CONFIG" "$OUT_DIR/workload.json"

  echo "Running AIPerf concurrency=$CONCURRENCY out_dir=$OUT_DIR"

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
      --warmup-request-count "$WARMUP_REQUEST_COUNT"
      --warmup-concurrency "$WARMUP_CONCURRENCY"
      --use-server-token-count
      --gpu-telemetry "$GPU_TELEMETRY_MODE"
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
      --gpu-telemetry "$GPU_TELEMETRY_MODE"
      --output-artifact-dir "$OUT_DIR"
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

  echo "Checking live GPU usage after AIPerf concurrency=$CONCURRENCY..."
  {
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
      ssh -q "$node" "
        host=\$(hostname)
        nvidia-smi \
          --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
          --format=csv,noheader,nounits \
        | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
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
    echo "AIPerf failed for concurrency=$CONCURRENCY. See $LOG_DIR/aiperf_${WORKLOAD_TYPE}_c${CONCURRENCY}.log" >&2
    exit "$AIPERF_RC"
  fi

  rm -f "$OUT_DIR/inputs.json"
done

python3 - "$RUN_DIR/run_info.json" "${AIPERF_OUT_DIRS[@]}" <<PY
import json
import sys

out = {
    "phase": "$PHASE",
    "mode": "aggregated_native_vllm",
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
    "aiperf_artifact_dirs": sys.argv[2:],
}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

{
  echo "mode=aggregated_native_vllm"
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
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

echo "Phase 4 aggregated native vLLM run complete."
echo "RUN_DIR=$RUN_DIR"
