#!/usr/bin/env bash
set -euo pipefail

# Phase 2 aggregated runner for the configured scale-out baseline.
# Naming remains phase2/p2, but this script explicitly runs:
#   Aggregated: E + P + D on one GPU
#
# Assumes repo root as working directory.

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"

# Workload image settings
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"

# Benchmark settings
REQUEST_COUNT="${REQUEST_COUNT:-100}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4 8 16 32}"

# Generation-length sweep
GENERATION_CLASSES="${GENERATION_CLASSES:-short medium long}"
SHORT_OUTPUT_TOKENS_MEAN="${SHORT_OUTPUT_TOKENS_MEAN:-64}"
MEDIUM_OUTPUT_TOKENS_MEAN="${MEDIUM_OUTPUT_TOKENS_MEAN:-256}"
LONG_OUTPUT_TOKENS_MEAN="${LONG_OUTPUT_TOKENS_MEAN:-1024}"
OUTPUT_TOKENS_STDDEV="${OUTPUT_TOKENS_STDDEV:-0}"

# AIPerf options
USE_LEGACY_MAX_TOKENS="${USE_LEGACY_MAX_TOKENS:-false}"
GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-pynvml}"
ENABLE_STREAMING="${ENABLE_STREAMING:-false}"

# Experiment framing
PHASE="${PHASE:-p2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
COMPARISON_FRAMING="${COMPARISON_FRAMING:-stage_disaggregated_scaleout}"
GPU_BUDGET="${GPU_BUDGET:-1}"

# Aggregated GPU placement.
# This is the important P2/P3-style correction:
# aggregated baseline is explicitly pinned to one visible GPU.
AGG_GPU="${AGG_GPU:-0}"

# Keep RUN_PREFIX human-readable and stable; timestamp is added separately.
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-aggregated-openai-chat}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR" "$SUMMARY_DIR"

cleanup() {
  set +e

  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ -n "${WORKER_PID:-}" ]]; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi

  pkill -P $$ python 2>/dev/null || true
  pkill -P $$ python3 2>/dev/null || true
}
trap cleanup EXIT

# Copy phase2 templates if they exist in repo
[[ -f src/phase2/run_manifest_template.json ]] && \
  cp src/phase2/run_manifest_template.json "$RUN_DIR/run_manifest.json"

[[ -f src/phase2/workloads/focused_workload.json ]] && \
  cp src/phase2/workloads/focused_workload.json "$RUN_DIR/workload.json"

if [[ -x scripts/capture_env.sh ]]; then
  bash scripts/capture_env.sh "$ENV_DIR" > "$LOG_DIR/capture_env.log" 2>&1 || true
else
  echo "scripts/capture_env.sh not found or not executable; skipping env capture" \
    > "$LOG_DIR/capture_env.log"
fi

cat > "$RUN_DIR/launch_config.json" <<EOF
{
  "mode": "aggregated",
  "phase": "${PHASE}",
  "comparison_framing": "${COMPARISON_FRAMING}",
  "gpu_budget": ${GPU_BUDGET},
  "gpu_mapping": {
    "gpu${AGG_GPU}": "aggregated_e_p_d"
  },
  "model": "${MODEL}",
  "port": ${PORT},
  "discovery_backend": "${DISCOVERY_BACKEND}",
  "enable_multimodal": true,
  "limit_mm_per_prompt": {
    "image": 1
  }
}
EOF

echo "Starting aggregated Dynamo frontend on port $PORT"
python3 -m dynamo.frontend \
  --http-port "$PORT" \
  --discovery-backend "$DISCOVERY_BACKEND" \
  > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "Starting aggregated vLLM multimodal worker for $MODEL on GPU $AGG_GPU"
CUDA_VISIBLE_DEVICES="$AGG_GPU" python3 -m dynamo.vllm \
  --model "$MODEL" \
  --discovery-backend "$DISCOVERY_BACKEND" \
  --kv-events-config '{"enable_kv_cache_events": false}' \
  --enable-multimodal \
  --limit-mm-per-prompt '{"image": 1}' \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!

echo "Frontend PID: $FRONTEND_PID"
echo "Worker PID: $WORKER_PID"
echo "Waiting for worker/frontend readiness..."

READY=0
for _ in $(seq 1 180); do
  if (( _ % 10 == 0 )); then
    echo "still waiting... $(date)"
  fi

  # Fail fast if worker died
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited early; inspect $LOG_DIR/worker.log" | tee "$RUN_DIR/summary.txt"
    exit 1
  fi

  # Fail fast if frontend died
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Frontend exited early; inspect $LOG_DIR/frontend.log" | tee "$RUN_DIR/summary.txt"
    exit 1
  fi

  worker_ready=0
  frontend_ready=0

  if grep -q "Application startup complete" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "Registered endpoint" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/worker.log" 2>/dev/null; then
    worker_ready=1
  fi

  if grep -q "Application startup complete" "$LOG_DIR/frontend.log" 2>/dev/null || \
     grep -q "Uvicorn running on" "$LOG_DIR/frontend.log" 2>/dev/null || \
     grep -q "listening" "$LOG_DIR/frontend.log" 2>/dev/null; then
    frontend_ready=1
  fi

  # Extra endpoint readiness check.
  # Do not require /v1/models to succeed on every iteration, but use it when available.
  if [[ "$worker_ready" -eq 1 ]]; then
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
  fi

  # Fallback: if both logs look ready, proceed.
  if [[ "$worker_ready" -eq 1 && "$frontend_ready" -eq 1 ]]; then
    READY=1
    break
  fi

  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  cat > "$RUN_DIR/summary.txt" <<EOF
Aggregated worker/frontend failed to become ready.

Inspect:
  $LOG_DIR/frontend.log
  $LOG_DIR/worker.log
EOF
  exit 1
fi

echo "Aggregated service is ready. Starting AIPerf sweep..."

AIPERF_OUT_DIRS=()

for GEN_CLASS in $GENERATION_CLASSES; do
  case "$GEN_CLASS" in
    short)
      OUTPUT_TOKENS_MEAN="$SHORT_OUTPUT_TOKENS_MEAN"
      ;;
    medium)
      OUTPUT_TOKENS_MEAN="$MEDIUM_OUTPUT_TOKENS_MEAN"
      ;;
    long)
      OUTPUT_TOKENS_MEAN="$LONG_OUTPUT_TOKENS_MEAN"
      ;;
    *)
      echo "Unknown generation class: $GEN_CLASS" >&2
      exit 1
      ;;
  esac

  for CONCURRENCY in $CONCURRENCY_VALUES; do
    OUT_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_${GEN_CLASS}_concurrency${CONCURRENCY}_${TIMESTAMP}"
    AIPERF_OUT_DIRS+=("$OUT_DIR")
    mkdir -p "$OUT_DIR"

    echo "Running AIPerf:"
    echo "  generation=$GEN_CLASS"
    echo "  output_tokens_mean=$OUTPUT_TOKENS_MEAN"
    echo "  concurrency=$CONCURRENCY"
    echo "  out_dir=$OUT_DIR"

    AIPERF_CMD=(
      aiperf profile
      --model "$MODEL"
      --url "http://localhost:${PORT}"
      --endpoint-type chat
      --image-width-mean "$IMAGE_WIDTH_MEAN"
      --image-height-mean "$IMAGE_HEIGHT_MEAN"
      --concurrency "$CONCURRENCY"
      --request-count "$REQUEST_COUNT"
      --use-server-token-count
      --prompt-output-tokens-mean "$OUTPUT_TOKENS_MEAN"
      --prompt-output-tokens-stddev "$OUTPUT_TOKENS_STDDEV"
      --gpu-telemetry "$GPU_TELEMETRY_MODE"
      --output-artifact-dir "$OUT_DIR"
    )

    if [[ "$USE_LEGACY_MAX_TOKENS" == "true" ]]; then
      AIPERF_CMD+=(--use-legacy-max-tokens)
    fi

    if [[ "$ENABLE_STREAMING" == "true" ]]; then
      AIPERF_CMD+=(--streaming)
    fi

    "${AIPERF_CMD[@]}" > "$LOG_DIR/aiperf_${GEN_CLASS}_c${CONCURRENCY}.log" 2>&1
  done
done

cat > "$RUN_DIR/run_info.json" <<EOF
{
  "mode": "aggregated",
  "phase": "${PHASE}",
  "comparison_framing": "${COMPARISON_FRAMING}",
  "gpu_budget": ${GPU_BUDGET},
  "gpu_mapping": {
    "gpu${AGG_GPU}": "aggregated_e_p_d"
  },
  "model": "${MODEL}",
  "run_prefix": "${RUN_PREFIX}",
  "timestamp": "${TIMESTAMP}",
  "run_dir": "${RUN_DIR}",
  "log_dir": "${LOG_DIR}",
  "port": ${PORT},
  "request_count": ${REQUEST_COUNT},
  "image_width_mean": ${IMAGE_WIDTH_MEAN},
  "image_height_mean": ${IMAGE_HEIGHT_MEAN},
  "concurrency_values": [$(echo "$CONCURRENCY_VALUES" | sed 's/ /, /g')],
  "generation_classes": ["short", "medium", "long"],
  "short_output_tokens_mean": ${SHORT_OUTPUT_TOKENS_MEAN},
  "medium_output_tokens_mean": ${MEDIUM_OUTPUT_TOKENS_MEAN},
  "long_output_tokens_mean": ${LONG_OUTPUT_TOKENS_MEAN},
  "output_tokens_stddev": ${OUTPUT_TOKENS_STDDEV},
  "gpu_telemetry_mode": "${GPU_TELEMETRY_MODE}",
  "enable_streaming": ${ENABLE_STREAMING},
  "endpoint_type": "chat",
  "discovery_backend": "${DISCOVERY_BACKEND}",
  "aggregated_gpu": "${AGG_GPU}"
}
EOF

if [[ -f src/phase2/summarize_run.py ]]; then
  if python3 src/phase2/summarize_run.py \
      --artifact-root "${ARTIFACT_ROOT}/${PHASE}" \
      --run-prefix "${RUN_PREFIX}" \
      > "$SUMMARY_DIR/summary.json" 2> "$LOG_DIR/summarize_run.err"; then
    echo "Wrote summary to $SUMMARY_DIR/summary.json"
  else
    echo "summarize_run.py failed; see $LOG_DIR/summarize_run.err" | tee -a "$RUN_DIR/summary.txt"
  fi
else
  echo "src/phase2/summarize_run.py not found" | tee -a "$RUN_DIR/summary.txt"
fi

{
  echo "mode=aggregated"
  echo "phase=$PHASE"
  echo "comparison_framing=$COMPARISON_FRAMING"
  echo "gpu_budget=$GPU_BUDGET"
  echo "gpu_mapping=gpu${AGG_GPU}:aggregated_e_p_d"
  echo "model=$MODEL"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "frontend_pid=$FRONTEND_PID"
  echo "worker_pid=$WORKER_PID"
  echo "port=$PORT"
  echo "request_count=$REQUEST_COUNT"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "generation_classes=$GENERATION_CLASSES"
  echo "short_output_tokens_mean=$SHORT_OUTPUT_TOKENS_MEAN"
  echo "medium_output_tokens_mean=$MEDIUM_OUTPUT_TOKENS_MEAN"
  echo "long_output_tokens_mean=$LONG_OUTPUT_TOKENS_MEAN"
  echo "output_tokens_stddev=$OUTPUT_TOKENS_STDDEV"
  echo "gpu_telemetry_mode=$GPU_TELEMETRY_MODE"
  echo "enable_streaming=$ENABLE_STREAMING"
  echo "run_dir=$RUN_DIR"
  echo "log_dir=$LOG_DIR"
  echo "summary_dir=$SUMMARY_DIR"
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

echo "Phase 2 aggregated run complete."
echo "RUN_DIR=$RUN_DIR"
