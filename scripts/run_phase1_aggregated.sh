#!/usr/bin/env bash
set -euo pipefail

# Phase 1 aggregated runner for the current multimodal baseline.
# Assumes repo root as working directory.

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-80}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4 16}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
PHASE="${PHASE:-p1}"

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
}
trap cleanup EXIT

# Copy templates if they exist in repo
[[ -f src/phase1/run_manifest_template.json ]] && \
  cp src/phase1/run_manifest_template.json "$RUN_DIR/run_manifest.json"

[[ -f src/phase1/workloads/baseline_workload.json ]] && \
  cp src/phase1/workloads/baseline_workload.json "$RUN_DIR/workload.json"

if [[ -x scripts/capture_env.sh ]]; then
  bash scripts/capture_env.sh "$ENV_DIR"
else
  echo "scripts/capture_env.sh not found or not executable; skipping env capture" \
    > "$LOG_DIR/capture_env.log"
fi

echo "Starting aggregated Dynamo frontend on port $PORT"
python -m dynamo.frontend \
  --http-port "$PORT" \
  --discovery-backend "$DISCOVERY_BACKEND" \
  > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "Starting aggregated vLLM multimodal worker for $MODEL"
python -m dynamo.vllm \
  --model "$MODEL" \
  --discovery-backend "$DISCOVERY_BACKEND" \
  --kv-events-config '{"enable_kv_cache_events": false}' \
  --enable-multimodal \
  --limit-mm-per-prompt '{"image": 1}' \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!

echo "Waiting for worker readiness..."
READY=0
for _ in $(seq 1 180); do
  # fail fast if worker died
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited early; inspect $LOG_DIR/worker.log" | tee "$RUN_DIR/summary.txt"
    exit 1
  fi

  if grep -q "Application startup complete" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "Registered endpoint" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/worker.log" 2>/dev/null; then
    READY=1
    break
  fi

  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  cat > "$RUN_DIR/summary.txt" <<EOF
Aggregated worker failed to become ready.

Inspect:
  $LOG_DIR/frontend.log
  $LOG_DIR/worker.log
EOF
  exit 1
fi

AIPERF_OUT_DIRS=()

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_concurrency${CONCURRENCY}_${TIMESTAMP}"
  AIPERF_OUT_DIRS+=("$OUT_DIR")

  mkdir -p "$OUT_DIR"

  echo "Running aiperf with concurrency=$CONCURRENCY"
  echo "OUT_DIR=$OUT_DIR"

  aiperf profile \
    --model "$MODEL" \
    --url "http://localhost:${PORT}" \
    --endpoint-type chat \
    --image-width-mean "$IMAGE_WIDTH_MEAN" \
    --image-height-mean "$IMAGE_HEIGHT_MEAN" \
    --concurrency "$CONCURRENCY" \
    --request-count "$REQUEST_COUNT" \
    --use-server-token-count \
    --output-artifact-dir "$OUT_DIR" \
    > "$LOG_DIR/aiperf_c${CONCURRENCY}.log" 2>&1
done

cat > "$RUN_DIR/run_info.json" <<EOF
{
  "mode": "aggregated",
  "phase": "${PHASE}",
  "model": "${MODEL}",
  "run_prefix": "${RUN_PREFIX}",
  "timestamp": "${TIMESTAMP}",
  "run_dir": "${RUN_DIR}",
  "log_dir": "${LOG_DIR}",
  "port": ${PORT},
  "request_count": ${REQUEST_COUNT},
  "image_width_mean": ${IMAGE_WIDTH_MEAN},
  "image_height_mean": ${IMAGE_HEIGHT_MEAN},
  "concurrency_values": [$(echo "$CONCURRENCY_VALUES" | sed 's/ /, /g')]
}
EOF

if [[ -f src/phase1/summarize_run.py ]]; then
  python src/phase1/summarize_run.py \
    --artifact-root "${ARTIFACT_ROOT}/${PHASE}" \
    --run-prefix "${RUN_PREFIX}" \
    > "$SUMMARY_DIR/summary.json" || true
fi

{
  echo "mode=aggregated"
  echo "phase=$PHASE"
  echo "model=$MODEL"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "frontend_pid=$FRONTEND_PID"
  echo "worker_pid=$WORKER_PID"
  echo "port=$PORT"
  echo "request_count=$REQUEST_COUNT"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "run_dir=$RUN_DIR"
  echo "log_dir=$LOG_DIR"
  echo "summary_dir=$SUMMARY_DIR"
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

echo "Phase 1 aggregated run complete."
echo "RUN_DIR=$RUN_DIR"
