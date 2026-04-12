#!/usr/bin/env bash
set -euo pipefail

# Phase 0 baseline runner for your current aggregated multimodal setup.
# Assumes repo root as working directory.

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-10}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4}"
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}_phase0_aggregated}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_DIR/logs"
ENV_DIR="$RUN_DIR/env"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR"

cleanup() {
  pkill -P $$ python || true
}
trap cleanup EXIT

# Copy templates if they exist in repo
[[ -f src/phase0/run_manifest_template.json ]] && cp src/phase0/run_manifest_template.json "$RUN_DIR/run_manifest.json"
[[ -f src/phase0/workloads/baseline_workload.json ]] && cp src/phase0/workloads/baseline_workload.json "$RUN_DIR/workload.json"

bash scripts/capture_env.sh "$ENV_DIR"

echo "Starting Dynamo frontend on port $PORT"
python -m dynamo.frontend \
  --http-port "$PORT" \
  --discovery-backend "$DISCOVERY_BACKEND" \
  > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "Starting vLLM multimodal worker for $MODEL"
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
for _ in $(seq 1 120); do
  if grep -q "Application startup complete" "$LOG_DIR/worker.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/worker.log" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "Worker failed to become ready; inspect $LOG_DIR/worker.log" | tee "$RUN_DIR/summary.txt"
  exit 1
fi

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="$ARTIFACT_ROOT/${MODEL//\//_}-openai-chat-concurrency${CONCURRENCY}"
  echo "Running aiperf with concurrency=$CONCURRENCY"
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

python src/phase0/summarize_run.py --artifact-root "$ARTIFACT_ROOT" --run-prefix "${MODEL//\//_}-openai-chat-concurrency" > "$RUN_DIR/summary.json" || true

cat > "$RUN_DIR/summary.txt" <<TXT
mode=aggregated
model=$MODEL
frontend_pid=$FRONTEND_PID
worker_pid=$WORKER_PID
port=$PORT
request_count=$REQUEST_COUNT
concurrency_values=$CONCURRENCY_VALUES
run_dir=$RUN_DIR
TXT

echo "Phase 0 aggregated run complete."
