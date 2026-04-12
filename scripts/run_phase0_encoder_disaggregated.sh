#!/usr/bin/env bash
set -euo pipefail

# Phase 0 runner for encoder-only disaggregation.
# Exact launch commands are supplied via environment variables.

: "${ENCODER_ONLY_LAUNCH_CMD:?Set ENCODER_ONLY_LAUNCH_CMD to the official E/PD launcher command}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-10}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}_phase0_encoder_only}"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR"

cleanup() {
  pkill -P $$ python || true
}
trap cleanup EXIT

[[ -f src/phase0/run_manifest_template.json ]] && cp src/phase0/run_manifest_template.json "$RUN_DIR/run_manifest.json"
[[ -f src/phase0/workloads/baseline_workload.json ]] && cp src/phase0/workloads/baseline_workload.json "$RUN_DIR/workload.json"

bash scripts/capture_env.sh "$ENV_DIR"

echo "Launching official encoder-disaggregated E/PD stack"
bash -lc "$ENCODER_ONLY_LAUNCH_CMD" > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Waiting for readiness..."
READY=0
for _ in $(seq 1 120); do
  if grep -q "Application startup complete" "$LOG_DIR/encoder.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "Application startup complete" "$LOG_DIR/decode.log" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "Encoder-only workers failed to become ready; inspect logs in $LOG_DIR" | tee "$RUN_DIR/summary.txt"
  exit 1
fi

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="$ARTIFACT_ROOT/${MODEL//\//_}-encoder-only-openai-chat-concurrency${CONCURRENCY}"
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

python3 src/phase0/summarize_run.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-prefix "${MODEL//\//_}-encoder-only-openai-chat-concurrency" \
  > "$RUN_DIR/summary.json" || true

cat > "$RUN_DIR/summary.txt" <<TXT
mode=encoder_only
model=$MODEL
launcher_pid=$PID
port=$PORT
request_count=$REQUEST_COUNT
concurrency_values=$CONCURRENCY_VALUES
run_dir=$RUN_DIR
TXT

echo "Phase 0 encoder-only run complete."
