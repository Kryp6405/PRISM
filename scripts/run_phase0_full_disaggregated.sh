#!/usr/bin/env bash
set -euo pipefail

: "${FULL_DISAGG_LAUNCH_CMD:?Set FULL_DISAGG_LAUNCH_CMD}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-10}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}_phase0_full_disagg}"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR"

# Launch command is now embedded by default
FULL_DISAGG_LAUNCH_CMD="${FULL_DISAGG_LAUNCH_CMD:-bash scripts/launch_full_disagg_perlmutter.sh}"

cleanup() {
  set +e
  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi
  pkill -P $$ python 2>/dev/null || true
}
trap cleanup EXIT

[[ -f src/phase0/run_manifest_template.json ]] && \
  cp src/phase0/run_manifest_template.json "$RUN_DIR/run_manifest.json"

[[ -f src/phase0/workloads/baseline_workload.json ]] && \
  cp src/phase0/workloads/baseline_workload.json "$RUN_DIR/workload.json"

if [[ -x scripts/capture_env.sh ]]; then
  echo "Capturing environment snapshot..."
  bash scripts/capture_env.sh "$ENV_DIR" > "$LOG_DIR/capture_env.log" 2>&1 || true
fi

echo "Launching full E/P/D stack..."
LOG_DIR="$LOG_DIR" PORT="$PORT" bash -lc "$FULL_DISAGG_LAUNCH_CMD" \
  > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Launcher PID: $LAUNCHER_PID"
echo "Waiting for readiness..."

READY=0
for _ in $(seq 1 240); do
  if (( _ % 10 == 0 )); then
    echo "still waiting... $(date)"
  fi

  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "Launcher exited early; inspect $LOG_DIR/launcher.log" | tee "$RUN_DIR/summary.txt"
    exit 1
  fi

  encoder_ready=0
  prefill_ready=0
  decode_ready=0
  proxy_ready=0

  if grep -q "Application startup complete" "$LOG_DIR/encoder.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/encoder.log" 2>/dev/null; then
    encoder_ready=1
  fi

  if grep -q "Application startup complete" "$LOG_DIR/prefill.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/prefill.log" 2>/dev/null; then
    prefill_ready=1
  fi

  if grep -q "Application startup complete" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "Registered endpoint 'generate'" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/decode.log" 2>/dev/null; then
    decode_ready=1
  fi

  if grep -q "Proxy listening on" "$LOG_DIR/proxy.log" 2>/dev/null || \
     grep -q "Application startup complete" "$LOG_DIR/proxy.log" 2>/dev/null; then
    proxy_ready=1
  fi

  if [[ "$encoder_ready" -eq 1 && "$prefill_ready" -eq 1 && "$decode_ready" -eq 1 && "$proxy_ready" -eq 1 ]]; then
    READY=1
    break
  fi

  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  cat > "$RUN_DIR/summary.txt" <<EOF
Full E/P/D workers failed to become ready.

Inspect:
  $LOG_DIR/launcher.log
  $LOG_DIR/encoder.log
  $LOG_DIR/prefill.log
  $LOG_DIR/decode.log
  $LOG_DIR/proxy.log
EOF
  exit 1
fi

echo "All workers ready. Starting aiperf..."

for CONCURRENCY in $CONCURRENCY_VALUES; do
  OUT_DIR="$ARTIFACT_ROOT/${MODEL//\//_}-full-disagg-openai-chat-concurrency${CONCURRENCY}"
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

if [[ -f src/phase0/summarize_run.py ]]; then
  python3 src/phase0/summarize_run.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --run-prefix "${MODEL//\//_}-full-disagg-openai-chat-concurrency" \
    > "$RUN_DIR/summary.json" || true
fi

cat > "$RUN_DIR/summary.txt" <<TXT
mode=full_disagg
model=$MODEL
launcher_pid=$LAUNCHER_PID
port=$PORT
request_count=$REQUEST_COUNT
concurrency_values=$CONCURRENCY_VALUES
run_dir=$RUN_DIR
log_dir=$LOG_DIR
TXT

echo "Phase 0 full disagg run complete."
