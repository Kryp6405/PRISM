#!/usr/bin/env bash
set -euo pipefail

# Phase 0 runner for encoder-only / partially disaggregated setup.
# Expects a launcher command that starts:
#   1) encoder worker
#   2) combined prefill+decode worker
#   3) proxy/frontend on PORT
#
# The launcher should write:
#   logs/encoder.log
#   logs/decode.log
#   logs/proxy.log   (recommended)
#
# Example:
#   export ENCODER_ONLY_LAUNCH_CMD='bash scripts/launch_encoder_partial.sh "$LOG_DIR"'
#   bash scripts/run_phase0_encoder_disaggregated.sh

: "${ENCODER_ONLY_LAUNCH_CMD:?Set ENCODER_ONLY_LAUNCH_CMD to the partial-disagg launcher command}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-10}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}_phase0_encoder_only}"
PHASE="p1"
RUN_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR"

ENCODER_ONLY_LAUNCH_CMD="${ENCODER_ONLY_LAUNCH_CMD:-bash scripts/launch_encoder_partial_perlmutter.sh}"

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
  bash scripts/capture_env.sh "$ENV_DIR"
else
  echo "scripts/capture_env.sh not found or not executable; skipping env capture" \
    > "$LOG_DIR/capture_env.log"
fi

echo "Launching encoder-disaggregated E/PD stack"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_DIR=$LOG_DIR"

# Pass LOG_DIR and PORT through to the launcher.
# The launcher may reference them when creating encoder/decode/proxy logs.
LOG_DIR="$LOG_DIR" PORT="$PORT" bash -lc "$ENCODER_ONLY_LAUNCH_CMD" \
  > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Waiting for readiness..."
READY=0
for _ in $(seq 1 180); do
  # Fail fast if launcher crashed
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

  # If the launcher doesn't create proxy.log, allow encoder+decode to be enough.
  if [[ "$encoder_ready" -eq 1 && "$decode_ready" -eq 1 ]]; then
    READY=1
    break
  fi

  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  cat > "$RUN_DIR/summary.txt" <<EOF
Encoder-only workers failed to become ready.

Inspect:
  $LOG_DIR/launcher.log
  $LOG_DIR/encoder.log
  $LOG_DIR/decode.log
  $LOG_DIR/proxy.log
EOF
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

if [[ -f src/phase0/summarize_run.py ]]; then
  python3 src/phase0/summarize_run.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --run-prefix "${MODEL//\//_}-encoder-only-openai-chat-concurrency" \
    > "$RUN_DIR/summary.json" || true
fi

cat > "$RUN_DIR/summary.txt" <<TXT
mode=encoder_only
model=$MODEL
launcher_pid=$LAUNCHER_PID
port=$PORT
request_count=$REQUEST_COUNT
concurrency_values=$CONCURRENCY_VALUES
run_dir=$RUN_DIR
log_dir=$LOG_DIR
TXT

echo "Phase 0 encoder-only run complete."
