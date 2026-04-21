#!/usr/bin/env bash
set -euo pipefail

: "${FULL_DISAGG_LAUNCH_CMD:?Set FULL_DISAGG_LAUNCH_CMD}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-10}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4 16}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
PHASE="${PHASE:-p1}"
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-full-disagg-openai-chat}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR" "$SUMMARY_DIR"

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
else
  echo "scripts/capture_env.sh not found or not executable; skipping env capture" \
    > "$LOG_DIR/capture_env.log"
fi

echo "Launching full E/P/D stack..."
echo "RUN_DIR=$RUN_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "CONCURRENCY_VALUES=$CONCURRENCY_VALUES"

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
     grep -q "Registered endpoint" "$LOG_DIR/decode.log" 2>/dev/null || \
     grep -q "engine initialized" "$LOG_DIR/decode.log" 2>/dev/null; then
    decode_ready=1
  fi

  if grep -q "Proxy listening on" "$LOG_DIR/proxy.log" 2>/dev/null || \
     grep -q "Application startup complete" "$LOG_DIR/proxy.log" 2>/dev/null || \
     grep -q "Uvicorn running on" "$LOG_DIR/proxy.log" 2>/dev/null; then
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
  "mode": "full_disagg",
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

if [[ -f src/phase0/summarize_run.py ]]; then
  python3 src/phase0/summarize_run.py \
    --artifact-root "${ARTIFACT_ROOT}/${PHASE}" \
    --run-prefix "${RUN_PREFIX}" \
    > "$SUMMARY_DIR/summary.json" || true
fi

{
  echo "mode=full_disagg"
  echo "phase=$PHASE"
  echo "model=$MODEL"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "launcher_pid=$LAUNCHER_PID"
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

echo "Phase 1 full disagg run complete."
echo "RUN_DIR=$RUN_DIR"
