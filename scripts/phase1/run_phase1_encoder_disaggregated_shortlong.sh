#!/usr/bin/env bash
set -euo pipefail

# Phase 1 runner for encoder-only / partially disaggregated setup.
# Expects a launcher command that starts:
#   1) encoder worker
#   2) combined prefill+decode worker
#   3) proxy/frontend on PORT
#
# Recommended logs written by launcher:
#   logs/encoder.log
#   logs/decode.log
#   logs/proxy.log
#
# Example:
#   export ENCODER_ONLY_LAUNCH_CMD='bash scripts/launch_encoder_partial_perlmutter.sh'
#   bash scripts/run_phase1_encoder_disaggregated.sh

ENCODER_ONLY_LAUNCH_CMD="${ENCODER_ONLY_LAUNCH_CMD:-bash scripts/launch_encoder_partial_perlmutter.sh}"

MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PORT="${PORT:-8000}"
IMAGE_WIDTH_MEAN="${IMAGE_WIDTH_MEAN:-512}"
IMAGE_HEIGHT_MEAN="${IMAGE_HEIGHT_MEAN:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-80}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 4 16}"

GENERATION_LENGTHS="${GENERATION_LENGTHS:-short long}"
SHORT_OUTPUT_TOKENS_MEAN="${SHORT_OUTPUT_TOKENS_MEAN:-64}"
LONG_OUTPUT_TOKENS_MEAN="${LONG_OUTPUT_TOKENS_MEAN:-256}"
OUTPUT_TOKENS_STDDEV="${OUTPUT_TOKENS_STDDEV:-0}"
USE_LEGACY_MAX_TOKENS="${USE_LEGACY_MAX_TOKENS:-false}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
PHASE="${PHASE:-p1}"

# Keep RUN_PREFIX stable and human-readable. Do not include timestamp here.
RUN_PREFIX="${RUN_PREFIX:-${MODEL//\//_}-encoder-only-openai-chat}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ARTIFACT_ROOT}/${PHASE}/${RUN_PREFIX}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/env"
SUMMARY_DIR="${RUN_DIR}/summary"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$ENV_DIR" "$SUMMARY_DIR"

cleanup() {
  set +e
  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi
  pkill -P $$ python 2>/dev/null || true
}
trap cleanup EXIT

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

echo "Launching encoder-disaggregated E/PD stack"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "CONCURRENCY_VALUES=$CONCURRENCY_VALUES"

# Launch stack
LOG_DIR="$LOG_DIR" PORT="$PORT" bash -lc "$ENCODER_ONLY_LAUNCH_CMD" \
  > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

echo "Waiting for readiness..."
READY=0
for _ in $(seq 1 180); do
  if (( _ % 10 == 0 )); then
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

  # Allow encoder+decode to be sufficient if proxy.log is not created
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

# Track all per-concurrency artifact dirs for later summarization
AIPERF_OUT_DIRS=()

for GEN_CLASS in $GENERATION_LENGTHS; do
  case "$GEN_CLASS" in
    short)
      OUTPUT_TOKENS_MEAN="$SHORT_OUTPUT_TOKENS_MEAN"
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

    echo "Running aiperf with generation=$GEN_CLASS output_tokens_mean=$OUTPUT_TOKENS_MEAN concurrency=$CONCURRENCY"
    echo "OUT_DIR=$OUT_DIR"

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
      --output-artifact-dir "$OUT_DIR"
    )

    if [[ "$USE_LEGACY_MAX_TOKENS" == "true" ]]; then
      AIPERF_CMD+=(--use-legacy-max-tokens)
    fi

    "${AIPERF_CMD[@]}" > "$LOG_DIR/aiperf_${GEN_CLASS}_c${CONCURRENCY}.log" 2>&1
  done
done

# Write machine-readable metadata for this run group
cat > "$RUN_DIR/run_info.json" <<EOF
{
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
  "concurrency_values": [$(echo "$CONCURRENCY_VALUES" | sed 's/ /, /g')],
  "generation_lengths": ["short", "long"],
  "short_output_tokens_mean": ${SHORT_OUTPUT_TOKENS_MEAN},
  "long_output_tokens_mean": ${LONG_OUTPUT_TOKENS_MEAN},
  "output_tokens_stddev": ${OUTPUT_TOKENS_STDDEV},
  "use_legacy_max_tokens": ${USE_LEGACY_MAX_TOKENS}
}
EOF

# Optional summarization
if [[ -f src/phase1/summarize_run.py ]]; then
  python3 src/phase1/summarize_run.py \
    --artifact-root "${ARTIFACT_ROOT}/${PHASE}" \
    --run-prefix "${RUN_PREFIX}" \
    > "$SUMMARY_DIR/summary.json" || true
fi

# Human-readable summary
{
  echo "mode=encoder_only"
  echo "phase=$PHASE"
  echo "model=$MODEL"
  echo "run_prefix=$RUN_PREFIX"
  echo "timestamp=$TIMESTAMP"
  echo "launcher_pid=$LAUNCHER_PID"
  echo "port=$PORT"
  echo "request_count=$REQUEST_COUNT"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "generation_lengths=$GENERATION_LENGTHS"
  echo "short_output_tokens_mean=$SHORT_OUTPUT_TOKENS_MEAN"
  echo "long_output_tokens_mean=$LONG_OUTPUT_TOKENS_MEAN"
  echo "output_tokens_stddev=$OUTPUT_TOKENS_STDDEV"
  echo "use_legacy_max_tokens=$USE_LEGACY_MAX_TOKENS"
  echo "run_dir=$RUN_DIR"
  echo "log_dir=$LOG_DIR"
  echo "summary_dir=$SUMMARY_DIR"
  echo
  echo "aiperf_artifact_dirs:"
  for d in "${AIPERF_OUT_DIRS[@]}"; do
    echo "  - $d"
  done
} > "$RUN_DIR/summary.txt"

echo "Phase 1 encoder-only run complete."
echo "RUN_DIR=$RUN_DIR"
