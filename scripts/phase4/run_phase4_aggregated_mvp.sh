#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/p4/mvp_8gpu_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ARTIFACT_ROOT/logs"
mkdir -p "$LOG_DIR"

REQUEST_COUNT="${REQUEST_COUNT:-2}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-2}"
CONCURRENCY="${CONCURRENCY:-1}"

LAUNCH_CMD="${LAUNCH_CMD:-bash scripts/phase4/launch_aggregated_vllm_mvp.sh}"

echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "LOG_DIR=$LOG_DIR"

LOG_DIR="$LOG_DIR" MODEL="$MODEL" PORT="$PORT" bash -lc "$LAUNCH_CMD" \
  > "$LOG_DIR/launcher.log" 2>&1 &
LAUNCH_PID=$!

echo "LAUNCH_PID=$LAUNCH_PID"
echo "Waiting for vLLM readiness..."

READY=0
for i in $(seq 1 900); do
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: launcher exited early"
    tail -100 "$LOG_DIR/launcher.log" || true
    tail -100 "$LOG_DIR/ray_8gpu_check.txt" || true
    tail -100 "$LOG_DIR/vllm_serve.log" || true
    exit 1
  fi

  if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: vLLM did not become ready"
  tail -100 "$LOG_DIR/launcher.log" || true
  tail -100 "$LOG_DIR/vllm_serve.log" || true
  exit 1
fi

echo "vLLM ready."

echo "Checking all 8 GPUs before AIPerf..."
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  ssh -q "$node" "
    host=\$(hostname)
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits \
    | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
  " 2>/dev/null
done | tee "$LOG_DIR/gpu_before_aiperf.csv"

GPU_MONITOR_CSV="$ARTIFACT_ROOT/gpu_during_aiperf.csv"
GPU_MONITOR_STOP="$ARTIFACT_ROOT/gpu_monitor.stop"

rm -f "$GPU_MONITOR_STOP"
echo "host,timestamp,gpu_index,memory_used_mb,memory_total_mb,gpu_util_percent,power_watts,temp_c" > "$GPU_MONITOR_CSV"

{
  while [[ ! -f "$GPU_MONITOR_STOP" ]]; do
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
      ssh -q "$node" "
        host=\$(hostname)
        nvidia-smi \
          --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
          --format=csv,noheader,nounits \
        | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
      " 2>/dev/null
    done
    sleep 2
  done
} >> "$GPU_MONITOR_CSV" &
MONITOR_PID=$!

echo "Running AIPerf..."
set +e
aiperf profile \
  --model "$MODEL" \
  --url "http://localhost:${PORT}" \
  --endpoint-type chat \
  --image-width-mean 512 \
  --image-height-mean 512 \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count "$WARMUP_REQUEST_COUNT" \
  --warmup-concurrency 1 \
  --use-server-token-count \
  --prompt-output-tokens-mean 64 \
  --prompt-output-tokens-stddev 0 \
  --gpu-telemetry pynvml \
  --streaming \
  --output-artifact-dir "$ARTIFACT_ROOT/aiperf" \
  > "$LOG_DIR/aiperf.log" 2>&1
AIPERF_RC=$?
set -e

touch "$GPU_MONITOR_STOP"
wait "$MONITOR_PID" 2>/dev/null || true

echo "Checking all 8 GPUs after AIPerf..."
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  ssh -q "$node" "
    host=\$(hostname)
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits \
    | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
  " 2>/dev/null
done | tee "$LOG_DIR/gpu_after_aiperf.csv"

echo
echo "=== Ray 8-GPU check ==="
cat "$LOG_DIR/ray_8gpu_check.txt" || true

echo
echo "=== Unique GPU telemetry hosts ==="
cut -d, -f1 "$GPU_MONITOR_CSV" | tail -n +2 | sort | uniq -c || true

echo
echo "=== GPU telemetry sample ==="
head -20 "$GPU_MONITOR_CSV"

if [[ "$AIPERF_RC" -ne 0 ]]; then
  echo "ERROR: AIPerf failed. Inspect $LOG_DIR/aiperf.log"
  exit "$AIPERF_RC"
fi

echo
echo "MVP complete."
echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "Check:"
echo "  $LOG_DIR/ray_8gpu_check.txt"
echo "  $GPU_MONITOR_CSV"
echo "  $LOG_DIR/aiperf.log"
