#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "ERROR: Run this inside the Slurm allocation."
  exit 1
fi

for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  echo "===== Cleaning $node ====="

  ssh -q "$node" '
    ray stop --force >/dev/null 2>&1 || true

    pkill -f "vllm serve" >/dev/null 2>&1 || true
    pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
    pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
    pkill -f "RayWorkerWrapper" >/dev/null 2>&1 || true
    pkill -f "raylet" >/dev/null 2>&1 || true
    pkill -f "gcs_server" >/dev/null 2>&1 || true
    pkill -f "aiperf profile" >/dev/null 2>&1 || true

    # Optional: keep these if you also run Dynamo in the same allocation.
    pkill -f "python -m dynamo.frontend" >/dev/null 2>&1 || true
    pkill -f "python -m dynamo.vllm" >/dev/null 2>&1 || true
  ' 2>/dev/null || true
done

sleep 5

echo "===== GPU state after cleanup ====="
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  ssh -q "$node" "
    host=\$(hostname)
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits \
    | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
  " 2>/dev/null || true
done
