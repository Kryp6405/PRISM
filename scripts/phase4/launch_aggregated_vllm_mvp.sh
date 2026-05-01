#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

VENV_ACTIVATE="${VENV_ACTIVATE:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_RAY_NODE="${CPUS_PER_RAY_NODE:-5}"

LOG_DIR="${LOG_DIR:-./artifacts/p4/mvp_8gpu_logs}"
mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "ERROR: Run inside a 2-node Slurm allocation." >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ "${#NODES[@]}" -lt 2 ]]; then
  echo "ERROR: Need 2 nodes. Found ${#NODES[@]}." >&2
  printf '%s\n' "${NODES[@]}" >&2
  exit 1
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(ssh -q "$HEAD_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "=== MVP 8-GPU vLLM launch ==="
echo "MODEL=$MODEL"
echo "HEAD_NODE=$HEAD_NODE"
echo "WORKER_NODE=$WORKER_NODE"
echo "HEAD_IP=$HEAD_IP"
echo "RAY_ADDRESS=$RAY_ADDRESS"
echo "TP_SIZE=$TP_SIZE"
echo "PP_SIZE=$PP_SIZE"
echo "EXPECTED_GPUS=$((TP_SIZE * PP_SIZE))"
echo "LOG_DIR=$LOG_DIR"

env_prefix() {
  cat <<EOF
set -euo pipefail
if [[ -n "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
fi

export HF_HOME="\${SCRATCH:-\$HOME}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="\$HF_HOME"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="\${SCRATCH:-\$HOME}/.cache/torch_inductor"
export NCCL_DEBUG="\${NCCL_DEBUG:-WARN}"
EOF
}

cleanup() {
  set +e
  echo "Stopping Ray..."
  for node in "$HEAD_NODE" "$WORKER_NODE"; do
    srun --overlap --exact -N 1 -n 1 -w "$node" --gpus="$GPUS_PER_NODE" bash -lc "
      if [[ -n '$VENV_ACTIVATE' ]]; then
        source '$VENV_ACTIVATE'
      fi
      ray stop --force >/dev/null 2>&1 || true
    " &
  done
  wait || true
}
trap cleanup EXIT

cleanup

echo "Starting Ray head..."
srun --overlap --exact -N 1 -n 1 -w "$HEAD_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

ray start --head \
  --node-ip-address='$HEAD_IP' \
  --port='$RAY_PORT' \
  --dashboard-host=0.0.0.0 \
  --dashboard-port='$RAY_DASHBOARD_PORT' \
  --num-cpus='$CPUS_PER_RAY_NODE' \
  --num-gpus='$GPUS_PER_NODE' \
  --block
" > "$LOG_DIR/ray_head.log" 2>&1 &

sleep 15

echo "Starting Ray worker..."
srun --overlap --exact -N 1 -n 1 -w "$WORKER_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

ray start \
  --address='$RAY_ADDRESS' \
  --num-cpus='$CPUS_PER_RAY_NODE' \
  --num-gpus='$GPUS_PER_NODE' \
  --block
" > "$LOG_DIR/ray_worker.log" 2>&1 &

sleep 20

echo "Checking Ray sees 8 GPUs..."
srun --overlap --exact -N 1 -n 1 -w "$HEAD_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

python3 - <<PY
import json
import ray

ray.init(address='$RAY_ADDRESS')

nodes = ray.nodes()
alive = [n for n in nodes if n.get('Alive')]
resources = ray.cluster_resources()

print('ALIVE_NODES=', len(alive))
print('CLUSTER_RESOURCES=')
print(json.dumps(resources, indent=2))

for n in alive:
    print('NODE=', n.get('NodeManagerAddress'), 'RESOURCES=', n.get('Resources'))

gpu_count = int(resources.get('GPU', 0))
if len(alive) < 2:
    raise SystemExit(f'ERROR: expected 2 alive Ray nodes, got {len(alive)}')
if gpu_count < 8:
    raise SystemExit(f'ERROR: expected 8 Ray GPUs, got {gpu_count}')

print('OK: Ray sees 2 nodes and 8 GPUs')
PY
" | tee "$LOG_DIR/ray_8gpu_check.txt"

echo "Starting vLLM..."
srun --overlap --exact -N 1 -n 1 -w "$HEAD_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

export RAY_ADDRESS='$RAY_ADDRESS'

vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$PORT' \
  --tensor-parallel-size '$TP_SIZE' \
  --pipeline-parallel-size '$PP_SIZE' \
  --distributed-executor-backend ray \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --limit-mm-per-prompt '{\"image\":1,\"video\":0}' \
  --mm-encoder-tp-mode data \
  --enable-request-id-headers
" > "$LOG_DIR/vllm_serve.log" 2>&1
