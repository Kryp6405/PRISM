#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 aggregated native vLLM launch script.
#
# Aggregated large-model baseline:
#   E + P + D in one logical vLLM deployment
#   2 nodes × 4 GPUs = 8 GPUs
#   TP=4 within each node
#   PP=2 across nodes
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"

VENV_ACTIVATE="${VENV_ACTIVATE:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_RAY_NODE="${CPUS_PER_RAY_NODE:-5}"

EXPECTED_NODES="${EXPECTED_NODES:-2}"
EXPECTED_GPUS_TOTAL="${EXPECTED_GPUS_TOTAL:-$((TP_SIZE * PP_SIZE))}"

LOG_DIR="${LOG_DIR:-./artifacts/p4/aggregated_native_vllm_logs}"
mkdir -p "$LOG_DIR"

CLUSTER_ENV_FILE="${LOG_DIR}/cluster_env.sh"

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "ERROR: This script expects to run inside a Slurm allocation." >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ "${#NODES[@]}" -lt 2 ]]; then
  echo "ERROR: Need at least 2 nodes. Found: ${#NODES[@]}" >&2
  printf '%s\n' "${NODES[@]}" >&2
  exit 1
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(ssh -q "$HEAD_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

if [[ -z "$HEAD_IP" ]]; then
  echo "ERROR: Could not resolve HEAD_IP for $HEAD_NODE" >&2
  exit 1
fi

echo "Launching native vLLM aggregated baseline"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "TP_SIZE=$TP_SIZE"
echo "PP_SIZE=$PP_SIZE"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CPUS_PER_RAY_NODE=$CPUS_PER_RAY_NODE"
echo "EXPECTED_NODES=$EXPECTED_NODES"
echo "EXPECTED_GPUS_TOTAL=$EXPECTED_GPUS_TOTAL"
echo "HEAD_NODE=$HEAD_NODE"
echo "WORKER_NODE=$WORKER_NODE"
echo "HEAD_IP=$HEAD_IP"
echo "RAY_ADDRESS=$RAY_ADDRESS"
echo "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export HEAD_NODE="$HEAD_NODE"
export WORKER_NODE="$WORKER_NODE"
export HEAD_IP="$HEAD_IP"
export RAY_ADDRESS="$RAY_ADDRESS"
export PORT="$PORT"
export MODEL="$MODEL"
export TP_SIZE="$TP_SIZE"
export PP_SIZE="$PP_SIZE"
export EXPECTED_NODES="$EXPECTED_NODES"
export EXPECTED_GPUS_TOTAL="$EXPECTED_GPUS_TOTAL"
EOF

echo "Wrote cluster env to $CLUSTER_ENV_FILE"

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
  echo "Stopping Ray on allocated nodes..."

  for node in "${NODES[@]}"; do
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

echo "Stopping any existing Ray processes..."
cleanup

echo "Starting Ray head on $HEAD_NODE"
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
RAY_HEAD_PID=$!

sleep 15

echo "Starting Ray worker on $WORKER_NODE"
srun --overlap --exact -N 1 -n 1 -w "$WORKER_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

ray start \
  --address='$RAY_ADDRESS' \
  --num-cpus='$CPUS_PER_RAY_NODE' \
  --num-gpus='$GPUS_PER_NODE' \
  --block
" > "$LOG_DIR/ray_worker.log" 2>&1 &
RAY_WORKER_PID=$!

sleep 20

echo "Checking Ray status..."
srun --overlap --exact -N 1 -n 1 -w "$HEAD_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

ray status --address='$RAY_ADDRESS'
" | tee "$LOG_DIR/ray_status.txt"

echo "Checking Ray cluster resources..."
srun --overlap --exact -N 1 -n 1 -w "$HEAD_NODE" --gpus="$GPUS_PER_NODE" bash -lc "
$(env_prefix)

python3 - <<PY
import json
import ray

ray.init(address='$RAY_ADDRESS')

nodes = ray.nodes()
alive_nodes = [n for n in nodes if n.get('Alive')]
resources = ray.cluster_resources()

print('=== Ray alive nodes ===')
for n in alive_nodes:
    print(json.dumps({
        'NodeManagerAddress': n.get('NodeManagerAddress'),
        'Alive': n.get('Alive'),
        'Resources': n.get('Resources'),
    }, indent=2))

print('=== Ray cluster resources ===')
print(json.dumps(resources, indent=2))

gpu_total = int(resources.get('GPU', 0))
expected_gpus = int('$EXPECTED_GPUS_TOTAL')
expected_nodes = int('$EXPECTED_NODES')

if len(alive_nodes) < expected_nodes:
    raise SystemExit(f'ERROR: Expected at least {expected_nodes} Ray nodes, saw {len(alive_nodes)}')

if gpu_total < expected_gpus:
    raise SystemExit(f'ERROR: Expected at least {expected_gpus} Ray GPUs, saw {gpu_total}')

print(f'OK: Ray sees {len(alive_nodes)} alive nodes and {gpu_total} GPUs.')
PY
" | tee "$LOG_DIR/ray_cluster_resources.txt"

if ! grep -q "OK: Ray sees" "$LOG_DIR/ray_cluster_resources.txt"; then
  echo "ERROR: Ray cluster validation failed. Inspect $LOG_DIR/ray_cluster_resources.txt" >&2
  exit 1
fi

echo "Starting vLLM serve on Ray cluster..."
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
  --limit-mm-per-prompt '{\"image\":$MAX_IMAGES_PER_PROMPT,\"video\":0}' \
  --mm-encoder-tp-mode data \
  --enable-request-id-headers
" > "$LOG_DIR/vllm_serve.log" 2>&1
