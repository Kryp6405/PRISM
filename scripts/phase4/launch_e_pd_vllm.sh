#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 E/PD native vLLM launch script.
#
# E/PD disaggregated encoder baseline:
#   Encoder: 2 GPUs on head node
#   P/D:     6 GPUs across 2 nodes
#
# Layout:
#   HEAD_NODE GPUs 0,1 -> Encoder vLLM producer
#   HEAD_NODE GPUs 2,3 -> P/D Ray head
#   WORKER_NODE GPUs 0,1,2,3 -> P/D Ray worker
#
# P/D parallelism:
#   TP=2, PP=3 => 6 GPUs
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"

# vLLM checkout and environment.
VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

# Shared EC cache path.
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache_epd}"

# GPU placement.
ENCODER_CUDA_VISIBLE_DEVICES="${ENCODER_CUDA_VISIBLE_DEVICES:-0,1}"
PD_HEAD_CUDA_VISIBLE_DEVICES="${PD_HEAD_CUDA_VISIBLE_DEVICES:-2,3}"
PD_WORKER_CUDA_VISIBLE_DEVICES="${PD_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ENCODER_TP_SIZE="${ENCODER_TP_SIZE:-2}"

PD_TP_SIZE="${PD_TP_SIZE:-2}"
PD_PP_SIZE="${PD_PP_SIZE:-3}"
PD_HEAD_GPUS="${PD_HEAD_GPUS:-2}"
PD_WORKER_GPUS="${PD_WORKER_GPUS:-4}"
PD_EXPECTED_GPUS_TOTAL="${PD_EXPECTED_GPUS_TOTAL:-$((PD_TP_SIZE * PD_PP_SIZE))}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
MAX_VIDEOS_PER_PROMPT="${MAX_VIDEOS_PER_PROMPT:-0}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
CPUS_PER_RAY_NODE="${CPUS_PER_RAY_NODE:-5}"

LOG_DIR="${LOG_DIR:-./artifacts/p4/e_pd_native_vllm_logs}"
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

: "${VLLM_ROOT:?VLLM_ROOT is empty}"
: "${VENV_ACTIVATE:?VENV_ACTIVATE is empty}"

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

echo "Launching Phase 4 E/PD native vLLM"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "ENCODE_PORT=$ENCODE_PORT"
echo "PD_PORT=$PD_PORT"
echo "HEAD_NODE=$HEAD_NODE"
echo "WORKER_NODE=$WORKER_NODE"
echo "HEAD_IP=$HEAD_IP"
echo "RAY_ADDRESS=$RAY_ADDRESS"
echo "VLLM_ROOT=$VLLM_ROOT"
echo "VENV_ACTIVATE=$VENV_ACTIVATE"
echo "EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
echo "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
echo "PD_HEAD_CUDA_VISIBLE_DEVICES=$PD_HEAD_CUDA_VISIBLE_DEVICES"
echo "PD_WORKER_CUDA_VISIBLE_DEVICES=$PD_WORKER_CUDA_VISIBLE_DEVICES"
echo "ENCODER_TP_SIZE=$ENCODER_TP_SIZE"
echo "PD_TP_SIZE=$PD_TP_SIZE"
echo "PD_PP_SIZE=$PD_PP_SIZE"
echo "PD_EXPECTED_GPUS_TOTAL=$PD_EXPECTED_GPUS_TOTAL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export MODE="e_pd"
export HEAD_NODE="$HEAD_NODE"
export WORKER_NODE="$WORKER_NODE"
export HEAD_IP="$HEAD_IP"
export RAY_ADDRESS="$RAY_ADDRESS"
export PORT="$PORT"
export ENCODE_PORT="$ENCODE_PORT"
export PD_PORT="$PD_PORT"
export MODEL="$MODEL"
export VLLM_ROOT="$VLLM_ROOT"
export EC_SHARED_STORAGE_PATH="$EC_SHARED_STORAGE_PATH"
export ENCODER_CUDA_VISIBLE_DEVICES="$ENCODER_CUDA_VISIBLE_DEVICES"
export PD_HEAD_CUDA_VISIBLE_DEVICES="$PD_HEAD_CUDA_VISIBLE_DEVICES"
export PD_WORKER_CUDA_VISIBLE_DEVICES="$PD_WORKER_CUDA_VISIBLE_DEVICES"
export ENCODER_TP_SIZE="$ENCODER_TP_SIZE"
export PD_TP_SIZE="$PD_TP_SIZE"
export PD_PP_SIZE="$PD_PP_SIZE"
export PD_EXPECTED_GPUS_TOTAL="$PD_EXPECTED_GPUS_TOTAL"
EOF

echo "Wrote cluster env to $CLUSTER_ENV_FILE"

remote_env_prefix() {
  cat <<EOF
set -euo pipefail
source "$VENV_ACTIVATE"
cd "$VLLM_ROOT"

export HF_HOME="\${SCRATCH:-\$HOME}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="\$HF_HOME"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="\${SCRATCH:-\$HOME}/.cache/torch_inductor"
export NCCL_DEBUG="\${NCCL_DEBUG:-WARN}"
EOF
}

cleanup() {
  set +e
  echo "Stopping E/PD native vLLM stack..."

  for node in "$HEAD_NODE" "$WORKER_NODE"; do
    ssh -q "$node" "
      source '$VENV_ACTIVATE' >/dev/null 2>&1 || true
      ray stop --force >/dev/null 2>&1 || true

      pkill -f 'disagg_epd_proxy.py' >/dev/null 2>&1 || true
      pkill -f 'vllm serve' >/dev/null 2>&1 || true
      pkill -f 'vllm.entrypoints.openai.api_server' >/dev/null 2>&1 || true
      pkill -f 'VLLM::EngineCore' >/dev/null 2>&1 || true
      pkill -f 'RayWorkerWrapper' >/dev/null 2>&1 || true
      pkill -f 'raylet' >/dev/null 2>&1 || true
      pkill -f 'gcs_server' >/dev/null 2>&1 || true
    " 2>/dev/null || true
  done
}

trap cleanup EXIT

echo "Cleaning existing processes..."
cleanup
sleep 5

echo "Starting encoder vLLM producer on $HEAD_NODE GPUs $ENCODER_CUDA_VISIBLE_DEVICES"
ssh -q "$HEAD_NODE" "
$(remote_env_prefix)

CUDA_VISIBLE_DEVICES='$ENCODER_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$ENCODE_PORT' \
  --tensor-parallel-size '$ENCODER_TP_SIZE' \
  --enforce-eager \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --max-num-batched-tokens 114688 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{\"image\":$MAX_IMAGES_PER_PROMPT,\"video\":$MAX_VIDEOS_PER_PROMPT}' \
  --ec-transfer-config '{
    \"ec_connector\":\"ECExampleConnector\",
    \"ec_role\":\"ec_producer\",
    \"ec_connector_extra_config\":{
      \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
    }
  }'
" > "$LOG_DIR/encoder.log" 2>&1 &
ENCODER_PID=$!

sleep 10

echo "Starting P/D Ray head on $HEAD_NODE GPUs $PD_HEAD_CUDA_VISIBLE_DEVICES"
ssh -q "$HEAD_NODE" "
$(remote_env_prefix)

CUDA_VISIBLE_DEVICES='$PD_HEAD_CUDA_VISIBLE_DEVICES' ray start --head \
  --node-ip-address='$HEAD_IP' \
  --port='$RAY_PORT' \
  --dashboard-host=0.0.0.0 \
  --dashboard-port='$RAY_DASHBOARD_PORT' \
  --num-cpus='$CPUS_PER_RAY_NODE' \
  --num-gpus='$PD_HEAD_GPUS' \
  --block
" > "$LOG_DIR/pd_ray_head.log" 2>&1 &
PD_RAY_HEAD_PID=$!

sleep 15

echo "Starting P/D Ray worker on $WORKER_NODE GPUs $PD_WORKER_CUDA_VISIBLE_DEVICES"
ssh -q "$WORKER_NODE" "
$(remote_env_prefix)

CUDA_VISIBLE_DEVICES='$PD_WORKER_CUDA_VISIBLE_DEVICES' ray start \
  --address='$RAY_ADDRESS' \
  --num-cpus='$CPUS_PER_RAY_NODE' \
  --num-gpus='$PD_WORKER_GPUS' \
  --block
" > "$LOG_DIR/pd_ray_worker.log" 2>&1 &
PD_RAY_WORKER_PID=$!

sleep 20

echo "Checking P/D Ray cluster resources..."
ssh -q "$HEAD_NODE" "
$(remote_env_prefix)

python3 - <<PY
import json
import ray

ray.init(address='$RAY_ADDRESS')

nodes = ray.nodes()
alive_nodes = [n for n in nodes if n.get('Alive')]
resources = ray.cluster_resources()

print('=== P/D Ray alive nodes ===')
for n in alive_nodes:
    print(json.dumps({
        'NodeManagerAddress': n.get('NodeManagerAddress'),
        'Alive': n.get('Alive'),
        'Resources': n.get('Resources'),
    }, indent=2))

print('=== P/D Ray cluster resources ===')
print(json.dumps(resources, indent=2))

gpu_total = int(resources.get('GPU', 0))
expected_gpus = int('$PD_EXPECTED_GPUS_TOTAL')

if gpu_total < expected_gpus:
    raise SystemExit(f'ERROR: Expected at least {expected_gpus} P/D Ray GPUs, saw {gpu_total}')

print(f'OK: P/D Ray sees {len(alive_nodes)} alive nodes and {gpu_total} GPUs.')
PY
" | tee "$LOG_DIR/pd_ray_cluster_resources.txt"

if ! grep -q "OK: P/D Ray sees" "$LOG_DIR/pd_ray_cluster_resources.txt"; then
  echo "ERROR: P/D Ray validation failed. Inspect $LOG_DIR/pd_ray_cluster_resources.txt" >&2
  exit 1
fi

echo "Starting P/D vLLM consumer on P/D Ray cluster"
ssh -q "$HEAD_NODE" "
$(remote_env_prefix)

export RAY_ADDRESS='$RAY_ADDRESS'
CUDA_VISIBLE_DEVICES='$PD_HEAD_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$PD_PORT' \
  --tensor-parallel-size '$PD_TP_SIZE' \
  --pipeline-parallel-size '$PD_PP_SIZE' \
  --distributed-executor-backend ray \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --enable-request-id-headers \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{\"image\":$MAX_IMAGES_PER_PROMPT,\"video\":$MAX_VIDEOS_PER_PROMPT}' \
  --ec-transfer-config '{
    \"ec_connector\":\"ECExampleConnector\",
    \"ec_role\":\"ec_consumer\",
    \"ec_connector_extra_config\":{
      \"shared_storage_path\":\"$EC_SHARED_STORAGE_PATH\"
    }
  }'
" > "$LOG_DIR/pd_vllm.log" 2>&1 &
PD_VLLM_PID=$!

sleep 10

echo "Starting E/PD proxy on $HEAD_NODE port $PORT"
ssh -q "$HEAD_NODE" "
set -euo pipefail
source '$VENV_ACTIVATE'
cd '$VLLM_ROOT/examples/online_serving/disaggregated_encoder'

python3 disagg_epd_proxy.py \
  --host 0.0.0.0 \
  --port '$PORT' \
  --encode-servers-urls 'http://localhost:$ENCODE_PORT' \
  --prefill-servers-urls 'disable' \
  --decode-servers-urls 'http://localhost:$PD_PORT'
" > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PD_RAY_HEAD_SHELL_PID=$PD_RAY_HEAD_PID"
  echo "PD_RAY_WORKER_SHELL_PID=$PD_RAY_WORKER_PID"
  echo "PD_VLLM_SHELL_PID=$PD_VLLM_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "HEAD_NODE=$HEAD_NODE"
  echo "WORKER_NODE=$WORKER_NODE"
  echo "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
  echo "PD_HEAD_CUDA_VISIBLE_DEVICES=$PD_HEAD_CUDA_VISIBLE_DEVICES"
  echo "PD_WORKER_CUDA_VISIBLE_DEVICES=$PD_WORKER_CUDA_VISIBLE_DEVICES"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PD_PORT=$PD_PORT"
  echo "PORT=$PORT"
} > "$LOG_DIR/launcher_pids.txt"

echo "E/PD native vLLM stack launched. Waiting on child processes..."
wait "$ENCODER_PID" "$PD_RAY_HEAD_PID" "$PD_RAY_WORKER_PID" "$PD_VLLM_PID" "$PROXY_PID"
