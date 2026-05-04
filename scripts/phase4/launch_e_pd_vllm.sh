#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 4 E/PD native vLLM launch script — NO RAY
#
# Stable encoder-disaggregated baseline:
#
#   NODE0 GPUs 0,1     -> Encoder vLLM producer
#   NODE0 GPUs 2,3     -> Idle
#   NODE1 GPUs 0,1,2,3 -> P/D vLLM consumer
#
# Proxy:
#   Runs on NODE0 port 8000 so AIPerf can use http://localhost:8000.
#
# This intentionally avoids cross-node P/D Ray placement. Each vLLM service
# is fully contained on one node.
###############################################################################

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"

VLLM_ROOT="${VLLM_ROOT:-$SCRATCH/PRISM_env/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRATCH/PRISM_env/venvs/vllm-env/bin/activate}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-$PSCRATCH/prism_ec_cache_epd}"

###############################################################################
# GPU placement
###############################################################################

# Node0 encoder.
ENCODER_CUDA_VISIBLE_DEVICES="${ENCODER_CUDA_VISIBLE_DEVICES:-0,1}"
ENCODER_TP_SIZE="${ENCODER_TP_SIZE:-2}"

# Node1 P/D.
PD_CUDA_VISIBLE_DEVICES="${PD_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Start with TP=4, PP=1 because P/D is fully inside one node.
# If multimodal embedding merge fails, rerun with:
#   PD_TP_SIZE=1 PD_PP_SIZE=4
PD_TP_SIZE="${PD_TP_SIZE:-4}"
PD_PP_SIZE="${PD_PP_SIZE:-1}"
PD_EXPECTED_GPUS_TOTAL="${PD_EXPECTED_GPUS_TOTAL:-$((PD_TP_SIZE * PD_PP_SIZE))}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
MAX_VIDEOS_PER_PROMPT="${MAX_VIDEOS_PER_PROMPT:-0}"
LIMIT_MM_PER_PROMPT="{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}"

LOG_DIR="${LOG_DIR:-./artifacts/p4/e_pd_native_vllm_logs}"
mkdir -p "$LOG_DIR"

CLUSTER_ENV_FILE="${LOG_DIR}/cluster_env.sh"

###############################################################################
# Slurm node resolution
###############################################################################

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

CLIENT_NODE="${NODES[0]}"
ENCODER_NODE="${NODES[0]}"
PD_NODE="${NODES[1]}"

CLIENT_IP="$(ssh -q "$CLIENT_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"
PD_IP="$(ssh -q "$PD_NODE" hostname --ip-address 2>/dev/null | awk '{print $1}')"

if [[ -z "$CLIENT_IP" || -z "$PD_IP" ]]; then
  echo "ERROR: Could not resolve CLIENT_IP or PD_IP." >&2
  echo "CLIENT_NODE=$CLIENT_NODE CLIENT_IP=$CLIENT_IP" >&2
  echo "PD_NODE=$PD_NODE PD_IP=$PD_IP" >&2
  exit 1
fi

: "${VLLM_ROOT:?VLLM_ROOT is empty}"
: "${VENV_ACTIVATE:?VENV_ACTIVATE is empty}"

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

###############################################################################
# Logging
###############################################################################

echo "Launching Phase 4 E/PD native vLLM: 2E + 4PD + 2 idle, no Ray"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "ENCODE_PORT=$ENCODE_PORT"
echo "PD_PORT=$PD_PORT"
echo "CLIENT_NODE=$CLIENT_NODE"
echo "ENCODER_NODE=$ENCODER_NODE"
echo "PD_NODE=$PD_NODE"
echo "CLIENT_IP=$CLIENT_IP"
echo "PD_IP=$PD_IP"
echo "VLLM_ROOT=$VLLM_ROOT"
echo "VENV_ACTIVATE=$VENV_ACTIVATE"
echo "EC_SHARED_STORAGE_PATH=$EC_SHARED_STORAGE_PATH"
echo "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
echo "ENCODER_TP_SIZE=$ENCODER_TP_SIZE"
echo "PD_CUDA_VISIBLE_DEVICES=$PD_CUDA_VISIBLE_DEVICES"
echo "PD_TP_SIZE=$PD_TP_SIZE"
echo "PD_PP_SIZE=$PD_PP_SIZE"
echo "PD_EXPECTED_GPUS_TOTAL=$PD_EXPECTED_GPUS_TOTAL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "LIMIT_MM_PER_PROMPT=$LIMIT_MM_PER_PROMPT"
echo "LOG_DIR=$LOG_DIR"

{
  echo "Allocated Slurm nodes:"
  printf '  %s\n' "${NODES[@]}"
  echo
  echo "Layout:"
  echo "  $ENCODER_NODE GPUs $ENCODER_CUDA_VISIBLE_DEVICES -> Encoder"
  echo "  $ENCODER_NODE GPUs 2,3 -> Idle"
  echo "  $PD_NODE GPUs $PD_CUDA_VISIBLE_DEVICES -> P/D"
  echo
  echo "No Ray is used. Encoder and P/D are separate node-local vLLM services."
} | tee "$LOG_DIR/slurm_nodes.txt"

cat > "$CLUSTER_ENV_FILE" <<EOF
export MODE="e_pd"
export CLIENT_NODE="$CLIENT_NODE"
export ENCODER_NODE="$ENCODER_NODE"
export PD_NODE="$PD_NODE"
export CLIENT_IP="$CLIENT_IP"
export PD_IP="$PD_IP"
export PORT="$PORT"
export ENCODE_PORT="$ENCODE_PORT"
export PD_PORT="$PD_PORT"
export MODEL="$MODEL"
export VLLM_ROOT="$VLLM_ROOT"
export EC_SHARED_STORAGE_PATH="$EC_SHARED_STORAGE_PATH"
export ENCODER_CUDA_VISIBLE_DEVICES="$ENCODER_CUDA_VISIBLE_DEVICES"
export ENCODER_TP_SIZE="$ENCODER_TP_SIZE"
export PD_CUDA_VISIBLE_DEVICES="$PD_CUDA_VISIBLE_DEVICES"
export PD_TP_SIZE="$PD_TP_SIZE"
export PD_PP_SIZE="$PD_PP_SIZE"
export PD_EXPECTED_GPUS_TOTAL="$PD_EXPECTED_GPUS_TOTAL"
export LIMIT_MM_PER_PROMPT='$LIMIT_MM_PER_PROMPT'
EOF

echo "Wrote cluster env to $CLUSTER_ENV_FILE"

###############################################################################
# Remote env helper
###############################################################################

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

###############################################################################
# Cleanup
###############################################################################

cleanup() {
  set +e
  echo "Stopping E/PD native vLLM stack..."

  for node in "$CLIENT_NODE" "$PD_NODE"; do
    ssh -q "$node" "
      source '$VENV_ACTIVATE' >/dev/null 2>&1 || true

      pkill -f 'disagg_epd_proxy.py' >/dev/null 2>&1 || true
      pkill -f 'vllm serve' >/dev/null 2>&1 || true
      pkill -f 'vllm.entrypoints.openai.api_server' >/dev/null 2>&1 || true
      pkill -f 'VLLM::EngineCore' >/dev/null 2>&1 || true
      pkill -f 'RayWorkerWrapper' >/dev/null 2>&1 || true
      ray stop --force >/dev/null 2>&1 || true
    " 2>/dev/null || true
  done
}

trap cleanup EXIT

echo "Cleaning existing processes..."
cleanup
sleep 5

###############################################################################
# Start encoder on node0 GPUs 0,1
###############################################################################

echo "Starting encoder vLLM producer on $ENCODER_NODE GPUs $ENCODER_CUDA_VISIBLE_DEVICES"

ssh -q "$ENCODER_NODE" "
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
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT' \
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

###############################################################################
# Start P/D on node1 GPUs 0,1,2,3 — no Ray
###############################################################################

echo "Starting P/D vLLM consumer on $PD_NODE GPUs $PD_CUDA_VISIBLE_DEVICES"

ssh -q "$PD_NODE" "
$(remote_env_prefix)

CUDA_VISIBLE_DEVICES='$PD_CUDA_VISIBLE_DEVICES' vllm serve '$MODEL' \
  --host 0.0.0.0 \
  --port '$PD_PORT' \
  --tensor-parallel-size '$PD_TP_SIZE' \
  --pipeline-parallel-size '$PD_PP_SIZE' \
  --max-model-len '$MAX_MODEL_LEN' \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --enable-request-id-headers \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT' \
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

###############################################################################
# Start proxy on node0
###############################################################################

echo "Starting E/PD proxy on $CLIENT_NODE port $PORT"

ssh -q "$CLIENT_NODE" "
set -euo pipefail
source '$VENV_ACTIVATE'
cd '$VLLM_ROOT/examples/online_serving/disaggregated_encoder'

python3 disagg_epd_proxy.py \
  --host 0.0.0.0 \
  --port '$PORT' \
  --encode-servers-urls 'http://localhost:$ENCODE_PORT' \
  --prefill-servers-urls 'disable' \
  --decode-servers-urls 'http://$PD_IP:$PD_PORT'
" > "$LOG_DIR/proxy.log" 2>&1 &

PROXY_PID=$!

###############################################################################
# Save PIDs
###############################################################################

{
  echo "ENCODER_SHELL_PID=$ENCODER_PID"
  echo "PD_VLLM_SHELL_PID=$PD_VLLM_PID"
  echo "PROXY_SHELL_PID=$PROXY_PID"
  echo "CLIENT_NODE=$CLIENT_NODE"
  echo "ENCODER_NODE=$ENCODER_NODE"
  echo "PD_NODE=$PD_NODE"
  echo "CLIENT_IP=$CLIENT_IP"
  echo "PD_IP=$PD_IP"
  echo "ENCODER_CUDA_VISIBLE_DEVICES=$ENCODER_CUDA_VISIBLE_DEVICES"
  echo "ENCODER_TP_SIZE=$ENCODER_TP_SIZE"
  echo "PD_CUDA_VISIBLE_DEVICES=$PD_CUDA_VISIBLE_DEVICES"
  echo "PD_TP_SIZE=$PD_TP_SIZE"
  echo "PD_PP_SIZE=$PD_PP_SIZE"
  echo "ENCODE_PORT=$ENCODE_PORT"
  echo "PD_PORT=$PD_PORT"
  echo "PORT=$PORT"
} > "$LOG_DIR/launcher_pids.txt"

echo "E/PD native vLLM stack launched. Waiting on child processes..."
wait "$ENCODER_PID" "$PD_VLLM_PID" "$PROXY_PID"
