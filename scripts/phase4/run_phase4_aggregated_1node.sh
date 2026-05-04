#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PHASE="${PHASE:-p4_7b}"
MODE_NAME="aggregated-1node-native-vllm"
LAUNCH_CMD="${LAUNCH_CMD:-bash scripts/phase4/launch_aggregated_1node_vllm.sh}"

source scripts/phase4/run_phase4_1node_common.sh
