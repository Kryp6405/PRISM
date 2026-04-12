#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:?usage: capture_env.sh <out_dir>}"
mkdir -p "$OUT_DIR"

now_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "timestamp_utc=$(now_utc)"
  echo "hostname=$(hostname 2>/dev/null || true)"
  echo "pwd=$(pwd)"
  echo "user=${USER:-unknown}"
} > "$OUT_DIR/runtime_context.txt"

# Git metadata
if command -v git >/dev/null 2>&1; then
  git rev-parse HEAD > "$OUT_DIR/git_commit.txt" 2>/dev/null || true
  git branch --show-current > "$OUT_DIR/git_branch.txt" 2>/dev/null || true
  git status --short > "$OUT_DIR/git_status.txt" 2>/dev/null || true
fi

# Python environment
if command -v python >/dev/null 2>&1; then
  python --version > "$OUT_DIR/python_version.txt" 2>&1 || true
fi
if command -v pip >/dev/null 2>&1; then
  pip freeze > "$OUT_DIR/pip_freeze.txt" 2>/dev/null || true
fi

# GPU + scheduler environment
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUT_DIR/nvidia_smi.txt" 2>/dev/null || true
  nvidia-smi topo -m > "$OUT_DIR/nvidia_smi_topo.txt" 2>/dev/null || true
fi

if command -v module >/dev/null 2>&1; then
  module list > "$OUT_DIR/module_list.txt" 2>&1 || true
fi

env | sort > "$OUT_DIR/env_all.txt"
env | sort | grep -E '^(HF_HOME|CUDA|VLLM|DYN|NCCL|SLURM|PATH|PYTHONPATH|CONDA|VIRTUAL_ENV)=' > "$OUT_DIR/env_filtered.txt" || true

# System info
uname -a > "$OUT_DIR/uname.txt" 2>/dev/null || true
lscpu > "$OUT_DIR/lscpu.txt" 2>/dev/null || true
free -h > "$OUT_DIR/free.txt" 2>/dev/null || true

echo "Wrote environment snapshot to $OUT_DIR"
