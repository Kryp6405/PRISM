# PRISM: Profiling Resource Inference & Scaling for  Multimodality

**Project:** Scaling Large Language Model Inference  
**Maintainers:** Akarsh Srivastava & Krisnajit Rajeshkhanna

PRISM is a systems project studying how **multimodal inference pipelines behave under different serving architectures**, with a focus on **Vision-Language Models (VLMs)** and **stage disaggregation**.

The project studies when it is better to serve a VLM as one monolithic/aggregated deployment versus splitting the request path into separate stages:

- **Aggregated serving**
- **Encoder / Prefill+Decode disaggregation (E/PD)**
- **Encoder / Prefill / Decode disaggregation (E/P/D)**

By Phase 4, the project moved away from Dynamo and uses **native vLLM serving plus NVIDIA AIPerf** for the final benchmarking harness.

---

## Research Question

Large multimodal models do not behave exactly like text-only LLMs. A VLM request may include:

1. **Visual encoding** of one or more images,
2. **Prefill** over the text-and-vision prompt,
3. **Decode** for autoregressive generation.

These stages have different compute, memory, and communication characteristics. PRISM asks:

> Under what workload and hardware conditions does multimodal stage disaggregation improve VLM serving performance over aggregated execution?

More concretely, PRISM studies how pipeline structure affects:

- End-to-end request latency
- Tail latency, especially p90 and p99
- Request throughput
- Output token throughput
- Image throughput
- GPU memory usage
- GPU utilization
- Stage placement efficiency

---

## Current Study Scope

The current experimental harness focuses on native vLLM serving and AIPerf profiling.

Primary model and platform:

- **Main model:** `Qwen/Qwen2.5-VL-32B-Instruct`
- **Control models:** `Qwen/Qwen2.5-VL-7B-Instruct`, optionally `Qwen/Qwen2.5-VL-14B-Instruct`
- **Platform:** NERSC Perlmutter
- **Benchmarking:** NVIDIA AIPerf
- **Serving stack:** native vLLM
- **Final phase:** Phase 4

Earlier Dynamo experiments were useful for initial exploration, but the final benchmark path uses native vLLM directly.

---

## Serving Modes

### 1. Aggregated Serving

A single vLLM deployment handles the full request lifecycle:

- image encoding
- prefill
- decode

This is the baseline architecture.

For the main 32B two-node experiment, the aggregated setup used:

```text
AGG:
  8 GPUs total
  TP=4, PP=2
```

For the one-node 7B/14B controls:

```text
AGG:
  4 GPUs total
  TP=4, PP=1
```

---

### 2. Encoder / Prefill+Decode Disaggregation: E/PD

The encoder is separated from the combined prefill/decode backend.

Conceptually:

```text
AIPerf -> E/PD proxy -> Encoder -> P/D
```

For the main 32B two-node experiment, the stable configuration was:

```text
E/PD:
  Encoder: TP=2
  P/D:     TP=4, PP=1
```

This used 6 active GPUs in the stable configuration:

```text
Node0:
  GPUs 0,1 -> Encoder
  GPUs 2,3 -> idle

Node1:
  GPUs 0,1,2,3 -> P/D
```

This was intentionally chosen to avoid unstable cross-node P/D placement and problematic tensor-parallel multimodal merge behavior.

For one-node 7B/14B controls, the safer E/PD configuration is:

```text
E/PD:
  Encoder: TP=2
  P/D:     TP=1, PP=2
```

`P/D TP=2` can trigger a multimodal embedding merge device mismatch in vLLM.

---

### 3. Full Encoder / Prefill / Decode Disaggregation: E/P/D

The pipeline is split into separate encoder, prefill, and decode services.

Conceptually:

```text
AIPerf -> E/P/D proxy -> Encoder -> Prefill -> Decode
```

For the main 32B two-node experiment, stable/final E/P/D used compatible Prefill/Decode layouts:

```text
E/P/D:
  Encoder: TP=2, PP=1
  Prefill: TP=1, PP=2
  Decode:  TP=2, PP=2
```

A previous version used:

```text
Encoder: TP=2
Prefill: TP=1
Decode:  TP=4
```

but this left one GPU idle and bottlenecked Prefill. Using `PP=2` for both Prefill and Decode fixed NIXL layout compatibility issues and allowed an 8-GPU E/P/D configuration.

For one-node 7B/14B controls:

```text
E/P/D:
  Encoder: TP=1
  Prefill: TP=1
  Decode:  TP=2
```

---

## Workloads

Phase 4 uses synthetic multimodal AIPerf workloads with fixed concurrency and request count.

Current standard workloads:

### Baseline

```text
image: 512x512
output tokens: 64
concurrency: 4
request count: 50
```

### Encoder-heavy

```text
image: 2048x2048
output tokens: 128
concurrency: 4
request count: 50
```

### Decode-heavy

```text
image: 512x512
output tokens: 1024
concurrency: 4
request count: 50
```

For some very slow aggregated 32B decode-heavy runs, request count may be reduced and should be reported explicitly in plots/tables.

Important AIPerf flag:

```bash
--extra-inputs ignore_eos:true
```

This forces the model to generate up to the requested output length rather than stopping early at EOS. Without this flag, decode-heavy workloads may generate far fewer tokens than requested.

---

## Repository Structure

Typical Phase 4 scripts are under:

```text
scripts/phase4/
```

Typical workload configs are under:

```text
src/phase4/workloads/
```

Typical artifact directories:

```text
artifacts/p4/
artifacts/p4_7b/
artifacts/p4_14b/
```

Example outputs:

```text
artifacts/p4/baseline/
artifacts/p4/encoder_heavy/
artifacts/p4/decode_heavy/
```

Each run writes logs, AIPerf exports, workload copies, and optional GPU telemetry.

---

## Environment Setup on Perlmutter

Use scratch, not the home directory, for model weights, venvs, and artifacts.

```bash
cd $PSCRATCH
git clone <repo-url> PRISM
cd PRISM
```

Set Hugging Face cache to scratch:

```bash
export HF_HOME=$PSCRATCH/.cache/hf_home
export HF_TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

Set up the Python/vLLM environment according to the project setup script or your current Perlmutter vLLM environment:

```bash
source $SCRATCH/PRISM_env/venvs/vllm-env/bin/activate
```

The final Phase 4 experiments assume:

- vLLM is installed and available as `vllm`
- AIPerf is installed and available as `aiperf`
- the target Qwen2.5-VL models are accessible from Hugging Face
- the run is inside a Slurm allocation with the expected number of GPUs

---

## Running Phase 4 Experiments

### 32B Two-Node Experiments

Use the Phase 4 two-node launch/run scripts.

Example aggregated run:

```bash
MODEL="Qwen/Qwen2.5-VL-32B-Instruct" \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
GPU_TELEMETRY_MODE=none \
ENABLE_STREAMING=false \
AIPERF_EXTRA_INPUTS="ignore_eos:true" \
bash scripts/phase4/run_phase4_aggregated.sh
```

Example E/PD run:

```bash
MODEL="Qwen/Qwen2.5-VL-32B-Instruct" \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
ENCODER_TP_SIZE=2 \
PD_TP_SIZE=4 \
PD_PP_SIZE=1 \
GPU_TELEMETRY_MODE=none \
ENABLE_STREAMING=false \
AIPERF_EXTRA_INPUTS="ignore_eos:true" \
bash scripts/phase4/run_phase4_e_pd.sh
```

Example E/P/D run:

```bash
MODEL="Qwen/Qwen2.5-VL-32B-Instruct" \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
ENCODER_TP_SIZE=2 \
PREFILL_TP_SIZE=1 \
PREFILL_PP_SIZE=2 \
DECODE_TP_SIZE=2 \
DECODE_PP_SIZE=2 \
GPU_TELEMETRY_MODE=none \
ENABLE_STREAMING=false \
AIPERF_EXTRA_INPUTS="ignore_eos:true" \
bash scripts/phase4/run_phase4_e_p_d.sh
```

---

### 7B / 14B One-Node Controls

The one-node controls use the same workload configs but write to separate artifact roots through the `PHASE` variable.

Example 7B aggregated:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct" \
PHASE=p4_7b \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
bash scripts/phase4/run_phase4_7b_aggregated_1node.sh
```

Example 7B E/PD:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct" \
PHASE=p4_7b \
PD_TP_SIZE=1 \
PD_PP_SIZE=2 \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
bash scripts/phase4/run_phase4_7b_e_pd_1node.sh
```

Example 7B E/P/D:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct" \
PHASE=p4_7b \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
bash scripts/phase4/run_phase4_7b_e_p_d_1node.sh
```

For 14B, only change the model and phase:

```bash
MODEL="Qwen/Qwen2.5-VL-14B-Instruct" \
PHASE=p4_14b \
WORKLOAD_CONFIG=src/phase4/workloads/baseline.json \
bash scripts/phase4/run_phase4_7b_aggregated_1node.sh
```

The script names may say `7b`, but they read `MODEL` and `PHASE` from the environment.

---

## Plotting Results

Use the Phase 4 plotting script:

```bash
python src/phase4/plot_stage_comparison.py \
  --artifact-root artifacts/p4 \
  --show-table
```

For one-node 7B controls:

```bash
python src/phase4/plot_stage_comparison.py \
  --artifact-root artifacts/p4_7b \
  --show-table
```

For one-node 14B controls:

```bash
python src/phase4/plot_stage_comparison.py \
  --artifact-root artifacts/p4_14b \
  --show-table
```

Plots are written under:

```text
artifacts/<phase>/<workload>/plots/
```

For example:

```text
artifacts/p4/baseline/plots/
artifacts/p4/encoder_heavy/plots/
artifacts/p4/decode_heavy/plots/
```

The plotting script reads exact AIPerf JSON fields such as:

```text
input_sequence_length.avg
output_sequence_length.avg
request_latency.avg
request_latency.p90
request_latency.p99
request_count.avg
```

This avoids accidentally using total token fields such as `total_isl`, `total_osl`, or `total_usage_prompt_tokens`.

---

## GPU Telemetry

The run scripts can collect lightweight GPU telemetry using `nvidia-smi`.

Useful telemetry plots include:

- GPU utilization over time
- GPU memory usage over time
- average GPU utilization by GPU
- average GPU utilization by stage
- memory footprint by stage
- node-level utilization over time for two-node runs

Telemetry is most useful for supporting systems-level claims such as:

- Encoder is often less memory-heavy than Prefill/Decode.
- Decode dominates long-output workloads.
- Disaggregation can concentrate work on the relevant stage.
- Aggregated multi-node serving may use more GPUs but still suffer from synchronization or communication overhead.

Telemetry should be interpreted as supporting evidence, not as a perfect fine-grained profiler. `nvidia-smi` sampling can miss short bursts.

---

## Known Issues and Lessons Learned

### Dynamo was not used in final Phase 4 results

Earlier versions of this project explored Dynamo-based serving. By Phase 4, the benchmark harness moved to native vLLM plus AIPerf because it provided a more direct and controllable path for the serving modes we wanted to compare.

### Multimodal TP can fail in some stage configurations

Several failures were caused by vLLM attempting to merge multimodal embeddings across tensor-parallel ranks, producing errors like:

```text
Expected all tensors to be on the same device,
but got source is on cuda:0, different from other tensors on cuda:1
```

This affected configurations such as:

```text
P TP > 1
P/D TP > 1 in some E/PD layouts
```

Safer alternatives used `PP` instead of `TP` for the affected stage when possible.

### NIXL requires compatible Prefill and Decode layouts

In E/P/D, Prefill and Decode KV-transfer layouts must be compatible. For example, `P PP=2` with `D PP=1` can fail NIXL metadata validation. Matching Prefill and Decode PP layouts, such as:

```text
P: TP=1, PP=2
D: TP=2, PP=2
```

resolved this for the final 8-GPU E/P/D configuration.

### Aggregated two-node serving can be much slower

For the 32B two-node setup, aggregated serving with `TP=4, PP=2` was significantly slower than disaggregated serving. This should not be interpreted as “aggregation is always bad” or “Ray is always slow.” The likely issue is the combination of:

- Ray/distributed execution overhead
- cross-node pipeline parallelism
- tensor-parallel communication
- per-token synchronization during decode
- VLM multimodal path overhead under distributed execution

The one-node 7B/14B controls help separate disaggregation benefits from multi-node overhead.

---

## Current Interpretation

The current results support a scale-dependent story:

```text
7B one-node:
  Aggregated serving can win because the model is small enough that disaggregation overhead dominates.

14B one-node:
  Useful intermediate control to test the crossover point.

32B two-node:
  Disaggregation can substantially outperform aggregated TP/PP serving because stage-aware placement avoids expensive monolithic distributed execution.
```

The strongest result is not simply that one mode always wins. Instead, PRISM shows that VLM serving performance depends on:

- model size,
- workload shape,
- output length,
- image resolution,
- stage placement,
- TP/PP layout,
- and whether communication occurs inside the per-token decode loop.

---

## Cleanup

To stop local vLLM/proxy processes inside an allocation:

```bash
pkill -f 'disagg_epd_proxy.py' || true
pkill -f 'vllm serve' || true
pkill -f 'vllm.entrypoints.openai.api_server' || true
pkill -f 'VLLM::EngineCore' || true
pkill -f 'RayWorkerWrapper' || true
ray stop --force || true
```

If ports are stuck, check:

```bash
lsof -i :8000
lsof -i :19534
lsof -i :19535
lsof -i :19536
```

---

## Status

PRISM is currently a research harness rather than a polished serving framework. The repository is optimized for controlled experimentation, reproducible script-based launches, and collecting artifacts for analysis.

The final Phase 4 path is:

```text
native vLLM serving + AIPerf benchmarking + GPU telemetry
```
