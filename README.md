# PRISM: Profiling Multimodal Disaggregation for VLM Serving

**Project:** Scaling Large Language Model Inference  
**Maintainers:** Akarsh Srivastava & Krisnajit Rajeshkhanna  

PRISM is a systems project studying **how multimodal inference pipelines behave under different serving architectures**, with a focus on **Vision-Language Models (VLMs)** and **stage disaggregation**.

Our primary goal is to understand:

- when **aggregated serving** is sufficient,
- when **encoder-only disaggregation** helps,
- when **full encoder / prefill / decode disaggregation** becomes worthwhile,
- and how these tradeoffs change with workload shape, concurrency, and hardware constraints.

The current repository serves as a **sandbox and experimental harness** for running and profiling multimodal inference on GPU clusters, starting with a **1-node Perlmutter validation path** and expanding toward more structured experiments.

---

## What PRISM Is About

Large multimodal models do not behave like standard text-only LLMs.

In a Vision-Language Model, a request may involve:
1. **visual encoding** of one or more images,
2. **prefill** over the text-and-vision prompt,
3. **decode** for autoregressive generation.

These stages have different compute and memory characteristics. In some workloads, combining them in one worker is simple and effective. In others, isolating stages may improve throughput, reduce interference, or better utilize available GPU resources.

PRISM studies these tradeoffs in a practical cluster setting.

### Core research question

**Under what workload conditions does multimodal disaggregation improve serving performance over aggregated execution?**

More concretely, we study how pipeline structure affects:

- **Throughput**
- **Time to First Token (TTFT)**
- **Time Between Tokens (TBT)**
- **End-to-end latency**
- **GPU utilization**
- **GPU memory usage**

---

## Current Study Scope

The current phase of PRISM is centered on:

- **Model:** `Qwen/Qwen2-VL-2B-Instruct`
- **Platform:** Perlmutter
- **Initial goal:** establish reliable multimodal serving and benchmarking flows
- **Immediate focus:** small-scale validation and reduced benchmark sweeps before deeper experiments

We use `Qwen2-VL-2B-Instruct` because it is:
- lightweight enough for rapid iteration,
- natively supported by vLLM,
- and suitable for multimodal chat-style benchmarking with synthetic image requests.

---

## Experimental Direction

PRISM is designed around three serving modes:

### 1. Aggregated serving
A single worker handles the full request lifecycle:
- image encoding
- prefill
- decode

This is the simplest deployment model and the baseline for comparison.

### 2. Encoder-only disaggregation
The vision encoder is isolated from the prefill/decode worker.

This tests whether separating multimodal feature extraction reduces contention and improves performance.

### 3. Full E/P/D disaggregation
The pipeline is split into:
- **E**ncoder
- **P**refill
- **D**ecode

This is the most fine-grained configuration and is intended to expose the full cost/benefit tradeoff of disaggregation.

---

## What This Repository Contains

This repository contains the scripts and scaffolding for:

- environment setup on Perlmutter
- building a patched multimodal vLLM stack
- launching serving configurations
- running smoke tests and profiling jobs
- collecting benchmark artifacts and logs

It is not intended to be a polished framework yet. It is primarily a **research harness** for controlled experiments.

---

## Repository Goals by Phase

### Phase 0: Validation / Smoke Testing
Confirm that each serving mode:
- launches successfully,
- serves multimodal requests correctly,
- works with `aiperf`,
- and writes logs/artifacts to expected locations.

### Phase 1: Coarse profiling
Run reduced experiments across:
- serving mode,
- concurrency,
- and generation length

to identify the most promising configurations.

### Phase 2: Focused analysis
Take the best configurations and study:
- latency breakdowns,
- utilization,
- memory behavior,
- and workload sensitivity in more detail.

---

## 🚨 Step 0: Cluster & Environment Prep

Because model weights and virtual environments are massive, **do not set this up in your home directory (`/global/u2/...`).** You will quickly hit your strict storage quota.

**1. Navigate to your scratch space and clone the repo:**
```bash
cd $PSCRATCH
# Replace with the actual URL of the repository
git clone [https://github.com/your-username/PRISM.git](https://github.com/your-username/PRISM.git)
cd PRISM
```

**2. Set your Hugging Face Cache to scratch:**
This forces the multi-gigabyte model weights to download to scratch, protecting your home folder. *(Tip: Add this line to your `~/.bashrc`)*.
```bash
export HF_HOME=$PSCRATCH/.cache/hf_home
```

**3. Initialize a fresh virtual environment:**
Using `uv` is highly recommended for speed on Perlmutter.
```bash
module load python
uv venv vllm-env
source vllm-env/bin/activate
```

## 📦 Step 1: Installation
All critical dependency fixes—including pinning vLLM to the stable `v0.17.0` branch and patching the missing `msgpack` and `aiperf` packages—are handled by the setup script.

With your virtual environment activated, build the stack:
```bash
source scripts/setup_prism_stack.sh
```

## 🚀 Step 2: 1-Node Sandbox Execution
For this phase, we use `Qwen/Qwen2-VL-2B-Instruct` as our test VLM. It is lightweight, natively supported by vLLM, and uses modern `tokenizer.json` formats (which prevents Dynamo frontend routing errors).

**1. Wipe the Slate Clean**
Ensure no zombie processes are blocking port `8000`:
```bash
pkill -P $$ python
```

**2. Start the Orchestrator (Frontend)**
Launch the Dynamo frontend in the background using local file discovery.
```bash
python -m dynamo.frontend \
    --http-port 8000 \
    --discovery-backend file > frontend.log 2>&1 &
```

**3. Start the VLM Worker**
Launch the backend engine.
*Note: Because this is a 1-node test without a dedicated message broker, we explicitly disable KV events to prevent NATS connection crashes. We also explicitly allocate VRAM for images using the required JSON dictionary syntax.*
```bash
python -m dynamo.vllm \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --discovery-backend file \
    --kv-events-config '{"enable_kv_cache_events": false}' \
    --enable-multimodal \
    --limit-mm-per-prompt '{"image": 1}' > worker.log 2>&1 &
```

**4. Verify Worker Initialization**
Do not fire the benchmark until the worker has fully loaded the multimodal weights into the GPU and registered with the frontend.
```bash
tail -f worker.log
```
*Wait until you see Application startup complete or Registered endpoint 'generate', then press Ctrl+C to exit the log view.*

**5. Fire the Benchmark (`aiperf`)**
Send synthetic image data (512x512 pixels) through the pipeline.
*Note: `--use-server-token-count` is required for VLMs. `aiperf` cannot calculate how many tokens an image patch will expand into on the server, so this flag tells the client to trust the server's token math.*

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --image-width-mean 512 \
    --image-height-mean 512 \
    --concurrency 4 \
    --request-count 10 \
    --use-server-token-count
```

**6. Cleanup**
When the metrics table prints successfully, kill the background processes to free up your node allocation and ports:

```bash
pkill -P $$ python
```
