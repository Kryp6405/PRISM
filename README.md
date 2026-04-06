# PRISM Profiling: Multimodal Disaggregation Sandbox

**Project:** Scaling Large Language Model Inference  
**Maintainers:** Akarsh Srivastava & Krisnajit Rajeshkhanna  

This repository contains the setup and benchmarking scripts for profiling a disaggregated Vision-Language Model (VLM) pipeline on Perlmutter. This specific guide covers the 1-node (4x A100) sandbox validation phase using NVIDIA's `ai-dynamo` orchestrator and the `aiperf` benchmarking client.

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
