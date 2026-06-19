# Docker

The `harmony-bench:cu129` image bundles vLLM, SGLang, and llama-cpp-turboquant for unified benchmarking. All commands in this doc use this tag — replace with your own registry path if you publish to a private registry.

### What's Included

| Engine | Version | Source |
|--------|---------|--------|
| vLLM | 0.23.0 | Official vLLM image (inherited) |
| SGLang | 0.5.8 | PyPI (isolated venv) |
| flashinfer-python | 0.6.1 | PyPI (FlashInfer wheel index) |
| llama-cpp-turboquant | Latest | Built from submodule in Docker |

### Architecture

Multi-stage Docker build:
1. **Stage 1** (`turboquant-build`): Builds llama-cpp-turboquant from `third_party/llama-cpp-turboquant/` submodule using CUDA 12.9 devel image
2. **Stage 2** (`runtime`): Uses official vLLM image as base, installs SGLang in isolated venv, copies llama binaries from Stage 1

**Separate Python environments:** vLLM and SGLang are incompatible in the same Python environment due to flashinfer and PyTorch ABI conflicts. vLLM uses the system Python (from base image), SGLang uses `/opt/venv-sglang/`. Each engine is launched as its own subprocess — they never share a process.

| Engine | Python Environment | Launch Command |
|--------|-------------------|----------------|
| vLLM | System (`python3`) | `vllm serve <model> --port 8000` |
| SGLang | `/opt/venv-sglang/` | `/opt/venv-sglang/bin/python -m sglang.launch_server --model <model> --port 8002` |
| llama.cpp | N/A (native binary) | `llama-server -m <model.gguf> --port 8001` |

Container behavior: Init-only toolbox — starts Redis/SSH, prints framework summary, sleeps forever. Users launch engines via `docker exec`.

---

## Quick Start

### 1. Build the Image

```bash
# Default (CUDA 12.9, compute capability 8.6 = A40/RTX 3090)
docker build -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# For RTX 4090 / Ada
docker build --build-arg CUDA_ARCH=8.9 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# For H100 / H200
docker build --build-arg CUDA_ARCH=9.0 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# Override versions
docker build --build-arg SGLANG_VERSION=0.5.8 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
```

### 2. Run a Model

Start the container with GPU access and models mounted:

```bash
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -it harmony-bench:cu129
```

Inside the container, launch any engine:

```bash
# vLLM (HuggingFace models, port 8000)
vllm serve /workspace/models/hf/qwen2.5-0.5b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1

# SGLang (HuggingFace models, port 8002)
/opt/venv-sglang/bin/python -m sglang.launch_server --model /workspace/models/hf/qwen2.5-0.5b --port 8002 --host 0.0.0.0 --mem-fraction-static 0.1

# llama-server (GGUF models, port 8001)
llama-server -m /workspace/models/gguf/qwen2.5-0.5b/ggml-model.gguf --port 8001 --host 0.0.0.0
```

### 3. Expose Ports to Host

To access the server from outside the container, add `-p` flag:

```bash
# vLLM on host port 8000
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "vllm serve /workspace/models/hf/qwen2.5-0.5b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1"
```

### 4. Test the Server

```bash
# Health check
curl http://localhost:8000/health

# List models
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Chat completion
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/workspace/models/hf/qwen2.5-0.5b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}' | python3 -m json.tool
```

---

## Running Multiple Engines Simultaneously

All three engines share the same GPU. Use memory-limiting flags to keep each engine's VRAM footprint small:

| Engine | Memory Control | Effect |
|--------|---------------|--------|
| vLLM | `--gpu-memory-utilization 0.1` | Caps GPU memory at 10% of total VRAM |
| SGLang | `--mem-fraction-static 0.1` | Allocates 10% of total VRAM for static memory |
| llama.cpp | `-np 1 -c 1024 -ctk q8_0 -ctv turbo4` | Minimal KV cache via small context + TurboQuant quantization |

```bash
# Terminal 1: vLLM (10% GPU memory)
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "vllm serve /workspace/models/hf/qwen2.5-0.5b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1"

# Terminal 2: SGLang (10% GPU memory)
docker run --rm --gpus all --ipc=host -p 8002:8002 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "/opt/venv-sglang/bin/python -m sglang.launch_server --model /workspace/models/hf/qwen2.5-0.5b --port 8002 --host 0.0.0.0 --mem-fraction-static 0.1"

# Terminal 3: llama.cpp (minimal KV cache, GGUF model required)
docker run --rm --gpus all --ipc=host -p 8001:8001 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "llama-server -m /workspace/models/gguf/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8001 --host 0.0.0.0 -np 1 -c 1024 -ctk q8_0 -ctv turbo4 -fa on"
```

**Note:** Only run one inference engine at a time for benchmarking. Running multiple engines simultaneously is for testing connectivity only.

---

## Host vLLM Server for Benchmarking

Complete workflow: start container → launch vLLM → benchmark from host → stop.

### 1. Start Container

Choose one mode:

**Background mode** (container stays alive, `docker exec` for everything):
```bash
docker run -d --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  --name harmony-bench \
  harmony-bench:cu129 sleep infinity
```

**Foreground mode** (server output visible, needs second terminal for benchmarks):
```bash
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  --name harmony-bench \
  harmony-bench:cu129 \
  bash -c "vllm serve <model_path> --port 8000 --host 0.0.0.0 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --served-model-name <HF_model_id>"
```

### 2. Launch vLLM Server (Background Mode)

```bash
docker exec -d harmony-bench vllm serve <model_path> --port 8000 --host 0.0.0.0 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --served-model-name <HF_model_id>
```

**Flags explained:**

| Flag | Purpose | Recommended |
|------|---------|-------------|
| `--max-model-len` | Max context length (KV cache) | 4096–16384 |
| `--gpu-memory-utilization` | % GPU memory for KV cache | 0.85–0.95 (higher for small models) |
| `--served-model-name` | HF-style model name reported by /v1/models | Enables `-m`-free benchmarking |

Without `--served-model-name`, the server reports the local path (e.g., `/workspace/models/hf/qwen2.5-0.5b`) and benchmarking requires `-m <HF_model_id>`. Setting it to the HF model ID removes that requirement.

### 3. Wait for Server Ready

```bash
docker exec harmony-bench bash -c \
  'while ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done && echo "vLLM ready"'
```

### 4. Test Connectivity

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<served_model_name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

### 5. Run Benchmarks

**Native llama-benchy sweeps (concurrency + prompt ladders):**
```bash
.venv/bin/python -c "import matplotlib" 2>/dev/null || uv add --group benchmark matplotlib numpy

ulimit -n 65536  # Required for high CCU (default 1024 fd limit)

./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1
```

**Cross-sweep (CCU ladder at each prompt size to measure KV cache pressure):**
```bash
ulimit -n 65536

./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 \
  --cross-sweep --early-exit \
  --ccu-mode add --ccu-start 1 --ccu-max 2001 --ccu-step 100 \
  --prompt-start 2048 --prompt-max 16384
```

**Visualize cross-sweep results:**
```bash
.venv/bin/python scripts/visualize_cross_sweep.py results/vllm/<session_dir>/
```

Outputs: `cross_sweep_table.md`, `ccu_ladder_pp*.png`, `ccu_vs_prompt.png`

### 6. Stop Container

```bash
docker stop harmony-bench   # --rm flag auto-deletes container
```

---

## Benchmarking

### Using `bench-models.sh` (Recommended)

The orchestrator script reads `configs/models.yaml` and runs benchmarks for all enabled models:

```bash
# Preview what will run
./scripts/bench-models.sh --dry-run

# Run all models, all backends
./scripts/bench-models.sh

# Run specific phase (p0 = 0.5B models)
./scripts/bench-models.sh -p p0

# Run specific backend only
./scripts/bench-models.sh -b vllm
./scripts/bench-models.sh -b llamacpp

# Run specific model
./scripts/bench-models.sh -m qwen0.5b

# Light load only
./scripts/bench-models.sh --bench-phase p1

# Stress test (incremental concurrency ramp)
./scripts/bench-models.sh --bench-type stress
```

**Prerequisites:**
1. Models downloaded to `models/` directory
2. Benchmark dataset at `datasets/sharegpt.json`
3. `configs/models.yaml` configured for your GPU setup
4. `.env` file with benchmark parameters (see below)

### Using Individual Benchmark Scripts

For benchmarking an already-running server:

```bash
# vLLM benchmark
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000 -m /workspace/models/hf/qwen2.5-0.5b --full

# SGLang benchmark
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002 -m /workspace/models/hf/qwen2.5-0.5b --full

# llama.cpp benchmark
./scripts/benchmark/llamacpp_bench.sh -u http://localhost:8001/v1 -m /workspace/models/gguf/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Running Benchmarks Inside Docker

Since `vllm` is installed inside the container, run benchmarks from there:

```bash
docker run --rm --network host --gpus all \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/.env:/workspace/.env \
  harmony-bench:cu129 \
  bash -c "cd /workspace && ./scripts/benchmark/vllm_bench.sh -u http://localhost:8000 -m /workspace/models/hf/qwen2.5-0.5b --full"
```

### Using `docker-bench.sh` (Host-Side Wrapper)

`docker-bench.sh` runs benchmarks from the host by forwarding commands to a running container via `docker exec`:

```bash
# Start the container
docker run -d --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/datasets:/workspace/datasets \
  --name harmony-bench harmony-bench:cu129 sleep infinity

# Run benchmarks from host
./scripts/docker/docker-bench.sh -- -b llamacpp
./scripts/docker/docker-bench.sh -- -b vllm --ccu-mode mul --ccu-max 256
./scripts/docker/docker-bench.sh -- -b sglang --full
./scripts/docker/docker-bench.sh --container my-bench -- -b llamacpp

# Or run inside container directly
docker exec -it harmony-bench ./scripts/benchmark/bench.sh -b llamacpp
```

### Using `bench.sh` (Unified Dispatcher)

`bench.sh` dispatches to per-backend scripts (`llamacpp_bench.sh`, `vllm_bench.sh`, `sglang_bench.sh`):

```bash
./scripts/benchmark/bench.sh -b llamacpp
./scripts/benchmark/bench.sh -b vllm --native
./scripts/benchmark/bench.sh -b sglang --full
```

---

## New Machine Setup Checklist

When deploying on a new machine, update these files:

### 1. `configs/models.yaml`

Update the `cluster` section to match your GPU setup:

```yaml
cluster:
  gpu_count: 0              # 0 = auto-detect, or set manually
  vllm:
    tp: 1                   # Tensor parallel size (match GPU count for large models)
    gpu_mem_util: "0.85"    # GPU memory utilization (0.0-1.0)
    max_model_len: 4096     # Max context length
  llamacpp:
    n_parallel: 4           # Concurrent slots
    ctx_size: 4096          # Context window
    tensor_split: ""        # Empty = auto, or "1,1" for 2 GPUs
```

### 2. `.env`

Update benchmark parameters:

```bash
# Dataset path
VLLM_BENCH_DATASET_PATH=datasets/sharegpt.json
SGLANG_BENCH_DATASET_PATH=datasets/sharegpt.json

# Model paths (must match actual model location in container)
VLLM_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.5b
SGLANG_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.5b

# Server URLs
VLLM_BENCH_URL=http://localhost:8000
SGLANG_BENCH_URL=http://localhost:8002
LLAMA_BENCH_URL=http://localhost:8001/v1
```

### 3. Dockerfile Build Args

Match your GPU architecture:

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Build with correct arch
docker build --build-arg CUDA_ARCH=8.6 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
```

| GPU | Compute Capability | CUDA_ARCH |
|-----|-------------------|-----------|
| T4, RTX 2080 | 7.5 | `7.5` |
| A100 | 8.0 | `8.0` |
| A40, RTX 3090 | 8.6 | `8.6` |
| RTX 4090, L40 | 8.9 | `8.9` |
| H100, H200 | 9.0 | `9.0` |
| RTX PRO 6000 Blackwell | 10.0 | `10.0` |

### 4. Models

Download models before benchmarking:

```bash
uv run python scripts/download-models.py
```

Or mount an existing models directory:

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/models:/workspace/models \
  harmony-bench:cu129
```

### 5. Benchmark Dataset

Ensure `datasets/sharegpt.json` exists:

```bash
./scripts/prepare-dataset.sh
```

---

## Verify

```bash
# Check all engines
docker run --rm harmony-bench:cu129 python3 -c "import vllm; print(f'vllm={vllm.__version__}')"
docker run --rm harmony-bench:cu129 /opt/venv-sglang/bin/python -c "import sglang; print(f'sglang={sglang.__version__}')"
docker run --rm harmony-bench:cu129 bash -c "command -v llama-server"

# GPU visibility
docker run --rm --gpus all harmony-bench:cu129 python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Startup summary
timeout 10 docker run --rm harmony-bench:cu129 2>&1 || true
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | Restrict GPU visibility |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `PUBLIC_KEY` | (empty) | SSH public key injection |
| `BENCHMARK_REPO` | (empty) | Git repo to clone on startup |
| `VLLM_BENCH_MODEL` | (required) | Model path for vLLM benchmark |
| `VLLM_BENCH_URL` | `http://localhost:8000` | vLLM server URL |
| `VLLM_BENCH_DATASET_PATH` | (required) | Path to sharegpt.json |
| `SGLANG_BENCH_MODEL` | (required) | Model path for SGLang benchmark |
| `SGLANG_BENCH_URL` | `http://localhost:8002` | SGLang server URL |
| `SGLANG_BENCH_DATASET_PATH` | (required) | Path to sharegpt.json |
| `LLAMA_BENCH_URL` | `http://localhost:8001/v1` | llama.cpp server URL |

---

## Image Size

~57 GB (includes vLLM, SGLang, flashinfer, llama-cpp-turboquant, CUDA runtime)

---

## Troubleshooting

### Port already allocated
```bash
docker stop $(docker ps -q) 2>/dev/null
```

### Model not found (404)
Check the model name matches what the server reports:
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### vLLM import error (ABI mismatch)
The SGLang venv is isolated — vLLM uses system Python. Do NOT run vLLM from the SGLang venv.

### Flashinfer version mismatch
The Dockerfile force-reinstalls flashinfer to prevent conflicts. If you see version errors, rebuild the image.

### Benchmark script can't find vllm
Run benchmarks from inside the Docker container (vllm is installed there, not on the host).
