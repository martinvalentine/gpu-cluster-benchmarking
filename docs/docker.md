# Docker

## Harmony Benchmark Image

A single Docker image bundling vLLM, SGLang, and llama-cpp-turboquant for unified benchmarking.

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
| SGLang | `/opt/venv-sglang/` | `/opt/venv-sglang/bin/python -m sglang.launch_server --model <model> --port 8003` |
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
docker build --build-arg SGLANG_VERSION=0.5.9 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
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
vllm serve /workspace/models/hf/qwen2.5-0.6b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1

# SGLang (HuggingFace models, port 8003)
/opt/venv-sglang/bin/python -m sglang.launch_server --model /workspace/models/hf/qwen2.5-0.6b --port 8003 --host 0.0.0.0 --mem-fraction-static 0.1

# llama-server (GGUF models, port 8001)
llama-server -m /workspace/models/gguf/qwen2.5-0.6b/ggml-model.gguf --port 8001 --host 0.0.0.0
```

### 3. Expose Ports to Host

To access the server from outside the container, add `-p` flag:

```bash
# vLLM on host port 8000
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "vllm serve /workspace/models/hf/qwen2.5-0.6b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1"
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
  -d '{"model": "/workspace/models/hf/qwen2.5-0.6b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}' | python3 -m json.tool
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
  bash -c "vllm serve /workspace/models/hf/qwen2.5-0.6b --port 8000 --host 0.0.0.0 --max-model-len 2048 --gpu-memory-utilization 0.1"

# Terminal 2: SGLang (10% GPU memory)
docker run --rm --gpus all --ipc=host -p 8003:8003 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "/opt/venv-sglang/bin/python -m sglang.launch_server --model /workspace/models/hf/qwen2.5-0.6b --port 8003 --host 0.0.0.0 --mem-fraction-static 0.1"

# Terminal 3: llama.cpp (minimal KV cache, GGUF model required)
docker run --rm --gpus all --ipc=host -p 8001:8001 \
  -v $(pwd)/models:/workspace/models \
  harmony-bench:cu129 \
  bash -c "llama-server -m /workspace/models/gguf/qwen2.5-0.6b/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8001 --host 0.0.0.0 -np 1 -c 1024 -ctk q8_0 -ctv turbo4 -fa on"
```

**Note:** Only run one inference engine at a time for benchmarking. Running multiple engines simultaneously is for testing connectivity only.

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
VLLM_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.6b ./scripts/benchmark/bench-vllm.sh -u http://localhost:8000

# SGLang benchmark
SGLANG_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.6b SGLANG_PYTHON=/opt/venv-sglang/bin/python ./scripts/benchmark/bench-sglang.sh -u http://localhost:8003

# llama.cpp benchmark
./scripts/benchmark/bench-llamacpp.sh -u http://localhost:8001/v1
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
  bash -c "cd /workspace && ./scripts/benchmark/bench-vllm.sh -u http://localhost:8000"
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
VLLM_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.6b
SGLANG_BENCH_MODEL=/workspace/models/hf/qwen2.5-0.6b

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
