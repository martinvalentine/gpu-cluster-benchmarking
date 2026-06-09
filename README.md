# GPU Cluster Benchmarking

LLM Self-Hosted Benchmark & Deployment — 6× NVIDIA A40 · vLLM · llama.cpp · SGLang

## Project Structure

```
gpu-cluster-benchmarking/
├── third_party/
│   └── llama.cpp/          # Git submodule (C++ source, built via CMake)
├── scripts/
│   └── build-llamacpp.sh   # CMake build script for llama.cpp
├── pyproject.toml           # uv dependency groups per framework
└── benchmark_plan.md
```

**Serving frameworks are separate:**
- **vLLM** / **SGLang** — Python packages, installed via `uv sync`
- **llama.cpp** — C++ submodule, compiled via CMake into `third_party/llama.cpp/build/bin/`

## Environment Setup with uv

Install [uv](https://docs.astral.sh/uv/) first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd gpu-cluster-benchmarking

# Or if already cloned:
git submodule update --init --depth 1
```

### 2. Sync by Serving Framework

Each serving framework has its own dependency group. Install only what you need:

```bash
# Common deps only (huggingface-hub, aiohttp, pandas, numpy, etc.)
uv sync

# vLLM
uv sync --group vllm

# SGLang
uv sync --group sglang

# llama.cpp Python scripts (GGUF conversion utilities)
uv sync --group llamacpp

# Multiple frameworks
uv sync --group vllm --group sglang

# Benchmark tools (locust, llama-benchy)
uv sync --group benchmark

# LiteLLM proxy + caching
uv sync --group litellm

# Monitoring (prometheus-client, psutil, gputil)
uv sync --group monitoring

# Everything
uv sync --group all
```

### 3. Build llama.cpp (C++ binary)

llama.cpp is a C++ project — build it separately with the provided script:

```bash
# Build with CUDA (default: A40 arch 86, all CPU cores)
./scripts/build-llamacpp.sh

# Custom CUDA architecture (e.g. A100 = 80, H100 = 90)
CUDA_ARCH=80 ./scripts/build-llamacpp.sh

# Binary location after build:
# third_party/llama.cpp/build/bin/llama-server
# third_party/llama.cpp/build/bin/llama-cli
```

### Run Serving Frameworks

```bash
# vLLM (Python — port 8000)
uv run --group vllm python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/hf/qwen2.5-32b-awq --tensor-parallel-size 6 --port 8000

# SGLang (Python — port 8002)
uv run --group sglang python -m sglang.launch_server \
  --model-path /workspace/models/hf/qwen2.5-32b-awq --tp 6 --port 8002

# llama.cpp (C++ binary — port 8001)
./third_party/llama.cpp/build/bin/llama-server \
  -m /workspace/models/gguf/qwen2.5-32b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8001

# LiteLLM proxy
uv run --group litellm litellm --config litellm_config.yaml --port 4000
```

## Dependency Groups Reference

| Group | Packages | Purpose |
|-------|----------|---------|
| `common` | huggingface-hub, aiohttp, pandas, numpy, tqdm, transformers, prometheus-client, psutil, gputil, redis | Base utilities & monitoring |
| `vllm` | vllm>=0.8.0 | PagedAttention + Continuous Batching + Prefix Caching |
| `sglang` | sglang[all]>=0.5.0, flashinfer-python | RadixAttention + Chunk Prefill + FlashInfer |
| `llamacpp` | third_party/llama.cpp (submodule, CMake build) | GGUF quantization with CUDA backend |
| `benchmark` | locust, llama-benchy, aiohttp, pandas, numpy, tqdm, transformers | Load testing & benchmark tools |
| `litellm` | litellm[proxy]>=1.40.0, redis | API gateway + semantic cache |
| `monitoring` | prometheus-client, psutil, gputil | GPU & system metrics collection |
| `all` | (all groups above) | Full install |

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url> && cd gpu-cluster-benchmarking

# Install Python deps
uv sync --group all

# Build llama.cpp
./scripts/build-llamacpp.sh

# Download models
uv run huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \
  --local-dir /workspace/models/hf/qwen2.5-32b-awq

# Start serving (each in a separate tmux session)
uv run --group vllm python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/hf/qwen2.5-32b-awq --tensor-parallel-size 6 --port 8000

uv run --group sglang python -m sglang.launch_server \
  --model-path /workspace/models/hf/qwen2.5-32b-awq --tp 6 --port 8002

./third_party/llama.cpp/build/bin/llama-server \
  -m /workspace/models/gguf/qwen2.5-32b-instruct-q4_k_m.gguf --port 8001

# Run benchmarks
uv run --group benchmark python benchmarks/benchmark_serving.py \
  --backend openai-chat --base-url http://localhost:8000 \
  --model /workspace/models/hf/qwen2.5-32b-awq \
  --dataset-name sharegpt --dataset-path /workspace/datasets/sharegpt.json \
  --num-prompts 100 --max-concurrency 8
```
