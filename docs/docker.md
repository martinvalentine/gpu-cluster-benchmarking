# Docker

## Harmony Benchmark Image

A single Docker image bundling vLLM, SGLang, and llama-cpp-turboquant for unified benchmarking.

### What's Included

| Engine | Version | Source |
|--------|---------|--------|
| vLLM | 0.20.1 | Official vLLM image (inherited) |
| SGLang | 0.5.8 | PyPI |
| flashinfer-python | 0.6.1 | PyPI (FlashInfer wheel index) |
| sgl-kernel | 0.3.21 | PyPI |
| llama-cpp-turboquant | Latest | Built from submodule in Docker |

### Build

```bash
# Default (CUDA 12.9, A40/RTX 3090)
docker build -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# Override GPU architecture
docker build --build-arg CUDA_ARCH=8.9 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# Override versions
docker build --build-arg SGLANG_VERSION=0.5.9 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
```

### Run

```bash
# Interactive session with GPU and models
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -it harmony-bench:cu129

# Background mode
docker run -d --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  --name harmony-bench \
  harmony-bench:cu129
```

### Launch Engines

Inside the container:

```bash
# vLLM (HuggingFace models, port 8000)
vllm serve /workspace/models/hf/<model-name> --port 8000

# SGLang (HuggingFace models, port 8003)
python3 -m sglang.launch_server --model /workspace/models/hf/<model-name> --port 8003

# llama-server (GGUF models, port 8001)
llama-server -m /workspace/models/gguf/<model-file>.gguf --port 8001

# llama-bench (benchmark GGUF models)
llama-bench -m /workspace/models/gguf/<model-file>.gguf -n 128
```

### Run Benchmarks

```bash
# Full benchmark (all models, all backends)
./scripts/bench-models.sh

# Specific backend
./scripts/bench-models.sh -b vllm
./scripts/bench-models.sh -b sglang
./scripts/bench-models.sh -b llamacpp

# Specific model
./scripts/bench-models.sh -m qwen0.5b

# Preview without running
./scripts/bench-models.sh --dry-run
```

### Verify

```bash
# Check all engines
docker run --rm harmony-bench:cu129 python3 -c "import sglang; print(f'sglang={sglang.__version__}')"
docker run --rm harmony-bench:cu129 python3 -c "import flashinfer; print('flashinfer ok')"
docker run --rm harmony-bench:cu129 bash -c "command -v vllm"
docker run --rm harmony-bench:cu129 bash -c "command -v llama-server"
docker run --rm harmony-bench:cu129 bash -c "ldd /usr/local/bin/llama-server 2>&1 | grep -i 'not found' || echo 'ALL DEPS OK'"

# GPU visibility
docker run --rm --gpus all harmony-bench:cu129 python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Startup summary
timeout 10 docker run --rm harmony-bench:cu129 2>&1 || true
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | Restrict GPU visibility |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `PUBLIC_KEY` | (empty) | SSH public key injection |
| `BENCHMARK_REPO` | (empty) | Git repo to clone on startup |

### Image Size

~53 GB (includes vLLM, SGLang, flashinfer, llama-cpp-turboquant, CUDA runtime)

### Architecture

Multi-stage Docker build:
1. **Stage 1** (`turboquant-build`): Builds llama-cpp-turboquant from `third_party/llama-cpp-turboquant/` submodule using CUDA 12.9 devel image
2. **Stage 2** (`runtime`): Uses official vLLM image as base, installs SGLang + flashinfer via PyPI, copies llama binaries from Stage 1

Container behavior: Init-only toolbox — starts Redis/SSH, prints framework summary, sleeps forever. Users launch engines via `docker exec`.
