# Build

llama-cpp-turboquant is a C++ project that must be compiled from source before serving.

## Build llama-cpp-turboquant

```bash
./scripts/build/build-llamacpp-turbo.sh
```

**Output:**

```
third_party/llama-cpp-turboquant/build/bin/llama-server
third_party/llama-cpp-turboquant/build/bin/llama-bench
```

**Build parameters** (set via environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_ARCHITECTURES` | `86` | GPU compute capability (86 = A40/3090, 89 = 4090, 90 = H100) |
| `BUILD_TYPE` | `Release` | CMake build type |
| `NPROC` | `$(nproc)` | Parallel compile jobs |

**Examples:**

```bash
# Default (A40/RTX 3090)
./scripts/build/build-llamacpp-turbo.sh

# For RTX 4090
CUDA_ARCHITECTURES=89 ./scripts/build/build-llamacpp-turbo.sh

# Limit parallel jobs (reduce RAM usage)
NPROC=4 ./scripts/build/build-llamacpp-turbo.sh
```

**Verify build:**

```bash
ls -la third_party/llama-cpp-turboquant/build/bin/llama-server
./third_party/llama-cpp-turboquant/build/bin/llama-server --help
```

---

## Docker Build (Harmony Image)

The Harmony image builds llama-cpp-turboquant inside Docker (no local build needed) and bundles vLLM + SGLang.

```bash
docker build -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
```

**Build parameters** (override via `--build-arg`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_VERSION` | `v0.23.0` | vLLM version (from Docker Hub) |
| `CUDA_VERSION` | `129` | CUDA version |
| `CUDA_ARCH` | `8.6` | GPU compute capability |
| `CMAKE_JOBS` | `8` | Parallel compile jobs for llama.cpp |
| `SGLANG_VERSION` | `0.5.8` | SGLang version |
| `SGL_KERNEL_VERSION` | `0.3.21` | sgl-kernel version |

**Examples:**

```bash
# Default (A40/RTX 3090)
docker build -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# For RTX 4090
docker build --build-arg CUDA_ARCH=8.9 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# For H100
docker build --build-arg CUDA_ARCH=9.0 -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .
```

See [docker.md](docker.md) for full usage and verification.
