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
