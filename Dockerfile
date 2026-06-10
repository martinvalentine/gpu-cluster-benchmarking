# ============================================================
# LLM Serving Base Image — GPU Cluster Benchmarking
# ============================================================
#
# Architecture:
#   /opt/venvs/vllm                     — vLLM virtualenv
#   /opt/venvs/sglang                   — SGLang virtualenv
#   /usr/local/bin/llama-server         — llama-cpp-turboquant server (port 8001)
#
# NOT included (git clone at runtime):
#   benchmark/, scripts/, configs/, results/
#
# Prerequisites:
#   git submodule update --init --recursive
#
# Build (default — A40 / RTX 3090):
#   docker build -t llm-serving-base:latest .
#
# Build (tune parallelism for your machine):
#   docker build --build-arg MAX_JOBS=4 --build-arg NVCC_THREADS=4 --build-arg CMAKE_JOBS=8 -t llm-serving-base:latest .
#
# Build Args Reference:
#   CUDA_ARCH     — GPU compute capability (default: 8.6 = A40)
#   MAX_JOBS      — Ninja parallel jobs for vLLM/SGLang C++ extensions (default: 4)
#   NVCC_THREADS  — Threads per nvcc invocation (default: 4)
#   CMAKE_JOBS    — Parallel jobs for llama.cpp cmake builds (default: 8)
#   PYTHON_VERSION — Python version (default: 3.12)
#
# Build (other GPUs):
#   docker build --build-arg CUDA_ARCH="8.0" -t myimage:latest .    # A100
#   docker build --build-arg CUDA_ARCH="8.9" -t myimage:latest .    # RTX 4090 / Ada
#   docker build --build-arg CUDA_ARCH="9.0" -t myimage:latest .    # H100 / H200
#
# Build (multi-arch — space-separated, dot format):
#   docker build --build-arg CUDA_ARCH="8.0 8.6 8.9 9.0" -t myimage:latest .
#
# Build (incremental — see guide):
#   docker build --target cpp-build-base -t llm-base:cpp .
#   docker build --target turboquant-build -t llm-base:turbo .
#   docker build -t llm-serving-base:latest .
#
# Run:
#   docker run --gpus all -it llm-serving-base:latest
#
# Usage (no venv activation):
#   /opt/venvs/vllm/bin/vllm serve <model>
#   /opt/venvs/sglang/bin/python -m sglang.launch_server --model <model>
#   llama-server -m <model.gguf>
#   llama-server-turbo -m <model.gguf>
#
# GPU Architecture Reference:
#   7.5  = Turing   (T4, RTX 2080)
#   8.0  = Ampere   (A100)
#   8.6  = Ampere   (A40, RTX 3090)
#   8.9  = Ada      (RTX 4090, L40)
#   9.0  = Hopper   (H100, H200)
#   10.0 = Blackwell (B200)
# ============================================================

ARG CUDA_VERSION=12.6.1
ARG UBUNTU_VERSION=22.04

# ============================================================
# STAGE 1 — C++ Build Base (for llama-cpp-turboquant)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cpp-build-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# CUDA stubs for linking without host driver (libcuda.so lives in driver, not toolkit)
ENV LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}
RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
           /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

# ============================================================
# STAGE 2 — Build llama-cpp-turboquant
# ============================================================
FROM cpp-build-base AS turboquant-build

ARG CUDA_ARCH=8.6
ARG CMAKE_JOBS=8

COPY third_party/llama-cpp-turboquant /src

# Convert PyTorch format "8.0 8.6 8.9" → CMake format "80;86;89"
# BUILD_SHARED_LIBS left at default (ON) — matches scripts/build/build-llamacpp-turbo.sh
# LLAMA_BUILD_EXAMPLES/TESTS=OFF — only need server + bench, avoids linker errors
RUN CUDA_ARCH_CMAKE="$(echo "${CUDA_ARCH}" | sed 's/\.//g; s/ /;/g')" \
    && cmake -B /build -S /src \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_CMAKE}" \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /build --config Release -j"${CMAKE_JOBS}"

# ============================================================
# STAGE 3 — Runtime Image
# ============================================================
# MUST use devel (not runtime/base): vLLM and SGLang JIT-compile
# CUDA kernels at runtime (FlashInfer, DeepGEMM, EP kernels).
# They need nvcc which requires the full CUDA toolkit.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION=3.12
ARG CUDA_ARCH=8.6
ARG MAX_JOBS=4
ARG NVCC_THREADS=4

ENV DEBIAN_FRONTEND=noninteractive
# TORCH_CUDA_ARCH_LIST uses PyTorch format: "8.0 8.6 8.9" (space-separated, with dot)
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV CUDA_HOME=/usr/local/cuda
# setuptools-scm needs .git to detect version; Docker COPY strips it.
# Pretend a version so build doesn't fail.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

# ── System Dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ninja-build \
    curl \
    wget \
    software-properties-common \
    openssh-server \
    redis-server \
    libibverbs-dev \
    libnuma-dev \
    numactl \
    ca-certificates \
    gnupg \
    lsb-release \
    jq \
    htop \
    tmux \
    vim \
    procps \
    less \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.12 ───────────────────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────
ENV UV_INSTALL_DIR="/opt/uv/bin"
ENV UV_CACHE_DIR="/opt/uv/cache"
ENV PATH="${UV_INSTALL_DIR}:${PATH}"

RUN mkdir -p "${UV_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# ── Rust (needed for vLLM / SGLang setuptools-rust) ───────────
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

# ── llama-server (turboquant) + Shared Libraries ──────────────
COPY --from=turboquant-build /build/bin/llama-server /usr/local/bin/llama-server
COPY --from=turboquant-build /build/bin/llama-bench  /usr/local/bin/llama-bench
COPY --from=turboquant-build /build/bin/*.so*        /usr/local/lib/
RUN ldconfig

# ── vLLM venv ─────────────────────────────────────────────────
RUN uv venv --seed --python "${PYTHON_VERSION}" /opt/venvs/vllm

# Copy dependency specs first (cache layer — only invalidated when deps change)
COPY third_party/vllm/requirements/ /tmp/third_party/vllm/requirements/

RUN . /opt/venvs/vllm/bin/activate \
    && uv pip install --upgrade pip wheel \
    && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 \
    && uv pip install numpy \
    && uv pip install -r /tmp/third_party/vllm/requirements/build/cuda.txt \
    && uv cache clean

# Copy full source and build (invalidated on any vllm source change)
COPY third_party/vllm /tmp/third_party/vllm

RUN . /opt/venvs/vllm/bin/activate \
    && cd /tmp/third_party/vllm \
    && export MAX_JOBS=${MAX_JOBS} \
    && export NVCC_THREADS=${NVCC_THREADS} \
    && uv pip install --no-build-isolation . \
    && uv cache clean \
    && rm -rf /tmp/third_party/vllm

# ── SGLang venv ───────────────────────────────────────────────
RUN uv venv --seed --python "${PYTHON_VERSION}" /opt/venvs/sglang

# Copy dependency specs first (cache layer)
COPY third_party/sglang/python/pyproject.toml \
     /tmp/third_party/sglang/python/pyproject.toml

RUN . /opt/venvs/sglang/bin/activate \
    && uv pip install --upgrade pip wheel \
    && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && uv pip install numpy \
    && uv pip install \
        "setuptools>=61.0" \
        setuptools-scm \
        setuptools-rust \
    && uv cache clean

# Copy full source and build
COPY third_party/sglang /tmp/third_party/sglang

RUN . /opt/venvs/sglang/bin/activate \
    && cd /tmp/third_party/sglang/python \
    && export MAX_JOBS=${MAX_JOBS} \
    && export NVCC_THREADS=${NVCC_THREADS} \
    && uv pip install --no-build-isolation . \
    && uv cache clean \
    && rm -rf /tmp/third_party/sglang

# ── Cleanup ───────────────────────────────────────────────────
RUN uv cache clean \
    && rm -rf /root/.cache/pip /tmp/third_party

# ── Workspace ─────────────────────────────────────────────────
RUN mkdir -p /workspace/{models/hf,models/gguf,datasets,results,logs} \
    && mkdir -p /run/sshd \
    && mkdir -p /root/.ssh \
    && chmod 700 /root/.ssh

# ── Start Script ──────────────────────────────────────────────
COPY scripts/docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 22 6379 8000 8001 8003 30000

WORKDIR /workspace

CMD ["/app/start.sh"]
