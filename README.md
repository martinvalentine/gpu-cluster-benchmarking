# GPU Cluster Benchmarking

LLM Self-Hosted Benchmark & Deployment — vLLM · llama-cpp-turboquant · SGLang

## Port Reference

| Service | Port | Protocol |
|---------|------|----------|
| vLLM API | 8000 | HTTP |
| llama-cpp-turboquant API | 8001 | HTTP |
| SGLang API | 8002 | HTTP |
| Embedding Server | 8003 | HTTP |
| LiteLLM Proxy | 4000 | HTTP |
| Redis | 6379 | TCP |

## Project Structure

```
gpu-cluster-benchmarking/
├── third_party/                     # Git submodules
│   ├── vllm/
│   ├── sglang/
│   └── llama-cpp-turboquant/        # llama.cpp with TurboQuant KV cache
├── scripts/
│   ├── run/                         # Start serving frameworks
│   │   ├── run-vllm.sh              # vLLM server (port 8000)
│   │   ├── run-sglang.sh            # SGLang server (port 8002)
│   │   ├── run-llamacpp.sh          # llama-cpp-turboquant (port 8001)
│   │   ├── run-embedding-server.sh  # Embedding server (port 8003)
│   │   └── run-proxy.sh             # LiteLLM proxy (port 4000)
│   ├── benchmark/                   # Benchmark runners
│   │   ├── vllm_bench.sh            # vLLM: llama-benchy sweeps + native
│   │   ├── sglang_bench.sh          # SGLang: llama-benchy sweeps + native
│   │   ├── llamacpp_bench.sh        # llama.cpp: single/concurrent/long-context
│   │   ├── find_best_np_llama.sh    # llama.cpp: -np sweep for throughput knee
│   │   ├── bench.sh                 # Unified dispatcher (routes -b flag)
│   │   ├── bench-litellm.sh         # LiteLLM async/locust
│   │   ├── bench-litellm-async.py
│   │   └── bench-litellm-locust.py
│   ├── build/                       # Build from source
│   │   ├── build-llamacpp-turbo.sh  # Build llama-cpp-turboquant
│   │   ├── build-vllm.sh
│   │   └── build-sglang.sh
│   ├── bench-models.sh              # Per-model benchmark orchestrator
│   ├── run-all-benchmarks.sh        # Master runner — all frameworks sequentially
│   ├── run-pipeline.sh              # Full pipeline: download → bench → report
│   ├── start-all-tmux.sh            # Start all servers in tmux
│   ├── stop-all.sh                  # Stop all servers
│   ├── prepare-dataset.sh           # Download & prepare benchmark dataset
│   ├── download-models.py           # Config-driven HuggingFace batch download
│   ├── gen-litellm-config.py        # Generate litellm_config.yaml
│   ├── parse_bench.py               # Aggregate JSON → CSV/TSV/markdown
│   ├── report.py                    # Generate formatted report (terminal + CSV + markdown)
│   ├── visualize_cross_sweep.py     # Cross-sweep PNG charts
│   ├── monitor-gpu.sh               # Real-time GPU monitoring
│   ├── litellm-semantic-cache.py    # Standalone semantic cache demo
│   ├── test-all.sh                  # Test all LLM endpoints
│   ├── init-bare-machine.sh         # Bare metal machine setup
│   ├── docker/
│   │   ├── start.sh                 # Container entrypoint
│   │   └── docker-bench.sh          # Host-side benchmark wrapper
│   └── test/
│       ├── test-llm.sh              # Test individual LLM endpoints
│       └── test-proxy.sh            # Test LiteLLM proxy
├── configs/
│   └── models.yaml                  # Model config + cluster settings (single source of truth)
├── docs/                            # Documentation
│   ├── setup.md                     # Environment setup
│   ├── build.md                     # Build llama-cpp-turboquant
│   ├── run.md                       # Start servers (all params)
│   ├── download-models.md           # Download models
│   ├── benchmark.md                 # Run benchmarks (phases, params, pipeline)
│   ├── config.md                    # Config files, env vars, options
│   ├── params.md                    # Full parameter reference for models.yaml
│   ├── docker.md                    # Docker images — Harmony image build/run/verify
│   ├── benchmark-guide.md           # End-to-end benchmark workflow
│   ├── benchmark-architecture.md     # Architecture diagrams + GPU allocation
│   └── serving-frameworks-params.md # Full vLLM/SGLang/llama.cpp parameter reference
├── pyproject.toml                   # uv dependency groups
├── litellm_config.yaml              # LiteLLM proxy config (auto-generated)
├── docker/
│   └── Dockerfile.vllm-sglang-llama   # Harmony image (vLLM + SGLang + llama-cpp-turboquant)
└── dev_benchmark_plan.md            # Benchmark strategy (dev notes)
```

## Quick Start

```bash
# 1. Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules <repo-url> && cd gpu-cluster-benchmarking
uv sync --group common --group benchmark --group litellm --group monitoring

# 2. Build llama-cpp-turboquant
./scripts/build/build-llamacpp-turbo.sh

# 3. Start Redis
redis-server --daemonize yes

# 4. Download models
uv run python scripts/download-models.py

# 5. Prepare benchmark dataset
./scripts/prepare-dataset.sh                    # → datasets/sharegpt.json

# 6. Start all servers in tmux (reads configs/models.yaml)
./scripts/start-all-tmux.sh

# 7. Run all benchmarks (per-model loop, reads configs/models.yaml)
./scripts/bench-models.sh

# 8. Or run specific phase/model/backend
./scripts/bench-models.sh -p p0            # P0 models only
./scripts/bench-models.sh -b vllm          # vLLM backends only
./scripts/bench-models.sh -b llamacpp      # llama.cpp backends only
./scripts/bench-models.sh -m qwen32b-awq   # Specific model
./scripts/bench-models.sh --dry-run        # Preview actions
```


## Docker (Recommended for Benchmarking)

The Harmony image bundles all three engines (vLLM + SGLang + llama-cpp-turboquant) in a single container.

```bash
# Build
docker build -f docker/Dockerfile.vllm-sglang-llama -t harmony-bench:cu129 .

# Run with GPU access and models mounted
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -it harmony-bench:cu129

# Inside container — launch any engine:
vllm serve /workspace/models/hf/<model> --port 8000
python3 -m sglang.launch_server --model /workspace/models/hf/<model> --port 8003
llama-server -m /workspace/models/gguf/<model>.gguf --port 8001

# Run benchmarks
./scripts/bench-models.sh
```

See [docs/docker.md](docs/docker.md) for full Docker documentation.
## Configuration

Everything is configured in **`configs/models.yaml`** — this is the single source of truth.

### How configuration flows

```
configs/models.yaml
  ├── start-all-tmux.sh  → reads YAML → starts servers with correct paths/settings
  ├── bench-models.sh    → reads YAML → loops models, manages server lifecycle
  ├── download-models.py → reads YAML → downloads model weights
  └── gen-litellm-config.py → reads YAML → generates litellm_config.yaml
```

**Orchestrator scripts** (`bench-models.sh`, `start-all-tmux.sh`) read `configs/models.yaml` directly — no env vars needed.

**Individual scripts** (`run-vllm.sh`, `vllm_bench.sh`, etc.) load `.env` from project root, then accept env vars:

```bash
# Copy template and customize
cp .env_example .env
vim .env

# Run script — .env is loaded automatically
./scripts/run/run-vllm.sh
```

Or set env vars inline:

```bash
VLLM_MODEL=/workspace/models/hf/qwen2.5-32b-awq VLLM_QUANT=awq ./scripts/run/run-vllm.sh
```

### Cluster Configuration

Edit `configs/models.yaml` to match your pod's GPU count. Defaults below match the shipped config for 5× NVIDIA A40 48GB:

```yaml
cluster:
  gpu_count: 5              # 0 = auto-detect from nvidia-smi

  vllm:
    tp: 5                   # Tensor parallel size
    gpu_mem_util: "0.85"    # GPU memory utilization
    max_model_len: 16384    # Max context length
    max_num_seqs: 1024
    max_batched_tokens: 65536

  llamacpp:
    n_parallel: 48          # Concurrent slots
    ctx_size: 16384         # Context window
    batch: 8192
    ubatch: 2048
    cache_val: turbo4

  sglang:
    mem_fraction: "0.85"
    max_model_len: 4096

# Benchmark dataset path
dataset:
  sharegpt: datasets/sharegpt.json
```

Per-model overrides take precedence:
```yaml
- name: qwen32b-awq
  vllm_tp: 5               # Match your GPU count for large models
  vllm_gpu_mem: "0.82"     # Override cluster default
  vllm_max_seqs: 192
```

Models requiring more GPUs than `cluster.gpu_count` are auto-skipped by `bench-models.sh`.

## Download Models

```bash
uv run python scripts/download-models.py
uv run python scripts/download-models.py --only qwen32b-awq
uv run python scripts/download-models.py --phase p0 p1
uv run python scripts/download-models.py --dry-run
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/setup.md](docs/setup.md) | Install uv, clone, deps, Redis |
| [docs/build.md](docs/build.md) | Build llama-cpp-turboquant from source |
| [docs/run.md](docs/run.md) | Start servers — all params for each framework |
| [docs/download-models.md](docs/download-models.md) | Download models from HuggingFace |
| [docs/benchmark.md](docs/benchmark.md) | Run benchmarks — params, pipeline, results |
| [docs/config.md](docs/config.md) | Config files, env vars, options reference |
| [docs/params.md](docs/params.md) | Full parameter reference for `configs/models.yaml` |
| [docs/docker.md](docs/docker.md) | Docker images — Harmony image build, run, verify |
| [docs/benchmark-guide.md](docs/benchmark-guide.md) | Complete benchmark guide — hosting, sweeps, parsing, visualization |
| [docs/benchmark-architecture.md](docs/benchmark-architecture.md) | Architecture diagrams, GPU allocation, metrics reference |
| [docs/serving-frameworks-params.md](docs/serving-frameworks-params.md) | Parameter reference for vLLM, SGLang, and llama.cpp |

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/bench-models.sh` | Per-model benchmark orchestrator (loops over all models) |
| `scripts/prepare-dataset.sh` | Download & prepare benchmark dataset (Vietnamese/English) |
| `scripts/start-all-tmux.sh` | Start all servers in tmux (reads cluster config) |
| `scripts/stop-all.sh` | Stop all server processes |
| `scripts/run/run-llamacpp.sh` | Start llama-cpp-turboquant (TurboQuant KV cache) |
| `scripts/run/run-vllm.sh` | Start vLLM (prefix caching + chunked prefill) |
| `scripts/run/run-sglang.sh` | Start SGLang (RadixAttention + torch.compile) |
| `scripts/run/run-embedding-server.sh` | Start embedding server for semantic cache |
| `scripts/run/run-proxy.sh` | Start LiteLLM proxy with Redis cache |
| `scripts/benchmark/llamacpp_bench.sh` | llama.cpp benchmark (single/concurrent/long context) |
| `scripts/benchmark/vllm_bench.sh` | vLLM benchmark (vllm bench serve) |
| `scripts/benchmark/sglang_bench.sh` | SGLang benchmark (sglang.bench_serving) |
| `scripts/benchmark/bench-litellm.sh` | LiteLLM async/locust load test |
| `scripts/run-all-benchmarks.sh` | Master runner — all frameworks sequentially |
| `scripts/run-pipeline.sh` | Full pipeline: download → benchmark → parse |
| `scripts/parse_bench.py` | Aggregate JSON → CSV summary |
| `scripts/report.py` | Generate formatted report (terminal + CSV + markdown) |
| `scripts/monitor-gpu.sh` | Real-time GPU monitoring with CSV logging |
| `scripts/download-models.py` | Config-driven HuggingFace batch download |
| `scripts/gen-litellm-config.py` | Generate litellm_config.yaml from models.yaml |

## Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `common` | huggingface-hub, pyyaml, aiohttp, pandas, numpy, tqdm, transformers, prometheus-client, psutil, gputil, redis, requests, datasets, pyarrow | Base utilities, config loading, monitoring |
| `vllm` | vllm (compiled from source) | PagedAttention + Continuous Batching |
| `sglang` | sglang (compiled from source) | RadixAttention + Chunk Prefill + FlashInfer |
| `benchmark` | locust, llama-benchy, aiohttp, pandas, numpy, tqdm, transformers, matplotlib | Load testing + sweep sweeps + chart generation |
| `litellm` | litellm[proxy]>=1.40.0, redis, redisvl | API gateway + semantic cache |
| `monitoring` | prometheus-client, psutil, gputil | GPU & system metrics |

**Notes:**
- `vllm` and `sglang` groups conflict — cannot install both in the same environment. The Harmony Docker image installs each in its own venv.
- `benchmark` group requires `matplotlib` for cross-sweep chart generation (`scripts/visualize_cross_sweep.py`).
- `common` group is always installed.
