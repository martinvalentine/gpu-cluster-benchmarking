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
│   │   ├── bench-vllm.sh            # vLLM: vllm bench serve
│   │   ├── bench-sglang.sh          # SGLang: sglang.bench_serving
│   │   ├── bench-llamacpp.sh        # llama.cpp: direct HTTP tests
│   │   ├── bench-litellm.sh         # LiteLLM async/locust
│   │   ├── bench-litellm-async.py
│   │   └── bench-litellm-locust.py
│   ├── build/                       # Build from source
│   │   ├── build-llamacpp-turbo.sh  # Build llama-cpp-turboquant
│   │   ├── build-vllm.sh
│   │   └── build-sglang.sh
│   ├── download-models.py           # Batch download from HuggingFace
│   ├── run-all-benchmarks.sh        # Master benchmark runner
│   ├── run-pipeline.sh              # Full pipeline: download → bench → report
│   ├── report.py                    # Generate formatted report (CSV + markdown)
│   ├── parse-results.py             # Aggregate results → CSV
│   ├── monitor-gpu.sh               # Real-time GPU monitoring
│   ├── init-bare-machine.sh         # Bare metal setup
│   ├── docker/
│   │   └── start.sh                 # Container entrypoint (RunPod)
│   └── test/
│       ├── test-llm.sh              # Test LLM endpoints
│       └── test-proxy.sh            # Test LiteLLM proxy
├── configs/
│   └── models.yaml                  # Model download config
├── docs/                            # Documentation
│   ├── setup.md                     # Environment setup
│   ├── build.md                     # Build llama-cpp-turboquant
│   ├── run.md                       # Start servers (all params)
│   ├── download-models.md           # Download models
│   ├── benchmark.md                 # Run benchmarks (all params)
│   └── config.md                    # Config files reference
├── pyproject.toml                   # uv dependency groups
├── litellm_config.yaml              # LiteLLM proxy config
├── .env                             # Environment variables
├── Dockerfile.serving               # Docker image (vLLM + llama-cpp-turboquant)
├── run_pod_cloud.md                 # RunPod deployment guide
└── benchmark_plan.md                # Benchmark strategy
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

# 4. Start servers (each in separate tmux)
uv run ./scripts/run/run-llamacpp.sh                         # port 8001
uv run ./scripts/run/run-vllm.sh                              # port 8000
uv run ./scripts/run/run-sglang.sh                            # port 8002
uv run ./scripts/run/run-embedding-server.sh                  # port 8003
OPENAI_API_BASE=http://localhost:8003/v1 OPENAI_API_KEY=EMPTY \
    uv run ./scripts/run/run-proxy.sh                         # port 4000

# 5. Run benchmarks
uv run ./scripts/benchmark/bench-llamacpp.sh                  # llama.cpp
uv run ./scripts/benchmark/bench-vllm.sh -p p1                # vLLM
uv run ./scripts/benchmark/bench-sglang.sh -p p1              # SGLang
uv run ./scripts/run-all-benchmarks.sh                        # all frameworks

# 6. Full pipeline (download → bench → parse)
uv run ./scripts/run-pipeline.sh --only-download              # download models
uv run ./scripts/run-pipeline.sh --skip-download              # benchmark only
```

## Download Models

```bash
# Edit configs/models.yaml to enable models, then:
uv run python scripts/download-models.py
uv run python scripts/download-models.py --dir /path/to/models
uv run python scripts/download-models.py --only qwen7b-gguf
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

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/run/run-llamacpp.sh` | Start llama-cpp-turboquant (TurboQuant KV cache) |
| `scripts/run/run-vllm.sh` | Start vLLM (prefix caching + chunked prefill) |
| `scripts/run/run-sglang.sh` | Start SGLang (RadixAttention + torch.compile) |
| `scripts/run/run-embedding-server.sh` | Start embedding server for semantic cache |
| `scripts/run/run-proxy.sh` | Start LiteLLM proxy with Redis cache |
| `scripts/benchmark/bench-llamacpp.sh` | llama.cpp benchmark (single/concurrent/long context) |
| `scripts/benchmark/bench-vllm.sh` | vLLM benchmark (vllm bench serve) |
| `scripts/benchmark/bench-sglang.sh` | SGLang benchmark (sglang.bench_serving) |
| `scripts/benchmark/bench-litellm.sh` | LiteLLM async/locust load test |
| `scripts/run-all-benchmarks.sh` | Master runner — all frameworks sequentially |
| `scripts/run-pipeline.sh` | Full pipeline: download → benchmark → parse |
| `scripts/parse-results.py` | Aggregate JSON → CSV summary |
| `scripts/report.py` | Generate formatted report (terminal + CSV + markdown) |
| `scripts/monitor-gpu.sh` | Real-time GPU monitoring with CSV logging |
| `scripts/download-models.py` | Config-driven HuggingFace batch download |

## Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `common` | huggingface-hub, aiohttp, pandas, numpy, tqdm, transformers, prometheus-client, psutil, gputil, redis | Base utilities & monitoring |
| `vllm` | vllm (from source) | PagedAttention + Continuous Batching |
| `sglang` | sglang (from source) | RadixAttention + Chunk Prefill + FlashInfer |
| `benchmark` | locust, aiohttp, pandas, numpy, tqdm, transformers | Load testing tools |
| `litellm` | litellm[proxy]>=1.40.0, redis, redisvl | API gateway + semantic cache |
| `monitoring` | prometheus-client, psutil, gputil | GPU & system metrics |

**Note:** `vllm` and `sglang` groups conflict — cannot install both in the same environment.
