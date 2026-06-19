# Documentation

## Quick links by task

| I want to... | Read |
|---|---|
| Set up the project (uv, deps, Redis) | [setup.md](setup.md) |
| Build llama-cpp-turboquant from source | [build.md](build.md) |
| Download model weights from HuggingFace | [download-models.md](download-models.md) |
| Start servers (vLLM, SGLang, llama.cpp, embedding, proxy) | [run.md](run.md) |
| Understand the full benchmark workflow (Docker → bench → viz) | [benchmark-guide.md](benchmark-guide.md) |
| Run benchmarks (orchestrator, per-backend scripts, sweep modes) | [benchmark.md](benchmark.md) |
| Understand benchmark architecture, GPU allocation, metrics | [benchmark-architecture.md](benchmark-architecture.md) |
| Build & run the Docker image | [docker.md](docker.md) |
| Edit `configs/models.yaml` | [config.md](config.md) |
| Look up a specific YAML key | [params.md](params.md) |
| Look up a vLLM/SGLang/llama.cpp CLI flag | [serving-frameworks-params.md](serving-frameworks-params.md) |

## All documents

| Document | Description |
|----------|-------------|
| [setup.md](setup.md) | Install uv, clone repo, install deps, start Redis |
| [build.md](build.md) | Build llama-cpp-turboquant from source |
| [run.md](run.md) | Start servers — all params for vLLM, llama.cpp, SGLang, embedding, proxy |
| [download-models.md](download-models.md) | Download models from HuggingFace |
| [benchmark.md](benchmark.md) | Run benchmarks — phases, params, pipeline, results |
| [config.md](config.md) | Config files, env vars, and options reference |
| [params.md](params.md) | Full parameter reference for `configs/models.yaml` |
| [docker.md](docker.md) | Docker images — Harmony image build, run, verify |
| [benchmark-guide.md](benchmark-guide.md) | End-to-end guide — Docker hosting, llama-benchy sweeps, metrics, troubleshooting |
| [benchmark-architecture.md](benchmark-architecture.md) | Benchmark architecture, GPU allocation, metrics reference |
| [serving-frameworks-params.md](serving-frameworks-params.md) | Complete parameter reference for vLLM, SGLang, and llama.cpp CLI flags |
