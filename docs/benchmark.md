# Benchmarks

Requires at least one server running (see [run.md](run.md)) and a dataset (see below).

## Prepare Dataset

```bash
# Download Vietnamese vi-alpaca → ShareGPT format (default)
./scripts/prepare-dataset.sh

# Custom directory
./scripts/prepare-dataset.sh -d /data/benchmarks

# Skip if exists
./scripts/prepare-dataset.sh --skip-existing
```

Dataset path is configured in `configs/models.yaml`:
```yaml
dataset:
  sharegpt: /workspace/datasets/sharegpt.json
```

## Per-Model Benchmark Loop (Recommended)

Runs end-to-end benchmarks for all enabled models, starting/stopping servers between each. Reads config from `configs/models.yaml` — **no env vars needed**.

```bash
./scripts/bench-models.sh [OPTIONS]
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output DIR` | `./results` | Results root directory |
| `-p, --phase PHASE` | `all` | Phase: p0, p1, p2, p3, all |
| `-m, --model NAME` | all enabled | Specific model (can repeat) |
| `--skip-health-check` | | Skip pre-flight health checks |
| `-y, --yes` | | Auto-accept prompts (CI) |
| `--dry-run` | | Preview without executing |

**Examples:**

```bash
# All enabled models
./scripts/bench-models.sh

# P0 models only
./scripts/bench-models.sh -p p0

# Specific model
./scripts/bench-models.sh -m qwen32b-awq

# Preview actions
./scripts/bench-models.sh --dry-run

# Skip health checks
./scripts/bench-models.sh --skip-health-check
```

**What it does per model:**

1. Starts vLLM with HF model → runs vLLM benchmarks → stops server
2. Starts llama.cpp with GGUF model → runs llama.cpp benchmarks → stops server
3. Results saved to `results/run-N/{vllm,llamacpp}/`

**Auto-skip:** Models with `vllm_tp > gpu_count` are automatically skipped.

## Start All Servers

```bash
./scripts/start-all-tmux.sh    # Start all servers in tmux
./scripts/stop-all.sh          # Stop all servers
```

## llama-cpp-turboquant Benchmark

Direct HTTP benchmark — tests single user latency, concurrent throughput, and long context.

```bash
uv run ./scripts/benchmark/bench-llamacpp.sh [OPTIONS]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-u` | `LLAMA_BENCH_URL` | `http://localhost:8001/v1` | Server URL |
| `-o` | `LLAMA_RESULTS_DIR` | `./results/llamacpp` | Output directory |
| `-c` | `LLAMA_CONC_LEVELS` | `1 4 8 16` | Concurrency levels to test |
| `-x` | `LLAMA_CTX_SIZES` | `1024 4096 16384` | Context sizes for long context test |

**Tests performed:**
1. **Single user** — 4 prompts (short/medium/code/long), measures TTFT + decode t/s
2. **Concurrent users** — each level tested individually with health check between
3. **Long context** — prefill speed at different context sizes
4. **Server metrics** — /metrics snapshot (if available)

## vLLM Benchmark

```bash
uv run ./scripts/benchmark/bench-vllm.sh [OPTIONS]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-u` | `VLLM_BENCH_URL` | `http://localhost:8000` | Server URL |
| `-m` | `VLLM_BENCH_MODEL` | | Model name |
| `-o` | `VLLM_RESULTS_DIR` | `./results/vllm` | Output directory |
| `-p` | | `all` | Phase: p1, p2, p3, all |

**Examples:**

```bash
uv run ./scripts/benchmark/bench-vllm.sh -p p1
uv run ./scripts/benchmark/bench-vllm.sh -p p3 -m /workspace/models/hf/qwen2.5-32b-awq
```

## SGLang Benchmark

```bash
uv run ./scripts/benchmark/bench-sglang.sh [OPTIONS]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-u` | `SGLANG_BENCH_URL` | `http://localhost:8002` | Server URL |
| `-m` | `SGLANG_BENCH_MODEL` | | Model name |
| `-o` | `SGLANG_RESULTS_DIR` | `./results/sglang` | Output directory |
| `-p` | | `all` | Phase: p1, p2, p3, all |

## All Frameworks

```bash
uv run ./scripts/run-all-benchmarks.sh [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f` | `all` | Framework: vllm, sglang, llamacpp, litellm |
| `-p` | `all` | Phase: p1, p2, p3, all (vLLM/SGLang only) |
| `--skip-health-check` | | Skip pre-flight connectivity checks |

## Pipeline

```bash
uv run ./scripts/run-pipeline.sh [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-download` | | Skip model download |
| `--only-download` | | Only download models, no benchmark |
| `--phase` | `all` | Benchmark phase (vLLM/SGLang) |

## Generate Report

```bash
uv run python scripts/report.py --results-dir results/
```

Outputs:
- Terminal table (colored)
- `results/benchmark_summary.csv`
- `results/report.md`

## GPU Monitoring

```bash
uv run ./scripts/monitor-gpu.sh -i 2 -d 300
```
