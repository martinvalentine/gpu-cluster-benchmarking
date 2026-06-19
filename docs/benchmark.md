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
  sharegpt: datasets/sharegpt.json
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
| `-b, --backend BACKEND` | all | Backend filter: vllm, llamacpp (can repeat) |
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

# Only vLLM backends
./scripts/bench-models.sh -b vllm

# Only llama.cpp backends
./scripts/bench-models.sh -b llamacpp

# Combine filters
./scripts/bench-models.sh -p p0 -b llamacpp
./scripts/bench-models.sh -b llamacpp -m qwen0.5b-gguf

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

Use `-b vllm` or `-b llamacpp` to run only one backend at a time.

**Auto-skip:** Models with `vllm_tp > gpu_count` are automatically skipped.

## Start All Servers

```bash
./scripts/start-all-tmux.sh    # Start all servers in tmux
./scripts/stop-all.sh          # Stop all servers
```

## llama-cpp-turboquant Benchmark

Direct HTTP benchmark — tests single user latency, concurrent throughput, and long context.

```bash
uv run ./scripts/benchmark/llamacpp_bench.sh [OPTIONS]
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
uv run ./scripts/benchmark/vllm_bench.sh --full [OPTIONS]
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
uv run ./scripts/benchmark/vllm_bench.sh --full -p p1
uv run ./scripts/benchmark/vllm_bench.sh --full -p p3 -m /workspace/models/hf/qwen2.5-32b-awq
```

## SGLang Benchmark

```bash
uv run ./scripts/benchmark/sglang_bench.sh --full [OPTIONS]
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

## Inspecting Server Params After a Run

When a benchmark result is unexpectedly slow, the question is almost always
"what was the server actually configured with?" — model size, KV cache type,
attention backend, TP size, GPU memory utilization. These are now captured
automatically.

**What gets written:**

| File | When | What |
|---|---|---|
| `results/<backend>/_active_params.json` | When `run-*.sh` starts a server | Full server config + hardware + system info for the currently-running server. Overwritten on the next server start. |
| `results/<backend>/<session>/params.json` | At sweep start | Snapshot copy of `_active_params.json`, frozen for that sweep. |
| `params` field in sweep JSON | After sweep completes | Embedded by `jq` into the sweep JSON so the params travel with the result. |

**Where to look in the report:**

- `report.py` renders a `## Key Parameters` section near the top with the
  most important fields (model, ctk/ctv, flash_attn, GPU, git commit)
- Per-row tables gain a `Params` column showing model, port, ctk/ctv, GPU, and
  commit for each row at a glance
- `parse_bench.py` adds flat `params.*` columns to the summary CSV (use
  `--hide-params` to suppress, `--only-params` for a params-only view)

**Quick check on the command line:**

```bash
# What model + cache type was the last server started with?
jq -r '"\(.server.model) ctk=\(.server.cache_key) ctv=\(.server.cache_val) fa=\(.server.flash_attn)"' \
  results/vllm/_active_params.json

# What GPU + CUDA?
jq -r '"\(.hardware.gpu_count)x \(.hardware.gpu_name) (cuda \(.hardware.cuda_version))"' \
  results/vllm/_active_params.json
```

**Manual override:**

If the server was started outside the `run-*.sh` scripts (e.g. directly via
`vllm serve` in tmux), the `params.json` won't be auto-written. To capture
params retroactively, run the `run-*.sh` script once with the same flags — it
will create `_active_params.json` without affecting the already-running server.
