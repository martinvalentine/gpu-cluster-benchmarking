# Benchmarks

Requires at least one server running (see [run.md](run.md)).

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

**Examples:**

```bash
# Run all tests
uv run ./scripts/benchmark/bench-llamacpp.sh

# Custom concurrency levels
uv run ./scripts/benchmark/bench-llamacpp.sh -c "1 4 8"

# Remote server
uv run ./scripts/benchmark/bench-llamacpp.sh -u http://gpu-pod:8001/v1
```

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
uv run ./scripts/benchmark/bench-vllm.sh -p p3 -m Qwen3.6-35B-A3B-UDT-Q5_K_XL_MTP
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
