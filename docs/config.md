# Configuration Reference

## configs/models.yaml

Controls which models to download and where to store them.

```yaml
base_dir: /workspace/models    # Root directory for all models

models:
  - name: qwen7b-gguf          # Short name (used for --only filtering)
    repo_id: Qwen/Qwen2.5-7B-Instruct-GGUF   # HuggingFace repo
    local_dir: gguf/qwen2.5-7b                # Relative to base_dir
    format: gguf                # hf or gguf
    include: "*q4_k_m.gguf"    # Glob pattern (gguf only)
    enabled: false              # Set true to download
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for `--only` filtering |
| `repo_id` | Yes | HuggingFace repository ID |
| `local_dir` | Yes | Destination path (relative to `base_dir`) |
| `format` | No | `hf` (full repo) or `gguf` (specific files). Default: `hf` |
| `include` | No | Glob pattern for GGUF files (e.g. `*q4_k_m.gguf`) |
| `exclude` | No | Glob pattern to skip files |
| `enabled` | No | `true` to download, `false` to skip. Default: `true` |

## litellm_config.yaml

LiteLLM proxy configuration with model routing and cache.

**Sections:**

- `model_list` — LLM backends (vLLM, llama.cpp, SGLang) + embedding model
- `router_settings` — Load balancing strategy (least-busy)
- `litellm_settings` — Cache (redis), timeouts, Prometheus callbacks

**Environment variable substitution:** Uses `os.environ/VAR_NAME` syntax for values that should read from env vars.

## .env

Environment variables for services.

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | (empty) | Redis password |
| `EMBEDDING_API_BASE` | `http://localhost:8003/v1` | Embedding server URL |
| `EMBEDDING_MODEL` | `openai/embedding-model` | Embedding model name |
| `EMBEDDING_API_KEY` | `EMPTY` | Embedding API key |
| `OPENAI_API_BASE` | `http://localhost:8003/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `EMPTY` | API key (EMPTY for local) |
| `VLLM_PORT` | `8000` | vLLM server port |
| `VLLM_HOST` | `0.0.0.0` | vLLM bind host |
| `VLLM_TP` | `6` | vLLM tensor parallel size |
| `LLAMA_PORT` | `8001` | llama.cpp server port |
| `LLAMA_HOST` | `0.0.0.0` | llama.cpp bind host |
| `SGLANG_PORT` | `8002` | SGLang server port |
| `LITELLM_PORT` | `4000` | LiteLLM proxy port |

## Benchmark Variables

| Variable | Script | Default | Description |
|----------|--------|---------|-------------|
| `LLAMA_BENCH_URL` | bench-llamacpp | `http://localhost:8001/v1` | llama.cpp server URL |
| `LLAMA_RESULTS_DIR` | bench-llamacpp | `./results/llamacpp` | Output directory |
| `LLAMA_CONC_LEVELS` | bench-llamacpp | `1 4 8 16` | Concurrency levels to test |
| `LLAMA_CTX_SIZES` | bench-llamacpp | `1024 4096 16384` | Context sizes for long context test |
| `LLAMA_MODEL` | run-llamacpp | (required) | Model path for llama.cpp server |
| `VLLM_BENCH_URL` | bench-vllm | `http://localhost:8000` | vLLM server URL |
| `VLLM_BENCH_MODEL` | bench-vllm | | Model name |
| `VLLM_RESULTS_DIR` | bench-vllm | `./results/vllm` | Output directory |
| `SGLANG_BENCH_URL` | bench-sglang | `http://localhost:8002` | SGLang server URL |
| `SGLANG_BENCH_MODEL` | bench-sglang | | Model name |
| `SGLANG_RESULTS_DIR` | bench-sglang | `./results/sglang` | Output directory |
