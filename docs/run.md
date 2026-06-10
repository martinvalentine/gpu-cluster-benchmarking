# Run Servers

Each server runs in its own terminal/tmux session.

## Port Reference

| Service | Port | Script |
|---------|------|--------|
| vLLM | 8000 | `scripts/run/run-vllm.sh` |
| llama-cpp-turboquant | 8001 | `scripts/run/run-llamacpp.sh` |
| SGLang | 8002 | `scripts/run/run-sglang.sh` |
| Embedding server | 8003 | `scripts/run/run-embedding-server.sh` |
| LiteLLM proxy | 4000 | `scripts/run/run-proxy.sh` |
| Redis | 6379 | `redis-server` |

## llama-cpp-turboquant (port 8001)

```bash
uv run ./scripts/run/run-llamacpp.sh [MODEL_PATH]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p` | `LLAMA_PORT` | `8001` | Server port |
| `-H` | `LLAMA_HOST` | `0.0.0.0` | Bind host |
| `-n` | `LLAMA_N_PARALLEL` | `4` | Concurrent slots (CCU) |
| `-c` | `LLAMA_CTX_SIZE` | `4096` | Total context size |
| `-ng` | `LLAMA_N_GPU_LAYERS` | `all` | GPU layers to offload |
| `-b` | `LLAMA_BATCH` | `2048` | Prefill batch size |
| `-ub` | `LLAMA_UBATCH` | `512` | Micro-batch size |
| `-t` | `LLAMA_THREADS` | `nproc` | CPU threads |
| `-fa` | `LLAMA_FLASH_ATTN` | `on` | Flash Attention (on/off/auto) |
| `-ctk` | `LLAMA_CACHE_KEY` | `q8_0` | KV cache key type |
| `-ctv` | `LLAMA_CACHE_VAL` | `turbo4` | KV cache value type (turbo4/turbo3) |

**Examples:**

```bash
# Default (4 slots, 4K context, turbo4)
uv run ./scripts/run/run-llamacpp.sh

# 8 slots, 32K context
uv run ./scripts/run/run-llamacpp.sh -n 8 -c 32768

# Single slot, 64K context, turbo3
uv run ./scripts/run/run-llamacpp.sh -n 1 -c 65536 -ctv turbo3

# Custom model path
uv run ./scripts/run/run-llamacpp.sh /path/to/model.gguf
```

**Verify:**

```bash
curl http://localhost:8001/v1/models
```

## vLLM (port 8000)

```bash
uv run ./scripts/run/run-vllm.sh [MODEL_PATH]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p` | `VLLM_PORT` | `8000` | Server port |
| `-H` | `VLLM_HOST` | `0.0.0.0` | Bind host |
| `-tp` | `VLLM_TP` | `6` | Tensor parallel size |
| `-gmu` | `VLLM_GPU_MEM_UTIL` | `0.87` | GPU memory utilization |
| `-mml` | `VLLM_MAX_MODEL_LEN` | `4096` | Max context length |
| `-mns` | `VLLM_MAX_NUM_SEQS` | `256` | Max concurrent sequences |
| `-q` | `VLLM_QUANT` | `awq` | Quantization (awq/gptq/none) |
| `-pc` | | | Enable prefix caching |
| `-no-pc` | | | Disable prefix caching |
| `-cp` | | | Enable chunked prefill |
| `-no-cp` | | | Disable chunked prefill |
| `-mbt` | `VLLM_MAX_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `-sw` | `VLLM_SWAP_SPACE` | `4` | Swap space (GB) |
| `-mp` | `VLLM_METRICS_PORT` | `9090` | Prometheus metrics port |

**Example:**

```bash
uv run ./scripts/run/run-vllm.sh -tp 6 -gmu 0.90 -q awq /workspace/models/hf/qwen2.5-32b-awq
```

## SGLang (port 8002)

```bash
uv run ./scripts/run/run-sglang.sh [MODEL_PATH]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p` | `SGLANG_PORT` | `8002` | Server port |
| `-H` | `SGLANG_HOST` | `0.0.0.0` | Bind host |
| `-tp` | `SGLANG_TP` | `6` | Tensor parallel size |
| `-mfs` | `SGLANG_MEM_FRAC` | `0.87` | Static memory fraction |
| `-mtt` | `SGLANG_MAX_TOTAL_TOKENS` | `1048576` | Max total tokens |
| `-cps` | `SGLANG_CHUNKED_PREFILL_SIZE` | `8192` | Chunked prefill size |
| `-attn` | `SGLANG_ATTN_BACKEND` | `flashinfer` | Attention backend |
| `-q` | `SGLANG_QUANT` | `awq` | Quantization |
| `-mrr` | `SGLANG_MAX_RUNNING` | `256` | Max running requests |
| | | | Enable torch.compile |
| `--no-torch-compile` | | | Disable torch.compile |
| `--disable-radix-cache` | | | Disable RadixAttention cache |

**Example:**

```bash
uv run ./scripts/run/run-sglang.sh -tp 6 -q awq /workspace/models/hf/qwen2.5-32b-awq
```

## Embedding Server (port 8003)

```bash
uv run ./scripts/run/run-embedding-server.sh [MODEL_PATH]
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p` | `LLAMA_EMBED_PORT` | `8003` | Server port |
| `-H` | `LLAMA_EMBED_HOST` | `0.0.0.0` | Bind host |
| `-ng` | `LLAMA_EMBED_GPU_LAYERS` | `999` | GPU layers |
| `-c` | `LLAMA_EMBED_CONTEXT` | `32768` | Context size |
| `-np` | `LLAMA_EMBED_POOLING` | `last` | Pooling (last/mean/cls) |
| `-t` | `LLAMA_EMBED_THREADS` | `nproc` | CPU threads |
| `-ccu` | `LLAMA_EMBED_CONCURRENT` | `20` | Concurrent slots |

**Example:**

```bash
uv run ./scripts/run/run-embedding-server.sh /path/to/embedding.gguf -p 8003 -np last
```

## LiteLLM Proxy (port 4000)

```bash
OPENAI_API_BASE=http://localhost:8003/v1 OPENAI_API_KEY=EMPTY \
    uv run ./scripts/run/run-proxy.sh
```

**Config:** `litellm_config.yaml`

Requires Redis and at least one backend running (vLLM/llama.cpp/SGLang).
