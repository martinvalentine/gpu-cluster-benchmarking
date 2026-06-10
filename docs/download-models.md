# Download Models

## Config

Edit `configs/models.yaml` to enable/disable models and set destination:

```yaml
base_dir: /workspace/models    # Default destination

models:
  - name: qwen7b
    repo_id: Qwen/Qwen2.5-7B-Instruct
    local_dir: hf/qwen2.5-7b
    format: hf
    enabled: true    # ← Set true to download
```

All models are `enabled: false` by default (except embedding). Flip to `true` to include.

## Download all enabled models

```bash
uv run python scripts/download-models.py
```

## Download to a custom directory

```bash
uv run python scripts/download-models.py --dir /path/to/models
```

## Download specific models only

```bash
uv run python scripts/download-models.py --only qwen7b qwen32b-awq
```

## Download by phase

```bash
uv run python scripts/download-models.py --phase p0 p1    # P0 + P1 models
uv run python scripts/download-models.py --phase p3        # P3 only
uv run python scripts/download-models.py --phase embedding # Embedding only
```

## Preview without downloading

```bash
uv run python scripts/download-models.py --dry-run
```

## Skip already downloaded

```bash
uv run python scripts/download-models.py --skip-existing
```

## List available models

```bash
uv run python scripts/download-models.py --list
```

## Available models

| Name | Phase | Params | Format | Backend | VRAM (TP=1) |
|------|-------|--------|--------|---------|-------------|
| qwen3-embedding | Embedding | 0.6B | GGUF | llama.cpp | ~0.5 GB |
| qwen0.5b | P0 Ultra-Light | 0.5B | HF | vLLM | ~0.2 GB |
| qwen0.5b-gguf | P0 Ultra-Light | 0.5B | GGUF | llama.cpp | ~0.5 GB |
| llama3.1-8b | P1 Light | 8B | HF | vLLM | ~3 GB |
| qwen7b | P1 Light | 7B | HF | vLLM | ~3 GB |
| qwen7b-gguf | P1 Light | 7B | GGUF | llama.cpp | ~4 GB |
| qwen14b | P2 Medium | 14B | HF | vLLM | ~5 GB |
| qwen14b-gguf | P2 Medium | 14B | GGUF | llama.cpp | ~8 GB |
| qwen32b-awq | P3 Heavy | 32B | HF (AWQ) | vLLM | ~20 GB |
| qwen32b-gguf | P3 Heavy | 32B | GGUF | llama.cpp | ~20 GB |

**Notes:**
- HF models are full-precision (BF16/FP16) unless marked AWQ
- GGUF models use Q4_K_M quantization
- AWQ models use Int4 quantization (smaller, faster on vLLM)
- `llama3.1-8b` requires Meta approval — run `hf auth login` first
