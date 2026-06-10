# Download Models

## Config

Edit `configs/models.yaml` to enable/disable models and set destination:

```yaml
base_dir: /workspace/models    # Default destination

models:
  - name: qwen7b-gguf
    repo_id: Qwen/Qwen2.5-7B-Instruct-GGUF
    local_dir: gguf/qwen2.5-7b
    format: gguf
    include: "*q4_k_m.gguf"
    enabled: true    # ← Set true to download
```

All models are `enabled: false` by default. Flip to `true` to include.

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
uv run python scripts/download-models.py --only qwen7b-gguf qwen3b-gguf
```

## Preview without downloading

```bash
uv run python scripts/download-models.py --dry-run
```

## Skip already downloaded

```bash
uv run python scripts/download-models.py --skip-existing
```

## Available models

| Name | Phase | Size | Format |
|------|-------|------|--------|
| qwen0.6b | P0 Ultra-Light | 0.6B | HF |
| qwen0.6b-gguf | P0 Ultra-Light | 0.6B | GGUF |
| qwen1.5b | P1 Light | 1.5B | HF |
| qwen1.5b-gguf | P1 Light | 1.5B | GGUF |
| qwen3b | P2 Medium | 3B | HF |
| qwen3b-gguf | P2 Medium | 3B | GGUF |
| qwen7b | P3 Heavy | 7B | HF |
| qwen7b-gguf | P3 Heavy | 7B | GGUF |
