#!/usr/bin/env python3
"""Generate litellm_config.yaml from configs/models.yaml.

Reads the single-source-of-truth model config and produces a LiteLLM proxy
config with correct model names, backend IDs, and ports.

Usage:
    uv run python scripts/gen-litellm-config.py                    # Default output
    uv run python scripts/gen-litellm-config.py -o custom.yaml     # Custom output
    uv run python scripts/gen-litellm-config.py --preview          # Print to stdout only
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_id(model: dict, base_dir: Path, project_root: Path) -> str:
    """Determine the model ID the backend will report.

    - vLLM/SGLang: reports the path passed to --model (relative to cwd)
    - llama.cpp: reports the GGUF filename

    Returns relative paths for vLLM/SGLang (matching how servers are started).
    """
    backend = model.get("backend", "vllm")
    local_dir = model.get("local_dir", "")

    if backend in ("vllm", "sglang"):
        # vLLM/SGLang report the path passed to --model
        # start-all-tmux.sh passes: models/hf/qwen2.5-0.6b (relative)
        return f"models/{local_dir}"

    if backend == "llamacpp":
        # Check if this is an embedding model (special case)
        if model.get("phase") == "embedding":
            # Embedding model: use the GGUF filename without path
            gguf_dir = base_dir / local_dir
            if gguf_dir.exists():
                for f in sorted(gguf_dir.glob("*.gguf")):
                    if f.suffix == ".gguf":
                        return f.name
            # Fallback: use repo_id last segment
            repo_id = model.get("repo_id", "")
            return repo_id.split("/")[-1] + ".gguf"

        # llama.cpp reports the GGUF filename
        gguf_dir = base_dir / local_dir
        if gguf_dir.exists():
            for f in sorted(gguf_dir.glob("*.gguf")):
                if f.suffix == ".gguf":
                    return f.name
        # Fallback: construct from local_dir name
        return Path(local_dir).name + ".gguf"

    return f"models/{local_dir}"


def generate_config(config: dict, project_root: Path) -> dict:
    """Generate litellm_config.yaml content from models.yaml."""
    ports = config.get("ports", {})
    base_dir = Path(config.get("base_dir", "models"))
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    models = config.get("models", [])

    # Collect endpoint-enabled models
    model_list = []
    for m in models:
        if not m.get("enabled", True):
            continue
        if m.get("endpoint") is False:
            continue

        backend = m.get("backend", "vllm")
        port = ports.get(backend, 8000)
        proxy_name = m.get("proxy_name", m.get("name", "unknown"))
        model_id = resolve_model_id(m, base_dir, project_root)

        model_list.append({
            "model_name": proxy_name,
            "litellm_params": {
                "model": f"openai/{model_id}",
                "api_base": f"http://localhost:{port}/v1",
                "api_key": "EMPTY",
            },
        })

    litellm_config = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": "least-busy",
        },
        "litellm_settings": {
            "set_verbose": False,
            "num_retries": 2,
            "request_timeout": 120,
            "success_callback": ["prometheus"],
            "failure_callback": ["prometheus"],
            "cache": True,
            "cache_params": {
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl": 3600,
            },
        },
    }

    return litellm_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/models.yaml"),
        help="Input config file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("litellm_config.yaml"),
        help="Output litellm config file",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Print to stdout only, don't write file",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    args.output = project_root / args.output

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(project_root / args.config)
    litellm_config = generate_config(config, project_root)

    output = yaml.dump(litellm_config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.preview:
        print(output)
        sys.exit(0)

    args.output.write_text(output)

    model_count = len(litellm_config.get("model_list", []))
    G, C, B, NC = "\033[0;32m", "\033[0;36m", "\033[1m", "\033[0m"

    print(f"\n  {G}{B}Generated LiteLLM config{NC}")
    print(f"  {C}Output{NC}     {args.output}")
    print(f"  {C}Models{NC}     {model_count} endpoint(s)")
    print()
    for m in litellm_config.get("model_list", []):
        name = m["model_name"]
        backend_id = m["litellm_params"]["model"]
        api_base = m["litellm_params"]["api_base"]
        print(f"  {C}{name:<25}{NC} → {backend_id} @ {api_base}")
    print()


if __name__ == "__main__":
    main()
