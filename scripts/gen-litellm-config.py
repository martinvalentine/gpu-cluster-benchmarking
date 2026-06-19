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


class UnresolvableFilename(Exception):
    """GGUF file not found in any search dir."""

    def __init__(self, model_name, local_dir, include_pattern, search_dirs_checked):
        self.model_name = model_name
        self.local_dir = local_dir
        self.include_pattern = include_pattern
        self.search_dirs_checked = search_dirs_checked
        super().__init__(self._format())

    def _format(self) -> str:
        lines = [
            f"Could not resolve GGUF filename for '{self.model_name}'",
            "Searched:",
        ]
        for d in self.search_dirs_checked:
            lines.append(f"  {d}/{self.local_dir}/")
        lines += [
            f"Include pattern from config: {self.include_pattern!r}",
            "Remediation:",
            "  - Run inside the Docker container (base_dir=/workspace/models is set)",
            "  - Or pass --base-dir to point at your local model location",
            "  - Or set $LITELLM_BASE_DIR to a path that contains the model files",
        ]
        return "\n".join(lines)


class AmbiguousFilename(Exception):
    """Multiple GGUF files matched in a single search dir."""

    def __init__(self, model_name, local_dir, include_pattern, candidates, search_dir):
        self.model_name = model_name
        self.local_dir = local_dir
        self.include_pattern = include_pattern
        self.candidates = candidates
        self.search_dir = search_dir
        super().__init__(self._format())

    def _format(self) -> str:
        lines = [
            f"Ambiguous GGUF match for '{self.model_name}' in {self.search_dir}/{self.local_dir}/",
            f"Include pattern: {self.include_pattern!r}",
            "Multiple candidates:",
        ]
        for c in self.candidates:
            lines.append(f"  {c.name}")
        lines += [
            "Remediation:",
            "  - Move all but one file out of the directory, or",
            "  - Tighten the 'include' pattern in models.yaml to match exactly one file",
        ]
        return "\n".join(lines)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_id(model: dict, base_dir: Path, project_root: Path) -> str:  # noqa: F811
    """Backwards-compat wrapper. Delegates to resolve_gguf_filename / resolve_hf_path."""
    # Kept for callers that import the old name; new code uses resolve_model_path.
    backend = model.get("backend", "vllm")
    local_dir = model.get("local_dir", "")
    if backend == "llamacpp":
        # Old signature: only one base_dir. Wrap in a list for the new helper.
        return _legacy_resolve_gguf(model, base_dir)
    return str(base_dir / local_dir)


def _legacy_resolve_gguf(model, base_dir):
    """Single-base-dir variant of resolve_gguf_filename. Used by old callers."""
    gguf_dir = base_dir / model.get("local_dir", "")
    if gguf_dir.exists():
        for f in sorted(gguf_dir.glob("*.gguf")):
            if f.suffix == ".gguf":
                return f.name
    raise UnresolvableFilename(
        model_name=model.get("name", "unknown"),
        local_dir=model.get("local_dir", ""),
        include_pattern=model.get("include"),
        search_dirs_checked=[base_dir],
    )


def resolve_gguf_filename(model, search_dirs):
    """Resolve the GGUF filename for a llamacpp model.

    Tries each search_dir in order. First match wins. Raises:
    - AmbiguousFilename: if multiple .gguf files exist in a single search_dir
    - UnresolvableFilename: if no .gguf files exist in any search_dir
    """
    local_dir = model.get("local_dir", "")
    include = model.get("include")

    for search_dir in search_dirs:
        gguf_dir = Path(search_dir) / local_dir
        if not gguf_dir.exists():
            continue
        matches = sorted(p for p in gguf_dir.glob("*.gguf") if p.is_file())
        if len(matches) > 1:
            raise AmbiguousFilename(
                model_name=model.get("name", "unknown"),
                local_dir=local_dir,
                include_pattern=include,
                candidates=matches,
                search_dir=search_dir,
            )
        if len(matches) == 1:
            return matches[0].name

    raise UnresolvableFilename(
        model_name=model.get("name", "unknown"),
        local_dir=local_dir,
        include_pattern=include,
        search_dirs_checked=[Path(d) for d in search_dirs],
    )


def resolve_hf_path(model, search_dirs, strict):
    """Return the HF model path. If strict and missing, raise with remediation."""
    path = Path(search_dirs[0]) / model.get("local_dir", "")
    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"HF model '{model.get('name', 'unknown')}' directory not found: {path}\n"
                f"  --base-dir was passed; expected the path to exist on this host.\n"
                f"Remediation:\n"
                f"  - Verify the --base-dir argument points to a real directory\n"
                f"  - Or drop --base-dir to use the default (container path) with soft warnings\n"
                f"  - Or set $LITELLM_BASE_DIR to a path that contains the model files"
            )
        # soft warning path: return path anyway, caller notes it
    return str(path)


def resolve_model_path(model, search_dirs, strict):
    """Thin dispatcher. Routes to the backend-specific resolver.

    Branching lives entirely inside this function — main() never
    switches on model['backend'] directly. The `strict` flag is
    consumed only by the HF helper; GGUF resolution failures are
    always hard (no silent fallback to construct a fake filename).
    """
    if model.get("backend") == "llamacpp":
        return resolve_gguf_filename(model, search_dirs)
    return resolve_hf_path(model, search_dirs, strict)


def generate_config(config: dict, project_root: Path, search_dirs, strict) -> dict:
    """Generate litellm_config.yaml content from models.yaml.

    search_dirs: list of directories to search for model files.
                 CLI-passed dirs (via --base-dir) come first;
                 config['base_dir'] is appended last as the default.
    strict: if True, missing HF paths raise hard errors.
            If False, missing HF paths produce soft warnings (WARN tag
            in summary, model still included in output).
    """
    ports = config.get("ports", {})
    base_dir = Path(config.get("base_dir", "models"))
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir
    # search_dirs: CLI-passed first, then config default
    all_search_dirs = list(search_dirs) + [base_dir]

    models = config.get("models", [])
    hard_errors: list[str] = []
    soft_warnings: list[tuple[str, str]] = []  # (model_name, warning_text)
    model_list = []

    for m in models:
        if not m.get("enabled", True):
            continue
        if m.get("endpoint") is False:
            continue

        try:
            model_id = resolve_model_path(m, all_search_dirs, strict=strict)
        except (UnresolvableFilename, AmbiguousFilename, FileNotFoundError) as e:
            hard_errors.append(str(e))
            continue

        # Soft warning: HF path missing in non-strict mode
        if m.get("backend") in ("vllm", "sglang"):
            if not Path(model_id).exists():
                soft_warnings.append((m.get("name", "unknown"),
                                     f"{model_id} not found in {Path(model_id).parent}"))

        backend = m.get("backend", "vllm")
        port = m.get("api_port", ports.get(backend, 8000))
        proxy_name = m.get("proxy_name", m.get("name", "unknown"))

        model_list.append({
            "model_name": proxy_name,
            "litellm_params": {
                "model": f"openai/{model_id}",
                "api_base": f"http://localhost:{port}/v1",
                "api_key": "EMPTY",
            },
        })

    # Exit with all errors printed if any hard error
    if hard_errors:
        for err in hard_errors:
            print(err, file=sys.stderr)
        print(f"\nAborted: {len(hard_errors)} hard error(s), 0 models written.", file=sys.stderr)
        sys.exit(1)

    return {
        "model_list": model_list,
        "router_settings": {"routing_strategy": "least-busy"},
        "litellm_settings": {
            "set_verbose": False,
            "num_retries": 2,
            "request_timeout": 120,
            "success_callback": ["prometheus"],
            "failure_callback": ["prometheus"],
            "cache": True,
            "cache_params": {"type": "redis", "host": "localhost", "port": 6379, "ttl": 3600},
        },
    }


def print_summary(litellm_config, config, project_root, args, file=None):
    """Placeholder; real implementation in Task 9."""
    if file is None:
        file = sys.stdout
    model_count = len(litellm_config.get("model_list", []))
    print(f"\n  Generated LiteLLM config ({model_count} models)", file=file)


def main(args=None):
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
    parser.add_argument(
        "--base-dir",
        action="append",
        default=None,
        dest="base_dirs",
        help="Path to search for models (can be passed multiple times; CLI dirs "
             "are prepended to the config's base_dir)",
    )
    args = parser.parse_args(args)

    # Reject empty --base-dir at parse time (before any other work)
    if args.base_dirs is not None:
        for d in args.base_dirs:
            if not d:
                parser.error("--base-dir cannot be empty (check shell variable expansion)")

    # strict is an explicit intent signal: user passed --base-dir, so they
    # expect validation. False otherwise (default config base_dir is
    # expected to live in the container).
    strict = args.base_dirs is not None
    cli_base_dirs = args.base_dirs or []

    project_root = Path(__file__).resolve().parent.parent
    args.config = project_root / args.config
    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    litellm_config = generate_config(config, project_root, search_dirs=cli_base_dirs, strict=strict)

    output = yaml.dump(litellm_config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    if args.preview:
        print(output)
        sys.exit(0)
    args.output = project_root / args.output
    args.output.write_text(output)
    print_summary(litellm_config, config, project_root, args, file=sys.stderr)


if __name__ == "__main__":
    main()
