#!/usr/bin/env python3
"""Download models from HuggingFace based on configs/models.yaml.

Usage:
    uv run python scripts/download-models.py                         # All enabled
    uv run python scripts/download-models.py --only qwen0.5b        # Specific model
    uv run python scripts/download-models.py --phase p0 p1           # By phase
    uv run python scripts/download-models.py --phase embedding       # Embedding only
    uv run python scripts/download-models.py --dry-run               # Preview
    uv run python scripts/download-models.py --skip-existing         # Skip if exists
    uv run python scripts/download-models.py --dir /custom/path      # Custom dir
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_hf_cli() -> str:
    """Find the best available HuggingFace CLI."""
    # Try 'hf' first (newer), then 'huggingface-cli' (legacy)
    for cmd in ("hf", "huggingface-cli"):
        if shutil.which(cmd):
            return cmd
    print("ERROR: No HuggingFace CLI found. Install: pip install huggingface-hub", file=sys.stderr)
    sys.exit(1)


def download_hf(cli: str, repo_id: str, local_dir: Path, exclude: str | None = None):
    if cli == "hf":
        cmd = [cli, "download", repo_id, "--local-dir", str(local_dir)]
    else:
        cmd = [
            cli, "download", repo_id,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False",
        ]
        if exclude:
            cmd += ["--exclude", exclude]
    return subprocess.run(cmd)


def download_gguf(cli: str, repo_id: str, local_dir: Path, include: str, exclude: str | None = None):
    if cli == "hf":
        cmd = [cli, "download", repo_id, "--include", include, "--local-dir", str(local_dir)]
    else:
        cmd = [
            cli, "download", repo_id,
            "--include", include,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False",
        ]
        if exclude:
            cmd += ["--exclude", exclude]
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--dir", "-d", type=Path, help="Destination directory (overrides config)")
    parser.add_argument("--only", "-o", nargs="+", help="Download only these model names")
    parser.add_argument("--phase", "-p", nargs="+", help="Download by phase: p0, p1, p2, p3, embedding, all")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without downloading")
    parser.add_argument("--skip-existing", "-s", action="store_true", help="Skip if already downloaded")
    parser.add_argument("--list", "-l", action="store_true", help="List all models and exit")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # Resolve base_dir relative to project root
    project_root = Path(__file__).resolve().parent.parent
    config_base = Path(config.get("base_dir", "models"))
    if not config_base.is_absolute():
        config_base = project_root / config_base
    base_dir = args.dir or config_base

    models = config.get("models", [])

    if not models:
        print("No models defined in config.")
        sys.exit(0)

    # Filter by name
    if args.only:
        only_set = {n.lower() for n in args.only}
        models = [m for m in models if m.get("name", "").lower() in only_set]
        if not models:
            print(f"ERROR: No models matched: {args.only}", file=sys.stderr)
            sys.exit(1)

    # Filter by phase
    if args.phase:
        phase_set = {p.lower() for p in args.phase}
        if "all" not in phase_set:
            models = [m for m in models if m.get("phase", "").lower() in phase_set]
            if not models:
                print(f"ERROR: No models matched phases: {args.phase}", file=sys.stderr)
                sys.exit(1)

    G, C, D, B, R, NC = "\033[0;32m", "\033[0;36m", "\033[2m", "\033[1m", "\033[0;31m", "\033[0m"

    # List mode
    if args.list:
        print(f"\n  {B}Available Models:{NC}\n")
        print(f"  {C}{'Name':<20} {'Phase':<10} {'Format':<6} {'Enabled':<8} {'Repo'}{NC}")
        print(f"  {'─'*80}")
        for m in models:
            enabled = f"{G}yes{NC}" if m.get("enabled", True) else f"{R}no{NC}"
            print(f"  {m['name']:<20} {m.get('phase','?'):<10} {m.get('format','?'):<6} {enabled:<17} {m.get('repo_id','?')}")
        print()
        sys.exit(0)

    cli = find_hf_cli()

    enabled = [m for m in models if m.get("enabled", True)]
    disabled = [m for m in models if not m.get("enabled", True)]

    print(f"\n  {G}{B}{'='*55}{NC}")
    print(f"  {G}{B}  Model Downloader{NC}")
    print(f"  {G}{B}{'='*55}{NC}\n")
    print(f"  {C}Config{NC}     {args.config}")
    print(f"  {C}Dest dir{NC}   {base_dir}")
    print(f"  {C}CLI{NC}        {cli}")
    print(f"  {C}Models{NC}     {len(enabled)} enabled / {len(disabled)} disabled")

    if args.phase:
        print(f"  {C}Phase filter{NC} {', '.join(args.phase)}")
    if args.only:
        print(f"  {C}Name filter{NC}  {', '.join(args.only)}")
    print()

    if not enabled:
        print(f"  {D}No enabled models. Edit configs/models.yaml to enable.{NC}\n")
        sys.exit(0)

    results = {"downloaded": 0, "skipped": 0, "failed": 0}

    for model in enabled:
        name = model.get("name", "unknown")
        repo_id = model.get("repo_id")
        local_dir = base_dir / model.get("local_dir", name)
        fmt = model.get("format", "hf")
        include = model.get("include")
        exclude = model.get("exclude")

        if not repo_id:
            print(f"  SKIP {name}: no repo_id")
            results["skipped"] += 1
            continue

        if args.skip_existing and local_dir.exists() and any(local_dir.iterdir()):
            print(f"  SKIP {name}: exists at {local_dir}")
            results["skipped"] += 1
            continue

        if args.dry_run:
            print(f"  {B}DRY-RUN{NC} {name}")
            print(f"    repo:    {repo_id}")
            print(f"    dest:    {local_dir}")
            print(f"    format:  {fmt}")
            if include:
                print(f"    include: {include}")
            print()
            continue

        print(f"  {B}Download{NC} {name}")
        print(f"    repo:    {repo_id}")
        print(f"    dest:    {local_dir}")
        start = time.time()
        local_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "gguf" and include:
            result = download_gguf(cli, repo_id, local_dir, include, exclude)
        else:
            result = download_hf(cli, repo_id, local_dir, exclude)

        elapsed = time.time() - start

        if result.returncode == 0:
            size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            print(f"    {G}OK{NC} ({size / 1024**3:.1f} GB, {elapsed:.0f}s)")
            results["downloaded"] += 1
        else:
            print(f"    {R}FAILED{NC} (exit {result.returncode})")
            results["failed"] += 1
        print()

    print(f"  {G}{B}{'='*55}{NC}")
    print(f"  {G}Done:{NC} {results['downloaded']} downloaded, {results['skipped']} skipped, {results['failed']} failed")
    print(f"  {G}{B}{'='*55}{NC}\n")

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
