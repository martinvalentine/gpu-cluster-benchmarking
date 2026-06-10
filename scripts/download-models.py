#!/usr/bin/env python3
"""Download models from HuggingFace.

Usage:
    python scripts/download-models.py --dir /path/to/models          # Download all enabled
    python scripts/download-models.py --dir /path/to/models --only qwen7b-gguf
    python scripts/download-models.py --dir /path/to/models --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_hf(repo_id: str, local_dir: Path, exclude: str | None = None):
    cmd = [
        "huggingface-cli", "download", repo_id,
        "--local-dir", str(local_dir),
        "--local-dir-use-symlinks", "False",
    ]
    if exclude:
        cmd += ["--exclude", exclude]
    return subprocess.run(cmd)


def download_gguf(repo_id: str, local_dir: Path, include: str, exclude: str | None = None):
    cmd = [
        "huggingface-cli", "download", repo_id,
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
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/models.yaml"), help="Config file")
    parser.add_argument("--dir", "-d", type=Path, help="Destination directory for all models")
    parser.add_argument("--only", "-o", nargs="+", help="Download only these model names")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without downloading")
    parser.add_argument("--skip-existing", "-s", action="store_true", help="Skip if already downloaded")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    base_dir = args.dir or Path(config.get("base_dir", "/workspace/models"))
    models = config.get("models", [])

    if not models:
        print("No models defined in config.")
        sys.exit(0)

    if args.only:
        only_set = {n.lower() for n in args.only}
        models = [m for m in models if m.get("name", "").lower() in only_set]
        if not models:
            print(f"ERROR: No models matched: {args.only}", file=sys.stderr)
            sys.exit(1)

    G, C, D, B, NC = "\033[0;32m", "\033[0;36m", "\033[2m", "\033[1m", "\033[0m"

    print(f"\n  {G}{B}{'='*50}{NC}")
    print(f"  {G}{B}  Model Downloader{NC}")
    print(f"  {G}{B}{'='*50}{NC}\n")
    print(f"  {C}Config{NC}     {args.config}")
    print(f"  {C}Dest dir{NC}   {base_dir}")
    print(f"  {C}Models{NC}     {len(models)} total")

    enabled = [m for m in models if m.get("enabled", True)]
    disabled = [m for m in models if not m.get("enabled", True)]

    print(f"  {C}Enabled{NC}    {len(enabled)}")
    if disabled:
        print(f"  {D}Disabled{NC}   {len(disabled)}: {', '.join(m['name'] for m in disabled)}")
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
            result = download_gguf(repo_id, local_dir, include, exclude)
        else:
            result = download_hf(repo_id, local_dir, exclude)

        elapsed = time.time() - start

        if result.returncode == 0:
            size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            print(f"    {G}OK{NC} ({size / 1024**3:.1f} GB, {elapsed:.0f}s)")
            results["downloaded"] += 1
        else:
            print(f"    \033[0;31mFAILED{NC} (exit {result.returncode})")
            results["failed"] += 1
        print()

    print(f"  {G}{B}{'='*50}{NC}")
    print(f"  {G}Done:{NC} {results['downloaded']} downloaded, {results['skipped']} skipped, {results['failed']} failed")
    print(f"  {G}{B}{'='*50}{NC}\n")

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
