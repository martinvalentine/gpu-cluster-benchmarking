#!/usr/bin/env python3
"""LiteLLM Semantic Cache Proxy — Programmatic launcher with cache management.

Launches the LiteLLM proxy with Redis-backed semantic caching and provides
utilities for cache inspection, invalidation, and hit-rate monitoring.

Usage:
    # Start proxy (equivalent to scripts/run/run-proxy.sh but in-process)
    python src/caching/litellm-semantic-cache.py

    # Start with custom settings
    python src/caching/litellm-semantic-cache.py --port 4000

    # Flush cache
    python src/caching/litellm-semantic-cache.py --flush

    # Show cache stats
    python src/caching/litellm-semantic-cache.py --stats
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import redis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "litellm_config.yaml"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
CACHE_PREFIX = "litellm:"


def get_redis_client(host: str = REDIS_HOST, port: int = REDIS_PORT) -> redis.Redis:
    return redis.Redis(host=host, port=port, decode_responses=True)


def flush_cache(r: redis.Redis) -> int:
    keys = r.keys(f"{CACHE_PREFIX}*")
    if keys:
        r.delete(*keys)
    return len(keys)


def cache_stats(r: redis.Redis) -> dict:
    keys = r.keys(f"{CACHE_PREFIX}*")
    mem = r.info("memory")
    stats = r.info("stats")
    hits = stats.get("keyspace_hits", 0)
    misses = stats.get("keyspace_misses", 0)
    total = hits + misses
    return {
        "cached_entries": len(keys),
        "redis_used_memory": mem.get("used_memory_human", "N/A"),
        "redis_peak_memory": mem.get("used_memory_peak_human", "N/A"),
        "keyspace_hits": hits,
        "keyspace_misses": misses,
        "hit_rate": f"{hits / total * 100:.1f}%" if total > 0 else "N/A",
    }


def start_proxy(config_path: str, port: int, host: str, debug: bool) -> None:
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"ERROR: Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        r = get_redis_client()
        r.ping()
        print(f"  Redis: connected at {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        print(f"  WARNING: Redis not reachable at {REDIS_HOST}:{REDIS_PORT}", file=sys.stderr)

    print(f"\n  LiteLLM Semantic Cache Proxy")
    print(f"  ├─ Config:   {config_file}")
    print(f"  ├─ Endpoint: http://{host}:{port}")
    print(f"  ├─ Cache:    redis-semantic (threshold=0.95, ttl=3600s)")
    print(f"  ├─ Router:   least-busy")
    print(f"  └─ Backends: vLLM :8000 | llama.cpp :8001 | SGLang :8002\n")

    cmd = [
        sys.executable, "-m", "litellm",
        "--config", str(config_file),
        "--port", str(port),
        "--host", host,
    ]
    if debug:
        cmd.append("--detailed_debug")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  Proxy stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="LiteLLM Semantic Cache Proxy with Redis-backed semantic caching"
    )
    parser.add_argument(
        "-c", "--config",
        default=str(DEFAULT_CONFIG),
        help="LiteLLM config YAML path",
    )
    parser.add_argument("-p", "--port", type=int, default=4000, help="Proxy port")
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("-d", "--debug", action="store_true", help="Detailed debug logging")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--flush", action="store_true", help="Flush all cached entries")
    group.add_argument("--stats", action="store_true", help="Show cache stats")

    args = parser.parse_args()

    r = get_redis_client()

    if args.flush:
        n = flush_cache(r)
        print(f"Flushed {n} cached entries.")
        return

    if args.stats:
        stats = cache_stats(r)
        print(json.dumps(stats, indent=2))
        return

    start_proxy(args.config, args.port, args.host, args.debug)


if __name__ == "__main__":
    main()
