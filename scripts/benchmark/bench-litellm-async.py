#!/usr/bin/env python3
"""Async benchmark for LiteLLM proxy — measures TTFT, TPOT, ITL, throughput.

Sends concurrent streaming requests through the LiteLLM proxy and measures
per-request latency metrics. Supports a configurable ratio of repeated prompts
(cache hits) vs unique prompts (cache misses) to evaluate semantic cache impact.

Usage:
    python bench-litellm-async.py
    python bench-litellm-async.py --concurrency 32 --num-requests 200
    python bench-litellm-async.py --base-url http://gpu-pod:4000 --model qwen32b-sglang
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path
from statistics import mean, median, quantiles

import aiohttp

REPEATED_PROMPTS = [
    "Explain the difference between TCP and UDP in detail",
    "What is the CAP theorem in distributed systems?",
    "Describe the attention mechanism in transformers",
    "How does PagedAttention work in vLLM?",
    "Explain gradient descent and its variants",
    "What are the key differences between SQL and NoSQL databases?",
    "Describe how garbage collection works in Java",
    "Explain the Raft consensus algorithm",
    "What is the purpose of a load balancer?",
    "How does HTTPS encryption work?",
]


def build_prompts(num_requests: int, cache_ratio: int) -> list[str]:
    """Build a prompt list with the given cache-hit ratio.

    Args:
        num_requests: Total number of prompts to generate.
        cache_ratio: Percentage (0-100) of prompts drawn from the repeated pool.

    Returns:
        List of prompt strings.
    """
    num_repeated = int(num_requests * cache_ratio / 100)
    num_unique = num_requests - num_repeated

    prompts = []
    for _ in range(num_repeated):
        prompts.append(random.choice(REPEATED_PROMPTS))
    for i in range(num_unique):
        prompts.append(f"Unique query {i}: {uuid.uuid4()}")
    random.shuffle(prompts)
    return prompts


async def stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send a single streaming chat completion and measure latency metrics.

    Returns:
        Dict with ttft_ms, tpot_ms, itl_ms, total_ms, token_count, error.
    """
    start = time.perf_counter()
    first_token_time = None
    token_times: list[float] = []
    error = None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "ttft_ms": 0,
                    "tpot_ms": 0,
                    "itl_ms": [],
                    "total_ms": 0,
                    "token_count": 0,
                    "error": f"HTTP {resp.status}: {body[:200]}",
                }

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

    except Exception as e:
        error = str(e)

    end = time.perf_counter()

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    total_ms = (end - start) * 1000

    inter_token_latencies = []
    if len(token_times) > 1:
        inter_token_latencies = [
            (b - a) * 1000 for a, b in zip(token_times, token_times[1:])
        ]

    tpot_ms = mean(inter_token_latencies) if inter_token_latencies else 0

    return {
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "itl_ms": [round(x, 2) for x in inter_token_latencies],
        "total_ms": round(total_ms, 2),
        "token_count": len(token_times),
        "error": error,
    }


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    qs = quantiles(values, n=100)
    return qs[min(p - 1, 98)]


async def run_benchmark(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    cache_ratio: int,
) -> dict:
    """Run the full benchmark and return aggregated results."""
    prompts = build_prompts(num_requests, cache_ratio)
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    print(f"  Sending {num_requests} requests (concurrency={concurrency})...")
    bench_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:

        async def bounded(prompt: str):
            async with semaphore:
                return await stream_request(
                    session, base_url, model, prompt, max_tokens
                )

        tasks = [bounded(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    bench_duration = time.perf_counter() - bench_start

    ok = [r for r in results if isinstance(r, dict) and r.get("error") is None]
    failed = [r for r in results if isinstance(r, dict) and r.get("error")]
    exceptions = [r for r in results if not isinstance(r, dict)]

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    all_itl = []
    for r in ok:
        all_itl.extend(r["itl_ms"])
    total_tokens = sum(r["token_count"] for r in ok)

    summary = {
        "successful_requests": len(ok),
        "failed_requests": len(failed) + len(exceptions),
        "benchmark_duration_s": round(bench_duration, 2),
        "total_generated_tokens": total_tokens,
        "request_throughput_rps": round(len(ok) / bench_duration, 3),
        "output_token_throughput_tps": round(total_tokens / bench_duration, 2),
        "ttft": _stats(ttfts),
        "tpot": _stats(tpots),
        "itl": _stats(all_itl),
    }

    if failed:
        summary["sample_errors"] = [r["error"] for r in failed[:3]]

    return summary


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    qs = quantiles(values, n=100)
    return {
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "p99": round(qs[98], 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "count": len(values),
    }


def print_report(summary: dict) -> None:
    """Print a human-readable benchmark report."""
    print()
    print("=" * 60)
    print("  LiteLLM Proxy Benchmark Results")
    print("=" * 60)
    print()

    print(f"  Successful requests:     {summary['successful_requests']}")
    print(f"  Failed requests:         {summary['failed_requests']}")
    print(f"  Duration:                {summary['benchmark_duration_s']}s")
    print(f"  Generated tokens:        {summary['total_generated_tokens']}")
    print(f"  Request throughput:      {summary['request_throughput_rps']} req/s")
    print(f"  Token throughput:        {summary['output_token_throughput_tps']} tok/s")
    print()

    for metric, label in [("ttft", "TTFT"), ("tpot", "TPOT"), ("itl", "ITL")]:
        s = summary[metric]
        print(f"  {label} (ms):")
        print(f"    Mean:    {s['mean']}")
        print(f"    Median:  {s['median']}")
        print(f"    P99:     {s['p99']}")
        print(f"    Min:     {s['min']}")
        print(f"    Max:     {s['max']}")
        print()

    if "sample_errors" in summary:
        print("  Sample errors:")
        for err in summary["sample_errors"]:
            print(f"    - {err}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Async benchmark for LiteLLM proxy with semantic cache"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:4000", help="LiteLLM proxy URL"
    )
    parser.add_argument("--model", default="qwen35b-llamacpp", help="Model name")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent")
    parser.add_argument("--num-requests", type=int, default=100, help="Total requests")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens/response")
    parser.add_argument(
        "--cache-ratio",
        type=int,
        default=60,
        help="Percentage of repeated prompts (0-100)",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"\n  LiteLLM Async Benchmark")
    print(f"  ├─ URL:        {args.base_url}")
    print(f"  ├─ Model:      {args.model}")
    print(f"  ├─ Concurrency:{args.concurrency}")
    print(f"  ├─ Requests:   {args.num_requests}")
    print(f"  ├─ Max tokens: {args.max_tokens}")
    print(f"  └─ Cache ratio:{args.cache_ratio}% repeated\n")

    summary = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            cache_ratio=args.cache_ratio,
        )
    )

    print_report(summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
