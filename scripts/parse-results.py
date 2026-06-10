#!/usr/bin/env python3
"""Aggregate benchmark JSON results into CSV summary (benchmark_plan.md F.2)."""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_vllm_result(data: dict) -> dict:
    """Extract metrics from vLLM benchmark_serving.py output."""
    return {
        "successful_requests": data.get("completed", data.get("successful_requests", 0)),
        "failed_requests": data.get("failed_requests", 0),
        "duration_s": round(data.get("duration", 0), 2),
        "req_throughput": round(data.get("request_throughput", 0), 3),
        "output_tok_s": round(data.get("output_throughput", 0), 2),
        "total_tok_s": round(data.get("total_token_throughput", 0), 2),
        "mean_ttft_ms": round(data.get("mean_ttft_ms", 0), 2),
        "median_ttft_ms": round(data.get("median_ttft_ms", 0), 2),
        "p99_ttft_ms": round(data.get("p99_ttft_ms", 0), 2),
        "mean_tpot_ms": round(data.get("mean_tpot_ms", 0), 2),
        "p99_tpot_ms": round(data.get("p99_tpot_ms", 0), 2),
        "mean_itl_ms": round(data.get("mean_itl_ms", 0), 2),
        "p99_itl_ms": round(data.get("p99_itl_ms", 0), 2),
    }


def parse_sglang_result(line: str) -> dict | None:
    """Parse a single JSONL line from sglang.bench_serving output."""
    try:
        data = json.loads(line)
        return {
            "successful_requests": 1,
            "failed_requests": 0 if data.get("error") is None else 1,
            "duration_s": round(data.get("latency", 0), 2),
            "req_throughput": 0,
            "output_tok_s": round(data.get("throughput", 0), 2),
            "total_tok_s": 0,
            "mean_ttft_ms": round(data.get("ttft", 0) * 1000, 2) if data.get("ttft") else 0,
            "median_ttft_ms": 0,
            "p99_ttft_ms": 0,
            "mean_tpot_ms": round(data.get("tpot", 0) * 1000, 2) if data.get("tpot") else 0,
            "p99_tpot_ms": 0,
            "mean_itl_ms": round(data.get("itl", 0) * 1000, 2) if data.get("itl") else 0,
            "p99_itl_ms": 0,
        }
    except (json.JSONDecodeError, KeyError):
        return None


def detect_framework(path: Path) -> str:
    """Detect framework from file path."""
    parts = [p.lower() for p in path.parts]
    if "vllm" in parts:
        return "vllm"
    if "sglang" in parts:
        return "sglang"
    if "llamacpp" in parts or "llama" in parts:
        return "llamacpp"
    if "litellm" in parts:
        return "litellm"
    return "unknown"


def detect_phase(path: Path) -> str:
    """Detect phase from file path."""
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("p1"):
            return "p1_light"
        if lower.startswith("p2"):
            return "p2_medium"
        if lower.startswith("p3"):
            return "p3_heavy"
    return "unknown"


def extract_concurrency(filename: str) -> int:
    """Extract concurrency number from filename like p1_light_conc32.json."""
    import re
    match = re.search(r"conc(\d+)", filename)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", "-d",
        type=Path,
        default=Path("results"),
        help="Root results directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-dir>/benchmark_summary.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_csv = args.output or results_dir / "benchmark_summary.csv"

    FIELDNAMES = [
        "framework", "phase", "concurrency", "file",
        "successful_requests", "failed_requests", "duration_s",
        "req_throughput", "output_tok_s", "total_tok_s",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "p99_itl_ms",
    ]

    rows = []

    # Process JSON files (vLLM, llama.cpp, LiteLLM)
    for json_file in sorted(results_dir.rglob("*.json")):
        if json_file.name == "benchmark_summary.csv":
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: Skipping {json_file}: {e}", file=sys.stderr)
            continue

        framework = detect_framework(json_file)
        phase = detect_phase(json_file)
        conc = extract_concurrency(json_file.name)

        metrics = parse_vllm_result(data)
        rows.append({
            "framework": framework,
            "phase": phase,
            "concurrency": conc,
            "file": json_file.name,
            **metrics,
        })

    # Process JSONL files (SGLang)
    for jsonl_file in sorted(results_dir.rglob("*.jsonl")):
        framework = detect_framework(jsonl_file)
        phase = detect_phase(jsonl_file)
        conc = extract_concurrency(jsonl_file.name)

        total_metrics = {
            "successful_requests": 0,
            "failed_requests": 0,
            "mean_ttft_ms": [],
            "mean_tpot_ms": [],
            "mean_itl_ms": [],
        }

        try:
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                result = parse_sglang_result(line)
                if result:
                    total_metrics["successful_requests"] += result["successful_requests"]
                    total_metrics["failed_requests"] += result["failed_requests"]
                    if result["mean_ttft_ms"] > 0:
                        total_metrics["mean_ttft_ms"].append(result["mean_ttft_ms"])
                    if result["mean_tpot_ms"] > 0:
                        total_metrics["mean_tpot_ms"].append(result["mean_tpot_ms"])
                    if result["mean_itl_ms"] > 0:
                        total_metrics["mean_itl_ms"].append(result["mean_itl_ms"])
        except OSError as e:
            print(f"  WARN: Skipping {jsonl_file}: {e}", file=sys.stderr)
            continue

        def avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0

        rows.append({
            "framework": framework,
            "phase": phase,
            "concurrency": conc,
            "file": jsonl_file.name,
            "successful_requests": total_metrics["successful_requests"],
            "failed_requests": total_metrics["failed_requests"],
            "duration_s": 0,
            "req_throughput": 0,
            "output_tok_s": 0,
            "total_tok_s": 0,
            "mean_ttft_ms": avg(total_metrics["mean_ttft_ms"]),
            "median_ttft_ms": 0,
            "p99_ttft_ms": 0,
            "mean_tpot_ms": avg(total_metrics["mean_tpot_ms"]),
            "p99_tpot_ms": 0,
            "mean_itl_ms": avg(total_metrics["mean_itl_ms"]),
            "p99_itl_ms": 0,
        })

    if not rows:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(0)

    # Sort by framework, phase, concurrency
    rows.sort(key=lambda r: (r["framework"], r["phase"], r["concurrency"]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} results -> {output_csv}")


if __name__ == "__main__":
    main()
