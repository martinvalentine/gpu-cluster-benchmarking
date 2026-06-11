#!/usr/bin/env python3
"""Aggregate benchmark results (vLLM JSON, SGLang JSONL, llama.cpp TSV) into CSV summary.

Outputs a CSV aligned with benchmark_plan.md Section E.1 tracking table:
  model, backend, phase, concurrency, successful_requests, failed_requests,
  duration_s, req_throughput, output_tok_s, total_tok_s,
  mean_ttft_ms, median_ttft_ms, p99_ttft_ms,
  mean_tpot_ms, p99_tpot_ms, mean_itl_ms, p99_itl_ms,
  total_input_tokens, total_generated_tokens

Lightweight CSV aggregation for pipeline use. For full formatted output
(terminal + CSV + markdown), see report.py.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

FIELDNAMES = [
    "model", "backend", "phase", "concurrency", "file",
    "successful_requests", "failed_requests", "duration_s",
    "req_throughput", "output_tok_s", "total_tok_s",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "total_input_tokens", "total_generated_tokens",
]


def detect_backend(path: Path) -> str:
    """Detect backend from file path."""
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


def detect_model(path: Path, results_dir: Path) -> str:
    """Extract model name from run directory name or parent path.

    Looks for 'run-N' directory and maps it to model name by scanning
    the bench log, or falls back to the run directory name.
    """
    # Walk up to find 'run-N' directory
    for part in path.parts:
        if part.startswith("run-"):
            return part
    # Fallback: use grandparent directory name
    try:
        rel = path.relative_to(results_dir)
        if len(rel.parts) >= 2:
            return rel.parts[0]  # run-N
    except ValueError:
        pass
    return "unknown"


def extract_concurrency(filename: str) -> int:
    """Extract concurrency number from filename."""
    match = re.search(r"conc(\d+)", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"concurrent_(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def empty_metrics() -> dict:
    """Return a dict with all metric keys set to zero."""
    return {k: 0 for k in FIELDNAMES if k not in ("model", "backend", "phase", "concurrency", "file")}


def parse_vllm_json(data: dict) -> dict:
    """Extract metrics from vLLM benchmark_serving.py JSON output."""
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
        "total_input_tokens": data.get("total_input_tokens", 0),
        "total_generated_tokens": data.get("total_generated_tokens", 0),
    }


def parse_sglang_jsonl(line: str) -> dict | None:
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
            "total_input_tokens": data.get("prompt_tokens", 0),
            "total_generated_tokens": data.get("completion_tokens", 0),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def parse_llamacpp_tsv_dir(tsv_dir: Path, model: str, phase: str) -> list[dict]:
    """Parse all llama.cpp TSV files in a directory into summary rows.

    Produces one row per concurrent test level plus a single-user row.
    """
    rows = []

    # ── Parse concurrent_N.tsv files (primary metrics) ─────────────
    for tsv_file in sorted(tsv_dir.glob("concurrent_*.tsv")):
        conc = extract_concurrency(tsv_file.name)
        try:
            lines = tsv_file.read_text().strip().split("\n")
            if len(lines) < 2:
                continue
            header = lines[0].split("\t")
            values = lines[1].split("\t")
            row_data = dict(zip(header, values))
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {tsv_file}: {e}", file=sys.stderr)
            continue

        completed = int(row_data.get("completed", 0))
        total_tokens = int(row_data.get("total_tokens", 0))
        wall_ms = int(row_data.get("wall_ms", 0))
        agg_tps = float(row_data.get("agg_tps", 0))
        per_user_tps = float(row_data.get("per_user_tps", 0))
        avg_latency_ms = float(row_data.get("avg_latency_ms", 0))

        wall_s = wall_ms / 1000 if wall_ms else 0
        req_throughput = round(completed / wall_s, 3) if wall_s else 0

        m = empty_metrics()
        m["successful_requests"] = completed
        m["failed_requests"] = max(0, conc - completed)
        m["duration_s"] = round(wall_s, 2)
        m["req_throughput"] = req_throughput
        m["output_tok_s"] = round(agg_tps, 2)
        m["total_tok_s"] = round(agg_tps, 2)
        m["mean_ttft_ms"] = round(avg_latency_ms, 2)
        m["total_input_tokens"] = total_tokens
        m["total_generated_tokens"] = total_tokens

        rows.append({
            "model": model,
            "backend": "llamacpp",
            "phase": phase,
            "concurrency": conc,
            "file": tsv_file.name,
            **m,
        })

    # ── Parse single_user.tsv (single-user latency) ────────────────
    single_file = tsv_dir / "single_user.tsv"
    if single_file.exists():
        try:
            lines = single_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                header = lines[0].split("\t")
                total_completion = 0
                total_prompt = 0
                decode_sum = 0
                decode_count = 0
                for line in lines[1:]:
                    vals = dict(zip(header, line.split("\t")))
                    total_completion += int(vals.get("completion_tokens", 0))
                    total_prompt += int(vals.get("prompt_tokens", 0))
                    dt = float(vals.get("decode_tps", 0))
                    if dt > 0:
                        decode_sum += dt
                        decode_count += 1

                avg_decode = round(decode_sum / decode_count, 2) if decode_count else 0

                m = empty_metrics()
                m["successful_requests"] = len(lines) - 1
                m["output_tok_s"] = avg_decode
                m["total_tok_s"] = avg_decode
                m["total_input_tokens"] = total_prompt
                m["total_generated_tokens"] = total_completion

                rows.append({
                    "model": model,
                    "backend": "llamacpp",
                    "phase": phase,
                    "concurrency": 1,
                    "file": "single_user.tsv",
                    **m,
                })
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {single_file}: {e}", file=sys.stderr)

    # ── Parse long_context.tsv (prefill speed) ─────────────────────
    long_file = tsv_dir / "long_context.tsv"
    if long_file.exists():
        try:
            lines = long_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                header = lines[0].split("\t")
                for line in lines[1:]:
                    vals = dict(zip(header, line.split("\t")))
                    ctx = int(vals.get("ctx_tokens", 0))
                    prefill_tps = vals.get("prefill_tps", "?")
                    if prefill_tps == "?" or not prefill_tps:
                        continue
                    m = empty_metrics()
                    m["output_tok_s"] = round(float(prefill_tps), 2)
                    m["total_input_tokens"] = ctx
                    rows.append({
                        "model": model,
                        "backend": "llamacpp",
                        "phase": phase,
                        "concurrency": 0,
                        "file": f"long_context_ctx{ctx}.tsv",
                        **m,
                    })
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {long_file}: {e}", file=sys.stderr)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
    rows = []

    # ── 1. Process JSON files (vLLM stress, llama-benchy, etc.) ────
    for json_file in sorted(results_dir.rglob("*.json")):
        if json_file.name == "benchmark_summary.csv":
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: Skipping {json_file}: {e}", file=sys.stderr)
            continue

        backend = detect_backend(json_file)
        phase = detect_phase(json_file)
        model = detect_model(json_file, results_dir)
        conc = extract_concurrency(json_file.name)

        metrics = parse_vllm_json(data)
        rows.append({
            "model": model,
            "backend": backend,
            "phase": phase,
            "concurrency": conc,
            "file": json_file.name,
            **metrics,
        })

    # ── 2. Process JSONL files (SGLang) ────────────────────────────
    for jsonl_file in sorted(results_dir.rglob("*.jsonl")):
        backend = detect_backend(jsonl_file)
        phase = detect_phase(jsonl_file)
        model = detect_model(jsonl_file, results_dir)
        conc = extract_concurrency(jsonl_file.name)

        agg = {
            "successful_requests": 0,
            "failed_requests": 0,
            "ttfts": [], "tpots": [], "itls": [],
            "total_input": 0, "total_output": 0,
        }

        try:
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                result = parse_sglang_jsonl(line)
                if result:
                    agg["successful_requests"] += result["successful_requests"]
                    agg["failed_requests"] += result["failed_requests"]
                    agg["total_input"] += result["total_input_tokens"]
                    agg["total_output"] += result["total_generated_tokens"]
                    if result["mean_ttft_ms"] > 0:
                        agg["ttfts"].append(result["mean_ttft_ms"])
                    if result["mean_tpot_ms"] > 0:
                        agg["tpots"].append(result["mean_tpot_ms"])
                    if result["mean_itl_ms"] > 0:
                        agg["itls"].append(result["mean_itl_ms"])
        except OSError as e:
            print(f"  WARN: Skipping {jsonl_file}: {e}", file=sys.stderr)
            continue

        def avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0

        rows.append({
            "model": model,
            "backend": backend,
            "phase": phase,
            "concurrency": conc,
            "file": jsonl_file.name,
            "successful_requests": agg["successful_requests"],
            "failed_requests": agg["failed_requests"],
            "duration_s": 0,
            "req_throughput": 0,
            "output_tok_s": 0,
            "total_tok_s": 0,
            "mean_ttft_ms": avg(agg["ttfts"]),
            "median_ttft_ms": 0,
            "p99_ttft_ms": 0,
            "mean_tpot_ms": avg(agg["tpots"]),
            "p99_tpot_ms": 0,
            "mean_itl_ms": avg(agg["itls"]),
            "p99_itl_ms": 0,
            "total_input_tokens": agg["total_input"],
            "total_generated_tokens": agg["total_output"],
        })

    # ── 3. Process TSV files (llama.cpp) ───────────────────────────
    processed_dirs = set()
    for tsv_file in sorted(results_dir.rglob("*.tsv")):
        parent = tsv_file.parent
        if parent.name != "llamacpp" or parent in processed_dirs:
            continue
        processed_dirs.add(parent)
        model = detect_model(parent, results_dir)
        phase = detect_phase(parent)
        tsv_rows = parse_llamacpp_tsv_dir(parent, model, phase)
        rows.extend(tsv_rows)

    if not rows:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(0)

    # Sort by backend, model, phase, concurrency
    rows.sort(key=lambda r: (r["backend"], r["model"], r["phase"], r["concurrency"]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} results -> {output_csv}")


if __name__ == "__main__":
    main()
