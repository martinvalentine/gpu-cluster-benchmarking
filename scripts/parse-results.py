#!/usr/bin/env python3
"""Aggregate benchmark results into CSV summary and comparison report.

Outputs:
  1. benchmark_summary.csv — detailed, one row per result file
  2. benchmark_report.tsv — pivoted comparison table (Phase × Model × Conc × Backend)

Supports: vLLM JSON, SGLang JSONL, llama.cpp TSV.
Aligned with benchmark_plan.md Section E.1/E.2 tracking tables.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Detailed CSV columns ────────────────────────────────────────
FIELDNAMES = [
    "model", "backend", "phase", "concurrency", "file",
    "successful_requests", "failed_requests", "duration_s",
    "req_throughput", "output_tok_s", "total_tok_s",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "total_input_tokens", "total_generated_tokens",
]

BACKENDS = ["vllm", "llamacpp", "sglang"]


# ── Path detection helpers ──────────────────────────────────────

def detect_backend(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if "vllm" in p:
            return "vllm"
        if "sglang" in p:
            return "sglang"
        if "llamacpp" in p or "llama" in p:
            return "llamacpp"
        if "litellm" in p:
            return "litellm"
    return "unknown"


def detect_phase(path: Path) -> str:
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("p1"):
            return "P1 Light"
        if lower.startswith("p2"):
            return "P2 Medium"
        if lower.startswith("p3"):
            return "P3 Heavy"
    return "unknown"


def detect_run_dir(path: Path) -> str:
    """Find the 'run-N' directory name from any descendant path."""
    for part in path.parts:
        if part.startswith("run-"):
            return part
    return "unknown"


def extract_concurrency(filename: str) -> int:
    match = re.search(r"conc(\d+)", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"concurrent_(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


# ── Model name resolution ───────────────────────────────────────

def build_model_map(results_dir: Path) -> dict[str, str]:
    """Parse bench.log to build {run-N: model_display_name} mapping.

    Looks for lines like:
      [HH:MM:SS] Starting llama.cpp: model=models/gguf/qwen2.5-7b/...
      [HH:MM:SS] Starting vLLM: model=./models/hf/qwen2.5-0.6b ...
    And matches them to the preceding 'Run N:' header.
    """
    model_map = {}
    log_files = list(results_dir.glob("bench.log")) + list(results_dir.glob("bench_models_*.log"))
    for log_file in log_files:
        try:
            lines = log_file.read_text().splitlines()
        except OSError:
            continue
        current_run = None
        for line in lines:
            run_match = re.search(r"Run (\d+):", line)
            if run_match:
                current_run = f"run-{run_match.group(1)}"
                continue
            if current_run and ("Starting" in line):
                model_path = re.search(r"model=(\S+)", line)
                if model_path:
                    model_map[current_run] = model_path_to_name(model_path.group(1))
    return model_map


def model_path_to_name(path_str: str) -> str:
    """Convert model path to human-readable name.

    Examples:
      models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -> Qwen2.5-7B Q4_K_M
      ./models/hf/qwen2.5-0.6b -> Qwen2.5-0.6B
      models/gguf/qwen2.5-32b/qwen2.5-32b-instruct-q4_k_m.gguf -> Qwen2.5-32B Q4_K_M
    """
    p = Path(path_str)
    name = p.name if p.suffix else p.name

    # Extract model family + size
    model_match = re.search(r"(qwen[\d.]+-(\d+\.?\d*b))", path_str, re.IGNORECASE)
    if not model_match:
        model_match = re.search(r"(llama[\d.]+-(\d+\.?\d*b))", path_str, re.IGNORECASE)
    if not model_match:
        return p.stem

    full = model_match.group(1)  # e.g. qwen2.5-7b
    size = model_match.group(2).upper()  # e.g. 7B

    # Detect quantization
    quant = ""
    if "q4_k_m" in path_str.lower():
        quant = "Q4_K_M"
    elif "q8_0" in path_str.lower():
        quant = "Q8_0"
    elif "awq" in path_str.lower():
        quant = "AWQ"
    elif "q5" in path_str.lower():
        quant = "Q5"

    family = full.split("-")[0].capitalize()  # Qwen2.5
    return f"{family}-{size} {quant}".strip()


# ── Metric parsers ──────────────────────────────────────────────

def empty_metrics() -> dict:
    return {k: 0 for k in FIELDNAMES if k not in ("model", "backend", "phase", "concurrency", "file")}


def parse_vllm_json(data: dict) -> dict:
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
    rows = []

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
        avg_latency_ms = float(row_data.get("avg_latency_ms", 0))

        wall_s = wall_ms / 1000 if wall_ms else 0

        m = empty_metrics()
        m["successful_requests"] = completed
        m["failed_requests"] = max(0, conc - completed)
        m["duration_s"] = round(wall_s, 2)
        m["req_throughput"] = round(completed / wall_s, 3) if wall_s else 0
        m["output_tok_s"] = round(agg_tps, 2)
        m["total_tok_s"] = round(agg_tps, 2)
        m["mean_ttft_ms"] = round(avg_latency_ms, 2)
        m["total_input_tokens"] = total_tokens
        m["total_generated_tokens"] = total_tokens

        rows.append({"model": model, "backend": "llamacpp", "phase": phase,
                      "concurrency": conc, "file": tsv_file.name, **m})

    single_file = tsv_dir / "single_user.tsv"
    if single_file.exists():
        try:
            lines = single_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                header = lines[0].split("\t")
                total_completion = total_prompt = decode_sum = decode_count = 0
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
                rows.append({"model": model, "backend": "llamacpp", "phase": phase,
                              "concurrency": 1, "file": "single_user.tsv", **m})
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {single_file}: {e}", file=sys.stderr)

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
                    rows.append({"model": model, "backend": "llamacpp", "phase": phase,
                                  "concurrency": 0, "file": f"long_ctx{ctx}.tsv", **m})
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {long_file}: {e}", file=sys.stderr)

    # ── Parse stress_summary.tsv (incremental stress test) ────────
    stress_file = tsv_dir / "stress_summary.tsv"
    if stress_file.exists():
        try:
            lines = stress_file.read_text().strip().split("\n")
            if len(lines) >= 2:
                header = lines[0].split("\t")
                for line in lines[1:]:
                    vals = dict(zip(header, line.split("\t")))
                    conc = int(vals.get("conc", 0))
                    completed = int(vals.get("completed", 0))
                    total_tokens = int(vals.get("total_tokens", 0))
                    wall_ms = int(vals.get("wall_ms", 0))
                    agg_tps = float(vals.get("agg_tps", 0))
                    status = vals.get("status", "")
                    wall_s = wall_ms / 1000 if wall_ms else 0
                    m = empty_metrics()
                    m["successful_requests"] = completed
                    m["failed_requests"] = 0 if status == "pass" else 1
                    m["duration_s"] = round(wall_s, 2)
                    m["req_throughput"] = round(completed / wall_s, 3) if wall_s else 0
                    m["output_tok_s"] = round(agg_tps, 2)
                    m["total_tok_s"] = round(agg_tps, 2)
                    m["total_input_tokens"] = total_tokens
                    m["total_generated_tokens"] = total_tokens
                    rows.append({"model": model, "backend": "llamacpp", "phase": phase,
                                  "concurrency": conc, "file": f"stress_conc{conc}.tsv", **m})
        except (OSError, ValueError) as e:
            print(f"  WARN: Skipping {stress_file}: {e}", file=sys.stderr)

    return rows


# ── Summary report (pivoted comparison table) ───────────────────

def generate_report(rows: list[dict], output_path: Path) -> None:
    """Produce a pivoted TSV: Phase | Model | Conc | vLLM TTFT | llama.cpp TTFT | SGLang TTFT | vLLM tok/s | ..."""
    # Group by (phase, model, concurrency), keeping best result per backend
    # For concurrent tests, prefer the highest concurrency result with all requests passing
    grouped: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["concurrency"] == 0:
            continue  # skip long_context rows
        key = (r["phase"], r["model"], r["concurrency"])
        backend = r["backend"]
        # Keep the entry with most successful requests (handles duplicates)
        existing = grouped[key].get(backend)
        if existing is None or r["successful_requests"] > existing["successful_requests"]:
            grouped[key][backend] = r

    if not grouped:
        return

    # Build report rows
    report_fieldnames = ["Phase", "Model", "Conc."]
    for b in BACKENDS:
        report_fieldnames.append(f"{b} TTFT ms")
        report_fieldnames.append(f"{b} tok/s")
    report_fieldnames.append("Notes")

    report_rows = []
    for (phase, model, conc), backends in sorted(grouped.items(),
                                                     key=lambda x: (x[0][0], x[0][1], x[0][2])):
        row = {"Phase": phase, "Model": model, "Conc.": conc}
        for b in BACKENDS:
            entry = backends.get(b)
            if entry:
                row[f"{b} TTFT ms"] = round(entry["mean_ttft_ms"], 1) if entry["mean_ttft_ms"] else ""
                row[f"{b} tok/s"] = round(entry["output_tok_s"], 1) if entry["output_tok_s"] else ""
            else:
                row[f"{b} TTFT ms"] = ""
                row[f"{b} tok/s"] = ""
        # Notes
        notes = []
        if conc == 1:
            notes.append("Baseline")
        for b, entry in backends.items():
            if entry["failed_requests"] > 0:
                notes.append(f"{b}:{entry['failed_requests']} failed")
        row["Notes"] = "; ".join(notes)
        report_rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)

    # ── Print formatted table ────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  BENCHMARK COMPARISON REPORT")
    print(f"{'='*100}\n")

    # Column widths
    widths = {}
    for col in report_fieldnames:
        widths[col] = max(len(col), max((len(str(r.get(col, ""))) for r in report_rows), default=0))

    # Header
    header_line = "  ".join(str(col).ljust(widths[col]) for col in report_fieldnames)
    print(header_line)
    print("  ".join("─" * widths[col] for col in report_fieldnames))

    # Rows
    for r in report_rows:
        line = "  ".join(str(r.get(col, "")).ljust(widths[col]) for col in report_fieldnames)
        print(line)

    print()
    print(f"  Report saved: {output_path}")
    print()


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "-d", type=Path, default=Path("results"),
                        help="Root results directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output CSV path (default: <results-dir>/benchmark_summary.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_csv = args.output or results_dir / "benchmark_summary.csv"
    output_report = results_dir / "benchmark_report.tsv"

    # Use session folder name as prefix for output files
    session_name = results_dir.name  # e.g. 20260611_070504_benchmark
    prefix = session_name.replace("_benchmark", "")  # e.g. 20260611_070504
    if not args.output:
        output_csv = results_dir / f"{prefix}_summary.csv"
    output_report = results_dir / f"{prefix}_report.tsv"

    # Build model name map from bench.log
    model_map = build_model_map(results_dir)

    rows = []

    # ── 1. vLLM JSON files ───────────────────────────────────────
    for json_file in sorted(results_dir.rglob("*.json")):
        if "benchmark_summary" in json_file.name or "benchmark_report" in json_file.name:
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: Skipping {json_file}: {e}", file=sys.stderr)
            continue

        backend = detect_backend(json_file)
        phase = detect_phase(json_file)
        run_name = detect_run_dir(json_file)
        model = model_map.get(run_name, run_name)
        conc = extract_concurrency(json_file.name)

        metrics = parse_vllm_json(data)
        rows.append({"model": model, "backend": backend, "phase": phase,
                      "concurrency": conc, "file": json_file.name, **metrics})

    # ── 2. SGLang JSONL files ────────────────────────────────────
    for jsonl_file in sorted(results_dir.rglob("*.jsonl")):
        backend = detect_backend(jsonl_file)
        phase = detect_phase(jsonl_file)
        run_name = detect_run_dir(jsonl_file)
        model = model_map.get(run_name, run_name)
        conc = extract_concurrency(jsonl_file.name)

        agg = {"succ": 0, "fail": 0, "ttfts": [], "tpots": [], "itls": [], "inp": 0, "out": 0}
        try:
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                result = parse_sglang_jsonl(line)
                if result:
                    agg["succ"] += result["successful_requests"]
                    agg["fail"] += result["failed_requests"]
                    agg["inp"] += result["total_input_tokens"]
                    agg["out"] += result["total_generated_tokens"]
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

        m = empty_metrics()
        m.update({
            "successful_requests": agg["succ"], "failed_requests": agg["fail"],
            "mean_ttft_ms": avg(agg["ttfts"]), "mean_tpot_ms": avg(agg["tpots"]),
            "mean_itl_ms": avg(agg["itls"]),
            "total_input_tokens": agg["inp"], "total_generated_tokens": agg["out"],
        })
        rows.append({"model": model, "backend": backend, "phase": phase,
                      "concurrency": conc, "file": jsonl_file.name, **m})

    # ── 3. llama.cpp TSV files ───────────────────────────────────
    processed_dirs = set()
    for tsv_file in sorted(results_dir.rglob("*.tsv")):
        parent = tsv_file.parent
        if parent in processed_dirs:
            continue
        # Detect llamacpp from path: look for *_run in ancestors
        backend_dir = None
        for ancestor in parent.parents:
            if ancestor.name.endswith("_run"):
                backend_dir = ancestor.name
                break
        if backend_dir is None:
            continue
        processed_dirs.add(parent)
        run_name = detect_run_dir(parent)
        model = model_map.get(run_name, run_name)
        phase = detect_phase(parent)
        tsv_rows = parse_llamacpp_tsv_dir(parent, model, phase)
        rows.extend(tsv_rows)

    if not rows:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(0)

    rows.sort(key=lambda r: (r["backend"], r["model"], r["phase"], r["concurrency"]))

    # Write detailed CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Parsed {len(rows)} results -> {output_csv}")

    # Generate comparison report
    generate_report(rows, output_report)


if __name__ == "__main__":
    main()
