#!/usr/bin/env python3
"""Generate a formatted benchmark report from results.

Full formatted report (terminal + CSV + markdown). For lightweight CSV
aggregation used by the pipeline, see parse_bench.py.

Reads JSON/TSV files from results/ and produces:
  - Terminal output (colored)
  - Markdown report (results/report.md)
  - CSV summary (results/benchmark_summary.csv)

Usage:
    python scripts/report.py                          # Auto-detect results
    python scripts/report.py --results-dir results/   # Custom dir
    python scripts/report.py --markdown               # Only generate MD
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median


def format_key_params(params):
    """Render a Key Parameters markdown section from a params dict.

    Returns an empty string if params is None or empty.
    """
    if not params:
        return ""
    s = (params.get("server") or {})
    h = (params.get("hardware") or {})
    sys_ = (params.get("system") or {})

    def _v(x, default="—"):
        return default if x in (None, "", "N/A") else x

    rows = [
        ("Model",             _v(s.get("model"))),
        ("Endpoint",          _v(s.get("endpoint"))),
        ("TP Size",           _v(s.get("tp_size"))),
        ("Max Model Len",     _v(s.get("max_model_len"))),
        ("Max Num Seqs",      _v(s.get("max_num_seqs"))),
        ("GPU Mem Util",      _v(s.get("gpu_mem_util"))),
        ("Flash Attn",        _v(s.get("flash_attn"))),
        ("KV Cache",          f"K={_v(s.get('cache_key'))} V={_v(s.get('cache_val'))}"),
        ("Prefix Caching",    _v(s.get("prefix_caching"))),
        ("Chunked Prefill",   _v(s.get("chunked_prefill"))),
        ("Attention Backend", _v(s.get("attention_backend"))),
        ("Quantization",      _v(s.get("quantization"))),
        ("Hardware",
            f"{_v(h.get('gpu_count'), '0')}× {_v(h.get('gpu_name'))} "
            f"({_v(h.get('gpu_vram_mib'), '0')} MiB)"),
        ("CUDA / Driver",     f"{_v(h.get('cuda_version'))} / {_v(h.get('driver_version'))}"),
        ("Git Commit",        f"{_v(sys_.get('git_commit'))} ({_v(sys_.get('git_branch'))})"),
        ("Docker Image",      _v(sys_.get("docker_image"))),
        ("Server Version",    _v(sys_.get("server_version"))),
        ("Captured At",       _v(sys_.get("timestamp"))),
    ]

    lines = ["## Key Parameters", "", "| Field | Value |", "|-------|-------|"]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    lines.append("")
    return "\n".join(lines)


def format_params_short(params):
    """Short params summary for per-row table cells: model | port | ctk/ctv | gpu | git."""
    if not params:
        return "—"
    s = (params.get("server") or {})
    h = (params.get("hardware") or {})
    sys_ = (params.get("system") or {})

    def _v(x, default="—"):
        return default if x in (None, "", "N/A") else x

    return (
        f"{_v(s.get('model'))} | "
        f":{_v(s.get('port'))} | "
        f"{_v(s.get('cache_key'))}/{_v(s.get('cache_val'))} | "
        f"{_v(h.get('gpu_name'))} | "
        f"{_v(sys_.get('git_commit'))}"
    )


def p99(arr):
    if not arr:
        return 0.0
    s = sorted(arr)
    return s[max(int(0.99 * len(s)) - 1, 0)]


def parse_vllm_json(path):
    try:
        d = json.loads(path.read_text())
        return {
            "framework": "vllm",
            "file": path.name,
            "successful": d.get("completed", d.get("successful_requests", 0)),
            "failed": d.get("failed_requests", 0),
            "duration_s": round(d.get("duration", 0), 2),
            "req_throughput": round(d.get("request_throughput", 0), 3),
            "output_tok_s": round(d.get("output_throughput", 0), 2),
            "total_tok_s": round(d.get("total_token_throughput", 0), 2),
            "mean_ttft": round(d.get("mean_ttft_ms", 0), 2),
            "median_ttft": round(d.get("median_ttft_ms", 0), 2),
            "p99_ttft": round(d.get("p99_ttft_ms", 0), 2),
            "mean_tpot": round(d.get("mean_tpot_ms", 0), 2),
            "p99_tpot": round(d.get("p99_tpot_ms", 0), 2),
            "mean_itl": round(d.get("mean_itl_ms", 0), 2),
            "p99_itl": round(d.get("p99_itl_ms", 0), 2),
        }
    except Exception:
        return None


def parse_sglang_jsonl(path):
    ttfts, tpots, itls = [], [], []
    ok = fail = 0
    try:
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            ok += 1
            if d.get("error"):
                fail += 1
                continue
            if d.get("ttft") is not None:
                ttfts.append(d["ttft"] * 1000)
            if d.get("tpot") is not None:
                tpots.append(d["tpot"] * 1000)
            if d.get("itl") is not None:
                itls.append(d["itl"] * 1000)
    except Exception:
        return None

    if ok == 0:
        return None

    return {
        "framework": "sglang",
        "file": path.name,
        "successful": ok,
        "failed": fail,
        "duration_s": 0,
        "req_throughput": 0,
        "output_tok_s": 0,
        "total_tok_s": 0,
        "mean_ttft": round(mean(ttfts), 2) if ttfts else 0,
        "median_ttft": round(median(ttfts), 2) if ttfts else 0,
        "p99_ttft": round(p99(ttfts), 2) if ttfts else 0,
        "mean_tpot": round(mean(tpots), 2) if tpots else 0,
        "p99_tpot": round(p99(tpots), 2) if tpots else 0,
        "mean_itl": round(mean(itls), 2) if itls else 0,
        "p99_itl": round(p99(itls), 2) if itls else 0,
    }


def parse_llamacpp_tsv(path):
    try:
        lines = path.read_text().strip().split("\n")
        if len(lines) < 2:
            return None

        headers = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            vals = line.split("\t")
            rows.append(dict(zip(headers, vals)))

        if "single_user" in path.name:
            ttfts = []
            decode_tps = []
            for r in rows:
                if r.get("ttft_ms", "?") != "?":
                    ttfts.append(float(r["ttft_ms"]))
                if r.get("decode_tps", "?") != "?":
                    decode_tps.append(float(r["decode_tps"]))

            return {
                "framework": "llamacpp",
                "file": path.name,
                "successful": len(rows),
                "failed": 0,
                "duration_s": 0,
                "req_throughput": 0,
                "output_tok_s": round(mean(decode_tps), 2) if decode_tps else 0,
                "total_tok_s": 0,
                "mean_ttft": round(mean(ttfts), 2) if ttfts else 0,
                "median_ttft": round(median(ttfts), 2) if ttfts else 0,
                "p99_ttft": round(p99(ttfts), 2) if ttfts else 0,
                "mean_tpot": 0,
                "p99_tpot": 0,
                "mean_itl": 0,
                "p99_itl": 0,
            }

        elif "concurrent" in path.name:
            agg_tps = []
            for r in rows:
                if r.get("agg_tps", "?") != "?":
                    agg_tps.append(float(r["agg_tps"]))

            return {
                "framework": "llamacpp",
                "file": path.name,
                "successful": sum(int(r.get("completed", 0)) for r in rows),
                "failed": 0,
                "duration_s": 0,
                "req_throughput": 0,
                "output_tok_s": round(max(agg_tps), 2) if agg_tps else 0,
                "total_tok_s": 0,
                "mean_ttft": 0,
                "median_ttft": 0,
                "p99_ttft": 0,
                "mean_tpot": 0,
                "p99_tpot": 0,
                "mean_itl": 0,
                "p99_itl": 0,
            }

        return None
    except Exception:
        return None


def parse_llama_benchy_json(path):
    try:
        data = json.loads(path.read_text())
        results = data.get("results", [])
        if not results:
            return None

        best = max(results, key=lambda r: r.get("total_throughput", 0))
        return {
            "framework": "llama-benchy",
            "file": path.name,
            "successful": len(results),
            "failed": 0,
            "duration_s": round(best.get("latency", 0), 2),
            "req_throughput": 0,
            "output_tok_s": round(best.get("generation_throughput", 0), 2),
            "total_tok_s": round(best.get("total_throughput", 0), 2),
            "mean_ttft": round(best.get("ttft", 0) * 1000, 2),
            "median_ttft": 0,
            "p99_ttft": 0,
            "mean_tpot": 0,
            "p99_tpot": 0,
            "mean_itl": round(best.get("itl", 0) * 1000, 2),
            "p99_itl": 0,
        }
    except Exception:
        return None


def collect_results(results_dir):
    """Collect all benchmark results AND their associated server params.

    For each result file, looks for a sibling `params.json` file (the snapshot
    written by the bench script) and attaches its parsed contents as 'params'.
    """
    results = []

    for f in sorted(results_dir.rglob("*.json")):
        if f.name == "ccu_sweep.json" or f.name.startswith("prompt_sweep_pp"):
            r = parse_llama_benchy_json(f)
        else:
            r = parse_vllm_json(f)
        if r:
            params = _read_sibling_params(f.parent)
            if params:
                r["params"] = params
            results.append(r)

    for f in sorted(results_dir.rglob("*.jsonl")):
        r = parse_sglang_jsonl(f)
        if r:
            params = _read_sibling_params(f.parent)
            if params:
                r["params"] = params
            results.append(r)

    for f in sorted(results_dir.rglob("*.tsv")):
        r = parse_llamacpp_tsv(f)
        if r:
            params = _read_sibling_params(f.parent)
            if params:
                r["params"] = params
            results.append(r)

    return results


def _read_sibling_params(dir_path):
    """Read params.json from dir_path. Returns dict or None."""
    params_file = dir_path / "params.json"
    if not params_file.exists():
        return None
    try:
        return json.loads(params_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def print_table(rows):
    if not rows:
        print("  No results found.")
        return

    header = f"{'Framework':<12} {'File':<40} {'OK':>5} {'TTFT ms':>10} {'TPOT ms':>10} {'tok/s':>10}"
    sep = "-" * len(header)

    print(f"\n  {header}")
    print(f"  {sep}")
    for r in rows:
        print(f"  {r['framework']:<12} {r['file']:<40} {r['successful']:>5} {r['mean_ttft']:>10.1f} {r['mean_tpot']:>10.2f} {r['output_tok_s']:>10.1f}")
    print()


def write_markdown(rows, output_path, include_params=True):
    """Write markdown report. If include_params is True, render a Key Parameters
    section at the top and a Params column in per-row tables.
    """
    if not rows:
        return

    lines = [
        "# Benchmark Report",
        "",
        f"Generated from: `{output_path.parent}`",
        "",
    ]

    # ── Key Parameters section ──
    if include_params:
        first_params = None
        for r in rows:
            if r.get("params"):
                first_params = r["params"]
                break
        key_params_md = format_key_params(first_params)
        if key_params_md:
            lines.append(key_params_md)

    # ── Summary table (with optional Params column) ──
    if include_params and any(r.get("params") for r in rows):
        lines.extend([
            "## Summary",
            "",
            "| Framework | File | Requests | TTFT (ms) | TPOT (ms) | tok/s | Params |",
            "|-----------|------|----------|-----------|-----------|-------|--------|",
        ])
    else:
        lines.extend([
            "## Summary",
            "",
            "| Framework | File | Requests | TTFT (ms) | TPOT (ms) | tok/s |",
            "|-----------|------|----------|-----------|-----------|-------|",
        ])

    for r in rows:
        if include_params and r.get("params"):
            lines.append(
                f"| {r['framework']} | {r['file']} | {r['successful']} | "
                f"{r['mean_ttft']:.1f} | {r['mean_tpot']:.2f} | {r['output_tok_s']:.1f} | "
                f"{format_params_short(r.get('params'))} |"
            )
        else:
            lines.append(
                f"| {r['framework']} | {r['file']} | {r['successful']} | "
                f"{r['mean_ttft']:.1f} | {r['mean_tpot']:.2f} | {r['output_tok_s']:.1f} |"
            )

    # ── Detailed Metrics table (with optional Params column) ──
    if include_params and any(r.get("params") for r in rows):
        lines.extend([
            "",
            "## Detailed Metrics",
            "",
            "| Framework | File | P99 TTFT | P99 TPOT | P99 ITL | Failed | Params |",
            "|-----------|------|----------|----------|---------|--------|--------|",
        ])
    else:
        lines.extend([
            "",
            "## Detailed Metrics",
            "",
            "| Framework | File | P99 TTFT | P99 TPOT | P99 ITL | Failed |",
            "|-----------|------|----------|----------|---------|--------|",
        ])

    for r in rows:
        if include_params and r.get("params"):
            lines.append(
                f"| {r['framework']} | {r['file']} | "
                f"{r['p99_ttft']:.1f} | {r['p99_tpot']:.2f} | {r['p99_itl']:.2f} | {r['failed']} | "
                f"{format_params_short(r.get('params'))} |"
            )
        else:
            lines.append(
                f"| {r['framework']} | {r['file']} | "
                f"{r['p99_ttft']:.1f} | {r['p99_tpot']:.2f} | {r['p99_itl']:.2f} | {r['failed']} |"
            )

    output_path.write_text("\n".join(lines) + "\n")


def write_csv(rows, output_path):
    if not rows:
        return

    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "-d", type=Path, default=Path("results"))
    parser.add_argument("--markdown", action="store_true", help="Only generate markdown report")
    parser.add_argument("--no-params", action="store_true", help="Suppress the Key Parameters section and the per-row params column")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    G, C, B, NC = "\033[0;32m", "\033[0;36m", "\033[1m", "\033[0m"

    print(f"\n  {G}{B}{'='*55}{NC}")
    print(f"  {G}{B}  Benchmark Report{NC}")
    print(f"  {G}{B}{'='*55}{NC}")
    print(f"  {C}Results dir{NC}  {results_dir}")

    rows = collect_results(results_dir)
    print(f"  {C}Files found{NC}  {len(rows)}")
    print()

    if not rows:
        print("  No benchmark results found in results/")
        print("  Run benchmarks first: ./scripts/run-all-benchmarks.sh")
        sys.exit(0)

    print_table(rows)

    csv_path = results_dir / "benchmark_summary.csv"
    write_csv(rows, csv_path)
    print(f"  CSV:  {csv_path}")

    md_path = results_dir / "report.md"
    write_markdown(rows, md_path, include_params=not args.no_params)
    print(f"  MD:   {md_path}")
    print()


if __name__ == "__main__":
    main()
