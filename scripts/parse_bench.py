#!/usr/bin/env python3
"""
Unified benchmark parser — auto-detects all result formats and generates reports.

Handles: vLLM JSON, SGLang JSONL, llama.cpp TSV, llama-benchy JSON, cross-sweep JSON.

Outputs:
  1. <prefix>_summary.csv       — detailed per-result CSV
  2. <prefix>_report.tsv         — pivoted comparison table (E.2 format from dev_benchmark_plan.md)
  3. <prefix>_table.md           — Markdown table (cross-sweep compatible)

Usage:
    .venv/bin/python scripts/parse_bench.py results/vllm/<session>/
    .venv/bin/python scripts/parse_bench.py results/  --all            # scan all sessions
    .venv/bin/python scripts/parse_bench.py results/vllm/<session>/ --md-only  # markdown only
"""

import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

FIELDNAMES = [
    "model", "backend", "phase", "concurrency", "pp", "file",
    "successful_requests", "failed_requests", "duration_s",
    "req_throughput", "output_tok_s", "total_tok_s",
    "tok_s_per_req",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "total_input_tokens", "total_generated_tokens",
    "params.server.model", "params.server.model_path", "params.server.endpoint", "params.server.port",
    "params.server.tp_size", "params.server.max_model_len", "params.server.max_num_seqs",
    "params.server.gpu_mem_util", "params.server.n_parallel", "params.server.ctx_size",
    "params.server.batch", "params.server.ubatch", "params.server.threads",
    "params.server.flash_attn", "params.server.cache_key", "params.server.cache_val",
    "params.server.cache_prompt", "params.server.chunked_prefill", "params.server.prefix_caching",
    "params.server.max_batched_tokens", "params.server.swap_space_gb",
    "params.server.quantization", "params.server.dtype", "params.server.block_size",
    "params.server.attention_backend", "params.server.radix_cache", "params.server.torch_compile",
    "params.server.trust_remote_code", "params.server.max_total_tokens", "params.server.chunked_prefill_size",
    "params.hardware.hostname", "params.hardware.gpu_name", "params.hardware.gpu_count",
    "params.hardware.gpu_vram_mib", "params.hardware.driver_version", "params.hardware.cuda_version",
    "params.hardware.cpu_name", "params.hardware.cpu_cores", "params.hardware.memory_gb",
    "params.system.timestamp", "params.system.git_commit", "params.system.git_branch",
    "params.system.docker_image", "params.system.server_version",
]

BACKENDS = ["vllm", "llamacpp", "sglang"]


# ═══════════════════════════════════════════════════════════════════
# File detection
# ═══════════════════════════════════════════════════════════════════

def detect_backend(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if "vllm" in p: return "vllm"
        if "sglang" in p: return "sglang"
        if "llamacpp" in p or "llama" in p: return "llamacpp"
        if "litellm" in p: return "litellm"
    return "unknown"


def detect_phase(path: Path) -> str:
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("p1"): return "P1 Light"
        if lower.startswith("p2"): return "P2 Medium"
        if lower.startswith("p3"): return "P3 Heavy"
    return "unknown"


def detect_run_dir(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run-"): return part
    return "unknown"


def extract_concurrency(filename: str) -> int:
    m = re.search(r"conc(\d+)", filename)
    if m: return int(m.group(1))
    m = re.search(r"_ccu(\d+)", filename)
    if m: return int(m.group(1))
    return 0


def extract_prompt_size(filename: str) -> int:
    m = re.search(r"_pp(\d+)", filename)
    if m: return int(m.group(1))
    return 0


def is_llama_benchy_file(path: Path) -> bool:
    name = path.name
    return (name == "ccu_sweep.json"
            or name == "prompt_sweep.json"
            or name.startswith("prompt_sweep_pp")
            or name.startswith("cross_sweep_pp"))


# ═══════════════════════════════════════════════════════════════════
# Metric parsers
# ═══════════════════════════════════════════════════════════════════

def empty_metrics() -> dict:
    return {k: 0 for k in FIELDNAMES
            if k not in ("model", "backend", "phase", "concurrency", "pp", "file")}


def _flatten_params(params: dict | None) -> dict:
    """Flatten params.{server,hardware,system}.* into params.* top-level keys.

    Returns an empty dict if params is None or empty, so old sweep JSONs without
    a 'params' field parse to empty param columns (not an error).
    """
    out: dict = {}
    if not params:
        return out
    for group in ("server", "hardware", "system"):
        for k, v in (params.get(group) or {}).items():
            out[f"params.{group}.{k}"] = v
    return out


def _load_params_from_dir(dir_path) -> "dict | None":
    """Look for a params.json file in dir_path and return its contents.

    Returns None if the file doesn't exist or is malformed. This is the
    resolution strategy for formats that don't embed params in the result
    file itself (sglang JSONL, llamacpp TSV).
    """
    params_file = dir_path / "params.json"
    if not params_file.exists():
        return None
    try:
        return json.loads(params_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def parse_vllm_json(data: dict, params: dict | None = None) -> dict:
    result = {
        "successful_requests": data.get("completed", data.get("successful_requests", 0)),
        "failed_requests": data.get("failed_requests", 0),
        "duration_s": round(data.get("duration", 0), 2),
        "req_throughput": round(data.get("request_throughput", 0), 3),
        "output_tok_s": round(data.get("output_throughput", 0), 2),
        "total_tok_s": round(data.get("total_token_throughput", 0), 2),
        "tok_s_per_req": 0,
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
    result.update(_flatten_params(params))
    return result


def parse_llama_benchy_json(data: dict, filename: str = "", params: dict | None = None) -> list[dict]:
    results = data.get("results", [])
    benchmarks = data.get("benchmarks", [])
    use_benchmarks = bool(benchmarks)
    items = benchmarks if use_benchmarks else results
    model = data.get("model", "unknown")
    rows = []

    for r in items:
        if use_benchmarks:
            conc = r.get("concurrency", 1)
            pp = r.get("prompt_size", 0)
            tg_meta = r.get("tg_throughput") or {}
            gen_throughput = tg_meta.get("mean", 0)
            total_throughput = gen_throughput
            ttfr_meta = r.get("ttfr") or {}
            ttft = ttfr_meta.get("mean", 0)
            latency = 0
        else:
            conc = r.get("concurrency", 1)
            pp = r.get("pp", 0)
            gen_throughput = r.get("generation_throughput", 0)
            total_throughput = r.get("total_throughput", 0)
            ttft = r.get("ttft", 0)
            latency = r.get("latency", 0)

        m = empty_metrics()
        m.update({
            "output_tok_s": round(gen_throughput, 2),
            "total_tok_s": round(total_throughput, 2),
            "tok_s_per_req": round(gen_throughput / conc, 2) if conc > 0 and gen_throughput else 0,
            "mean_ttft_ms": round(ttft, 2) if use_benchmarks else round(ttft * 1000, 2) if ttft else 0,
            "duration_s": round(latency, 2) if latency else 0,
        })
        flat_params = _flatten_params(params)
        rows.append({
            "model": model, "backend": "llamacpp",
            "phase": f"pp={pp}" if use_benchmarks else "llama-benchy",
            "pp": pp, "concurrency": conc,
            "file": filename or f"pp{pp}_conc{conc}",
            **m, **flat_params,
        })
    return rows


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
            "tok_s_per_req": 0,
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
            if len(lines) < 2: continue
            header, values = lines[0].split("\t"), lines[1].split("\t")
            row_data = dict(zip(header, values))
        except (OSError, ValueError) as e:
            print(f"  WARN: {tsv_file}: {e}", file=sys.stderr); continue

        completed = int(row_data.get("completed", 0))
        total_tokens = int(row_data.get("total_tokens", 0))
        wall_ms = int(row_data.get("wall_ms", 0))
        agg_tps = float(row_data.get("agg_tps", 0))
        wall_s = wall_ms / 1000 if wall_ms else 0

        m = empty_metrics()
        m.update({
            "successful_requests": completed,
            "failed_requests": max(0, conc - completed),
            "duration_s": round(wall_s, 2),
            "req_throughput": round(completed / wall_s, 3) if wall_s else 0,
            "output_tok_s": round(agg_tps, 2),
            "total_tok_s": round(agg_tps, 2),
            "tok_s_per_req": round(agg_tps / conc, 2) if conc and agg_tps else 0,
            "total_input_tokens": total_tokens,
            "total_generated_tokens": total_tokens,
        })
        # Load params from sibling params.json (TSV doesn't embed params)
        tsv_params = _load_params_from_dir(tsv_file.parent)
        tsv_flat = _flatten_params(tsv_params)
        rows.append({"model": model, "backend": "llamacpp", "phase": phase, "pp": 0,
                      "concurrency": conc, "file": tsv_file.name, **m, **tsv_flat})
    return rows


# ═══════════════════════════════════════════════════════════════════
# Model name resolution
# ═══════════════════════════════════════════════════════════════════

def build_model_map(results_dir: Path) -> dict[str, str]:
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
            if current_run and "Starting" in line:
                model_path = re.search(r"model=(\S+)", line)
                if model_path:
                    model_map[current_run] = model_path_to_name(model_path.group(1))
    return model_map


def model_path_to_name(path_str: str) -> str:
    p = Path(path_str)
    name = p.name if p.suffix else p.name
    model_match = re.search(r"(qwen[\d.]+-(\d+\.?\d*b))", path_str, re.IGNORECASE)
    if not model_match:
        model_match = re.search(r"(llama[\d.]+-(\d+\.?\d*b))", path_str, re.IGNORECASE)
    if not model_match:
        return p.stem
    full = model_match.group(1)
    size = model_match.group(2).upper()
    quant = ""
    if "q4_k_m" in path_str.lower(): quant = "Q4_K_M"
    elif "q8_0" in path_str.lower(): quant = "Q8_0"
    elif "awq" in path_str.lower(): quant = "AWQ"
    family = full.split("-")[0].capitalize()
    return f"{family}-{size} {quant}".strip()


# ═══════════════════════════════════════════════════════════════════
# Collection
# ═══════════════════════════════════════════════════════════════════

def collect_results(results_dir: Path) -> list[dict]:
    model_map = build_model_map(results_dir)
    rows = []

    # ── JSON files (vLLM + llama-benchy) ──────────────────────
    for json_file in sorted(results_dir.rglob("*.json")):
        if any(x in json_file.name for x in ("benchmark_summary", "benchmark_report")):
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: {json_file}: {e}", file=sys.stderr); continue

        if is_llama_benchy_file(json_file):
            benchy_rows = parse_llama_benchy_json(data, json_file.name, params=data.get("params"))
            for r in benchy_rows:
                r["file"] = json_file.name
                rows.append(r)
            continue

        backend = detect_backend(json_file)
        phase = detect_phase(json_file)
        run_name = detect_run_dir(json_file)
        model = model_map.get(run_name, run_name)
        conc = extract_concurrency(json_file.name)

        metrics = parse_vllm_json(data, params=data.get("params"))
        metrics["tok_s_per_req"] = round(metrics["output_tok_s"] / conc, 2) if conc and metrics["output_tok_s"] else 0
        rows.append({"model": model, "backend": backend, "phase": phase, "pp": 0,
                      "concurrency": conc, "file": json_file.name, **metrics})

    # ── JSONL files (SGLang) ─────────────────────────────────
    for jsonl_file in sorted(results_dir.rglob("*.jsonl")):
        backend = detect_backend(jsonl_file)
        phase = detect_phase(jsonl_file)
        run_name = detect_run_dir(jsonl_file)
        model = model_map.get(run_name, run_name)
        conc = extract_concurrency(jsonl_file.name)

        agg = {"succ": 0, "fail": 0, "ttfts": [], "tpots": [], "itls": [], "inp": 0, "out": 0}
        try:
            for line in jsonl_file.read_text().strip().split("\n"):
                if not line.strip(): continue
                result = parse_sglang_jsonl(line)
                if result:
                    agg["succ"] += result["successful_requests"]
                    agg["fail"] += result["failed_requests"]
                    agg["inp"] += result["total_input_tokens"]
                    agg["out"] += result["total_generated_tokens"]
                    if result["mean_ttft_ms"] > 0: agg["ttfts"].append(result["mean_ttft_ms"])
                    if result["mean_tpot_ms"] > 0: agg["tpots"].append(result["mean_tpot_ms"])
                    if result["mean_itl_ms"] > 0: agg["itls"].append(result["mean_itl_ms"])
        except OSError as e:
            print(f"  WARN: {jsonl_file}: {e}", file=sys.stderr); continue

        # Load params from sibling params.json (sglang JSONL doesn't embed params in the result file itself)
        sglang_params = _load_params_from_dir(jsonl_file.parent)
        flat_params = _flatten_params(sglang_params)

        def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0

        m = empty_metrics()
        m.update({
            "successful_requests": agg["succ"], "failed_requests": agg["fail"],
            "mean_ttft_ms": avg(agg["ttfts"]), "mean_tpot_ms": avg(agg["tpots"]),
            "mean_itl_ms": avg(agg["itls"]),
            "total_input_tokens": agg["inp"], "total_generated_tokens": agg["out"],
        })
        rows.append({"model": model, "backend": backend, "phase": phase, "pp": 0,
                      "concurrency": conc, "file": jsonl_file.name, **m, **flat_params})

    # ── TSV files (llama.cpp) ────────────────────────────────
    processed_dirs = set()
    for tsv_file in sorted(results_dir.rglob("*.tsv")):
        parent = tsv_file.parent
        if parent in processed_dirs: continue
        backend_dir = None
        for ancestor in parent.parents:
            if ancestor.name.endswith("_run"):
                backend_dir = ancestor.name; break
        if backend_dir is None: continue
        processed_dirs.add(parent)
        run_name = detect_run_dir(parent)
        model = model_map.get(run_name, run_name)
        phase = detect_phase(parent)
        tsv_rows = parse_llamacpp_tsv_dir(parent, model, phase)
        rows.extend(tsv_rows)

    return rows


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(rows: list[dict], output_path: Path):
    """Pivoted TSV: Phase | Model | Conc. | vLLM TTFT | vLLM tok/s | ..."""
    grouped: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["concurrency"] == 0: continue
        key = (r["phase"], r["model"], r["concurrency"])
        backend = r["backend"]
        existing = grouped[key].get(backend)
        if existing is None or r["successful_requests"] > existing["successful_requests"]:
            grouped[key][backend] = r

    if not grouped: return

    report_fieldnames = ["Phase", "Model", "Conc."]
    for b in BACKENDS:
        report_fieldnames.append(f"{b} TTFT ms")
        report_fieldnames.append(f"{b} tok/s")
        report_fieldnames.append(f"{b} tok/s/req")
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
                row[f"{b} tok/s/req"] = round(entry["output_tok_s"] / conc, 1) if entry["output_tok_s"] and conc > 0 else ""
            else:
                row[f"{b} TTFT ms"] = row[f"{b} tok/s"] = row[f"{b} tok/s/req"] = ""
        notes = []
        if conc == 1: notes.append("Baseline")
        for b, entry in backends.items():
            if entry["failed_requests"] > 0: notes.append(f"{b}:{entry['failed_requests']} failed")
        row["Notes"] = "; ".join(notes)
        report_rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)

    widths = {}
    for col in report_fieldnames:
        widths[col] = max(len(col), max((len(str(r.get(col, ""))) for r in report_rows), default=0))

    print(f"\n{'='*100}")
    print(f"  BENCHMARK COMPARISON REPORT")
    print(f"{'='*100}\n")
    header_line = "  ".join(str(col).ljust(widths[col]) for col in report_fieldnames)
    print(header_line)
    print("  ".join("─" * widths[col] for col in report_fieldnames))
    for r in report_rows:
        line = "  ".join(str(r.get(col, "")).ljust(widths[col]) for col in report_fieldnames)
        print(line)
    print(f"\n  Report saved: {output_path}\n")


def generate_markdown(rows: list[dict], output_path: Path):
    """Markdown table for cross-sweep / aggregate results."""
    lines = ["# Benchmark Results", ""]
    lines.append(f"**Source:** `{output_path.parent}`")
    lines.append(f"**Data points:** {len(rows)}")
    lines.append("")

    # Pivot by phase × concurrency
    phases = sorted(set(r["phase"] for r in rows))
    all_ccus = sorted({r["concurrency"] for r in rows if r["concurrency"] > 0})

    # Aggregate throughput table
    lines.append("## Aggregate Throughput (tok/s)")
    lines.append("")
    header = "| Phase |" + "|".join(f" ccu={c} " for c in all_ccus) + "|"
    sep = "|-------|" + "|".join("-------" for _ in all_ccus) + "|"
    lines.append(header); lines.append(sep)
    for phase in phases:
        row = f"| {phase} "
        phase_rows = {r["concurrency"]: r for r in rows if r["phase"] == phase}
        for ccu in all_ccus:
            entry = phase_rows.get(ccu)
            if entry and entry["output_tok_s"]:
                row += f"| {entry['output_tok_s']:.0f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    lines.append("")

    # Per-request throughput
    lines.append("## Per-Request Throughput (tok/s/req)")
    lines.append("")
    lines.append("| Phase |" + "|".join(f" ccu={c} " for c in all_ccus) + "|")
    lines.append("|-------|" + "|".join("-------" for _ in all_ccus) + "|")
    for phase in phases:
        row = f"| {phase} "
        phase_rows = {r["concurrency"]: r for r in rows if r["phase"] == phase}
        for ccu in all_ccus:
            entry = phase_rows.get(ccu)
            if entry and entry["tok_s_per_req"]:
                row += f"| {entry['tok_s_per_req']:.1f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    lines.append("")

    # TTFT table
    lines.append("## TTFT (ms)")
    lines.append("")
    lines.append("| Phase |" + "|".join(f" ccu={c} " for c in all_ccus) + "|")
    lines.append("|-------|" + "|".join("-------" for _ in all_ccus) + "|")
    for phase in phases:
        row = f"| {phase} "
        phase_rows = {r["concurrency"]: r for r in rows if r["phase"] == phase}
        for ccu in all_ccus:
            entry = phase_rows.get(ccu)
            if entry and entry["mean_ttft_ms"]:
                row += f"| {entry['mean_ttft_ms']:.0f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown → {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Results directory or root")
    parser.add_argument("--all", action="store_true", help="Scan all session dirs under input")
    parser.add_argument("--md-only", action="store_true", help="Only generate markdown table")
    parser.add_argument("--csv-only", action="store_true", help="Only generate CSV (no report)")
    parser.add_argument("--hide-params", action="store_true",
                        help="Suppress the params.* columns in the CSV/TSV output")
    parser.add_argument("--only-params", action="store_true",
                        help="Show only the params.* columns (drop the metric columns)")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        sys.exit(1)

    if args.all:
        dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    else:
        dirs = [input_dir]

    for d in dirs:
        if not d.is_dir(): continue
        print(f"\n── {d.name} ──")
        rows = collect_results(d)
        if not rows:
            print("  No results found"); continue
        rows.sort(key=lambda r: (r["backend"], r["model"], r["phase"], r["concurrency"]))

        session_name = d.name
        prefix = session_name.replace("_benchmark", "")
        csv_path = d / f"{prefix}_summary.csv"
        report_path = d / f"{prefix}_report.tsv"
        md_path = d / f"{prefix}_table.md"

        if not args.md_only:
            if args.only_params:
                fieldnames = [f for f in FIELDNAMES if f.startswith("params.")]
            else:
                fieldnames = list(FIELDNAMES)
                if args.hide_params:
                    fieldnames = [f for f in fieldnames if not f.startswith("params.")]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  CSV → {csv_path}  ({len(rows)} rows)")

        if not args.csv_only:
            generate_report(rows, report_path)
            generate_markdown(rows, md_path)


if __name__ == "__main__":
    main()
