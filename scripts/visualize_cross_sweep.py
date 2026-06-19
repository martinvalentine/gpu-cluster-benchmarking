#!/usr/bin/env python3
"""
Visualize cross-sweep benchmark results — CCU ladder at each prompt size.

Reads cross_sweep_pp{pp}_ccu{ccu}.json files and generates:
  1. cross_sweep_table.md — Markdown table (prompt × CCU → throughput)
  2. ccu_ladder_{pp}.png — Per-prompt CCU saturation curves
  3. ccu_vs_prompt.png — Max CCU vs prompt context
  4. throughput_heatmap.png — Prompt × CCU heatmap (if multiple prompt sizes)

Usage:
    python scripts/visualize_cross_sweep.py <results-dir>
    python scripts/visualize_cross_sweep.py results/vllm/2026-06-17_06h31_qwen2.5-0.5b_vllm/
"""

import json, re, sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("WARN: matplotlib not installed — charts disabled. pip install matplotlib")


def collect_data(results_dir):
    data = defaultdict(dict)
    for f in sorted(Path(results_dir).glob("cross_sweep_*.json")):
        m = re.match(r"cross_sweep_pp(\d+)_ccu(\d+)", f.stem)
        if not m:
            continue
        pp, ccu = int(m.group(1)), int(m.group(2))
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        benchmarks = d.get("benchmarks", [])
        if not benchmarks:
            continue
        b = benchmarks[0]
        tg_meta = b.get("tg_throughput") or {}
        pp_meta = b.get("pp_throughput") or {}
        peak_meta = b.get("peak_throughput") or {}
        ttfr_meta = b.get("ttfr") or {}
        data[pp][ccu] = {
            "tg_mean": tg_meta.get("mean", 0),
            "tg_std": tg_meta.get("std", 0),
            "pp_mean": pp_meta.get("mean", 0),
            "pp_std": pp_meta.get("std", 0),
            "peak_mean": peak_meta.get("mean", 0),
            "ttfr_mean": ttfr_meta.get("mean", 0),
            "est_ppt": b.get("est_ppt", {}).get("mean", 0),
        }
    return data


def generate_table(data, out_path):
    """Generate Markdown table: prompt × CCU → generation tok/s."""
    prompt_sizes = sorted(data.keys())
    all_ccus = sorted({ccu for pp in data for ccu in data[pp]})

    lines = []
    lines.append("# Cross-Sweep Results")
    lines.append("")
    lines.append(f"**Model:** {len(prompt_sizes)} prompt sizes × up to {len(all_ccus)} CCU levels")
    lines.append("")

    # Header
    header = "| CCU |" + "|".join(f" pp={pp} " for pp in prompt_sizes) + "|"
    sep = "|-----|" + "|".join("------|" for _ in prompt_sizes)
    lines.append(header)
    lines.append(sep)

    for ccu in all_ccus:
        row = f"| {ccu} "
        for pp in prompt_sizes:
            entry = data[pp].get(ccu)
            if entry:
                agg = entry['tg_mean']
                per_req = agg / ccu
                row += f"| {agg:.0f} "
            else:
                row += "|  ⛔  " if ccu > min(data[pp].keys(), default=0) else "| — "
        row += "|"
        lines.append(row)

    lines.append("")
    lines.append("### Per-Request Throughput (tok/s ÷ CCU)")
    lines.append("")
    lines.append("| CCU |" + "|".join(f" pp={pp} " for pp in prompt_sizes) + "|")
    lines.append("|-----|" + "|".join("------|" for _ in prompt_sizes))
    for ccu in all_ccus:
        row = f"| {ccu} "
        for pp in prompt_sizes:
            entry = data[pp].get(ccu)
            if entry:
                row += f"| {entry['tg_mean'] / ccu:.1f} "
            else:
                row += "|  ⛔  " if ccu > min(data[pp].keys(), default=0) else "| — "
        row += "|"
        lines.append(row)

    lines.append("")
    lines.append("*Units: generation tokens/sec (tg_throughput mean)*")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Table → {out_path}")


def plot_ccu_ladder(data, out_dir):
    """One chart per prompt size: X=CCU, Y=generation tok/s, shows saturation knee."""
    if plt is None:
        return
    for pp in sorted(data):
        ccus = sorted(data[pp])
        tgs = [data[pp][c]["tg_mean"] for c in ccus]
        peaks = [data[pp][c]["peak_mean"] for c in ccus]
        ttfr = [data[pp][c]["ttfr_mean"] for c in ccus]
        tgs_per_req = [data[pp][c]["tg_mean"] / c for c in ccus]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(ccus, tgs, "b-o", label="Aggregate tok/s", markersize=4)
        ax1.plot(ccus, peaks, "g--s", label="Peak throughput", markersize=4, alpha=0.7)
        ax1.set_ylabel("Aggregate tok/s")
        ax1.set_title(f"CCU Ladder — pp={pp}")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax1r = ax1.twinx()
        ax1r.plot(ccus, tgs_per_req, "orange-o", label="tok/s per request", markersize=4)
        ax1r.set_ylabel("tok/s per request", color="orange")
        ax1r.tick_params(axis="y", labelcolor="orange")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1r, labels1r = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines1r, labels1 + labels1r, loc="upper left", fontsize=8)

        ax2.plot(ccus, ttfr, "r-o", label="TTFR (ms)", markersize=4)
        ax2.set_xlabel("CCU (concurrent users)")
        ax2.set_ylabel("ms")
        ax2.set_title("Time to First Response")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        outf = out_dir / f"ccu_ladder_pp{pp}.png"
        plt.tight_layout()
        plt.savefig(outf, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart → {outf}")


def plot_max_ccu_vs_prompt(data, out_path):
    """Bar/line chart: X = prompt tokens, Y = max CCU achieved."""
    if plt is None:
        return
    prompts = sorted(data)
    max_ccus = [max(data[pp]) for pp in prompts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([str(p) for p in prompts], max_ccus, width=0.4, color="steelblue", edgecolor="navy")
    for bar, val in zip(bars, max_ccus):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Prompt Context (tokens)")
    ax.set_ylabel("Max CCU Before Failure")
    ax.set_title("CCU Capacity vs Prompt Context Length")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out_path}")


def plot_heatmap(data, out_path):
    """Heatmap: X = prompt, Y = CCU, color = generation throughput."""
    if plt is None:
        return
    prompts = sorted(data)
    all_ccus = sorted({c for pp in data for c in data[pp]})
    if len(prompts) < 2:
        return  # Heatmap only makes sense with 2+ prompt sizes

    import numpy as np
    grid = np.full((len(all_ccus), len(prompts)), np.nan)
    for i, ccu in enumerate(all_ccus):
        for j, pp in enumerate(prompts):
            if ccu in data[pp]:
                grid[i, j] = data[pp][ccu]["tg_mean"]

    fig, ax = plt.subplots(figsize=(len(prompts) * 2.5, max(6, len(all_ccus) * 0.3)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([str(p) for p in prompts])
    ax.set_yticks(range(len(all_ccus)))
    ax.set_yticklabels([str(c) for c in all_ccus])
    ax.set_xlabel("Prompt Context (tokens)")
    ax.set_ylabel("CCU")
    ax.set_title("Throughput Heatmap — Generation tok/s")
    plt.colorbar(im, ax=ax, label="tok/s")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out_path}")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        sys.exit(1)

    data = collect_data(results_dir)
    if not data:
        print("No cross_sweep_*.json files found")
        sys.exit(1)

    n_pp = len(data)
    n_rows = sum(len(v) for v in data.values())
    print(f"Found {n_rows} data points across {n_pp} prompt sizes")

    generate_table(data, results_dir / "cross_sweep_table.md")
    plot_ccu_ladder(data, results_dir)
    plot_max_ccu_vs_prompt(data, results_dir / "ccu_vs_prompt.png")
    plot_heatmap(data, results_dir / "throughput_heatmap.png")
    print("Done.")


if __name__ == "__main__":
    main()
