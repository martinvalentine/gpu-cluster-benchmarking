#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

BASE_URL="${LLAMA_BENCH_URL:-http://localhost:8001/v1}"
MODEL="${LLAMA_BENCH_MODEL:-Qwen/Qwen3-35B-A3B}"
RESULTS_DIR="${LLAMA_RESULTS_DIR:-${PROJECT_ROOT}/results/llamacpp}"
FORMAT="${LLAMA_BENCH_FORMAT:-json}"
RUNS="${LLAMA_BENCH_RUNS:-3}"

PP="${LLAMA_BENCH_PP:-128 256 512}"
TG="${LLAMA_BENCH_TG:-64 128 256}"
DEPTH="${LLAMA_BENCH_DEPTH:-0 512 2048}"
CONCURRENCY="${LLAMA_BENCH_CONCURRENCY:-1 4 8}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark llama.cpp server using llama-benchy.
Measures PP (prefill), TG (decode), TTFR at various context depths and concurrency.

OPTIONS:
  -u, --url URL          Base URL (default: http://localhost:8001/v1)
  -m, --model NAME       HF model name for tokenizer (default: Qwen/Qwen3-35B-A3B)
  -o, --output DIR       Results directory (default: ./results/llamacpp)
  -f, --format FMT       Output format: json, csv, md (default: json)
  -r, --runs N           Runs per test (default: 3)
  -p, --phase PHASE      Run specific phase: p1, p2, p3, all (default: all)
  -h, --help             Show this help

ENV OVERRIDES:
  LLAMA_BENCH_URL, LLAMA_BENCH_MODEL, LLAMA_RESULTS_DIR,
  LLAMA_BENCH_FORMAT, LLAMA_BENCH_RUNS, LLAMA_BENCH_PP,
  LLAMA_BENCH_TG, LLAMA_BENCH_DEPTH, LLAMA_BENCH_CONCURRENCY

PHASES (from benchmark_plan.md D.3):
  P1 Light   --pp 128        --tg 64         --depth 0         --concurrency 1
  P2 Medium  --pp 128 256    --tg 64 128     --depth 0 512     --concurrency 1 4
  P3 Heavy   --pp 128 256 512 --tg 64 128 256 --depth 0 512 2048 --concurrency 1 4 8

EXAMPLES:
  $(basename "$0")                              # Run all phases
  $(basename "$0") -p p1                        # Light load only
  $(basename "$0") -p p3 -r 5                   # Heavy, 5 runs each
  $(basename "$0") -u http://gpu-pod:8001/v1    # Remote server
EOF
    exit 0
}

PHASE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)       BASE_URL="$2"; shift 2 ;;
        -m|--model)     MODEL="$2"; shift 2 ;;
        -o|--output)    RESULTS_DIR="$2"; shift 2 ;;
        -f|--format)    FORMAT="$2"; shift 2 ;;
        -r|--runs)      RUNS="$2"; shift 2 ;;
        -p|--phase)     PHASE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown: $1" >&2; usage ;;
    esac
done

mkdir -p "$RESULTS_DIR"

BENCHY="uv run llama-benchy"

echo "=== llama.cpp Benchmark (llama-benchy) ==="
echo "  URL:      $BASE_URL"
echo "  Model:    ${MODEL}"
echo "  Results:  $RESULTS_DIR"
echo "  Format:   $FORMAT"
echo "  Runs:     $RUNS"
echo ""

run_phase() {
    local phase_name=$1
    shift
    local args=("$@")
    local outfile="${RESULTS_DIR}/${phase_name}.${FORMAT}"

    echo "--- Phase: ${phase_name} ---"
    echo "  Args: ${args[*]}"
    echo "  Output: $outfile"
    echo ""

    $BENCHY \
        --base-url "$BASE_URL" \
        ${MODEL:+--model "$MODEL"} \
        --runs "$RUNS" \
        --format "$FORMAT" \
        --save-result "$outfile" \
        "${args[@]}"

    echo ""
    echo "  Saved: $outfile"
    echo ""
}

if [[ "$PHASE" == "all" || "$PHASE" == "p1" ]]; then
    run_phase "p1_light" \
        --pp 128 \
        --tg 64 \
        --depth 0 \
        --concurrency 1
fi

if [[ "$PHASE" == "all" || "$PHASE" == "p2" ]]; then
    run_phase "p2_medium" \
        --pp 128 256 \
        --tg 64 128 \
        --depth 0 512 \
        --concurrency 1 4
fi

if [[ "$PHASE" == "all" || "$PHASE" == "p3" ]]; then
    run_phase "p3_heavy" \
        --pp 128 256 512 \
        --tg 64 128 256 \
        --depth 0 512 2048 \
        --concurrency 1 4 8
fi

echo "=== All benchmarks complete ==="
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
