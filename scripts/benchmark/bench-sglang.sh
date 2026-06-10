#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

BASE_URL="${SGLANG_BENCH_URL:-http://localhost:8002}"
MODEL="${SGLANG_BENCH_MODEL:-models/hf/qwen2.5-0.6b}"
RESULTS_DIR="${SGLANG_RESULTS_DIR:-${PROJECT_ROOT}/results/sglang}"
DATASET="${SGLANG_BENCH_DATASET:-sharegpt}"
DATASET_PATH="${SGLANG_BENCH_DATASET_PATH:-${PROJECT_ROOT}/datasets/sharegpt.json}"
SGLANG_BIN="${SGLANG_BIN:-$(command -v sglang 2>/dev/null || echo "${PROJECT_ROOT}/.venv/bin/python3 -m sglang")}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark SGLang server using sglang.bench_serving (benchmark_plan.md D.2).
Measures TTFT, TPOT, ITL, throughput across phases.

OPTIONS:
  -u, --url URL           SGLang base URL (default: http://localhost:8002)
  -m, --model PATH        Model path (default: models/hf/qwen2.5-0.6b)
  -o, --output DIR        Results directory (default: ./results/sglang)
  -d, --dataset NAME      Dataset: sharegpt, random (default: sharegpt)
  -dp, --dataset-path P   Dataset file path
  -p, --phase PHASE       Run phase: p1, p2, p3, all (default: all)
  -h, --help              Show this help

ENV OVERRIDES:
  SGLANG_BENCH_URL, SGLANG_BENCH_MODEL, SGLANG_RESULTS_DIR,
  SGLANG_BENCH_DATASET, SGLANG_BENCH_DATASET_PATH

PHASES (from benchmark_plan.md D.2):
  P1 Light   8B model    concurrency: 1 32 64 128
  P2 Medium  14B model   concurrency: 1 16 32 64
  P3 Heavy   32B AWQ     concurrency: 1 4 8 16

EXAMPLES:
  $(basename "$0")                                    # All phases
  $(basename "$0") -p p1 -u http://localhost:8002     # P1 only
  $(basename "$0") -p p3 --dataset random             # Random dataset
EOF
    exit 0
}

PHASE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)           BASE_URL="$2"; shift 2 ;;
        -m|--model)         MODEL="$2"; shift 2 ;;
        -o|--output)        RESULTS_DIR="$2"; shift 2 ;;
        -d|--dataset)       DATASET="$2"; shift 2 ;;
        -dp|--dataset-path) DATASET_PATH="$2"; shift 2 ;;
        -p|--phase)         PHASE="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown: $1" >&2; usage ;;
    esac
done

mkdir -p "$RESULTS_DIR"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

row() { printf "  ${CYAN}%-16s${NC} %s\n" "$1" "$2"; }
sep() { echo -e "  ${DIM}$(printf '%.0s─' {1..48})${NC}"; }

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  SGLang Benchmark${NC}${DIM} — sglang.bench_serving${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Base URL"    "$BASE_URL"
row "Model"       "$MODEL"
row "Dataset"     "$DATASET"
row "Phase"       "$PHASE"
row "Results"     "$RESULTS_DIR"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

run_bench() {
    local phase_name=$1
    local model=$2
    local conc=$3
    local num_prompts=$((conc * 5))
    local outfile="${RESULTS_DIR}/${phase_name}_conc${conc}.jsonl"

    echo "[$(date +%H:%M:%S)] SGLang ${phase_name} conc=${conc} num_prompts=${num_prompts}"

    $SGLANG_BIN.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$model" \
        --dataset-name "$DATASET" \
        ${DATASET_PATH:+--dataset-path "$DATASET_PATH"} \
        --num-prompts "$num_prompts" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --output-file "$outfile" \
        2>&1

    echo "  Saved: $outfile"
    echo ""
}

# P1: Light Load — 8B model
if [[ "$PHASE" == "all" || "$PHASE" == "p1" ]]; then
    P1_MODEL="${SGLANG_P1_MODEL:-${PROJECT_ROOT}/models/hf/llama3.1-8b}"
    echo "--- Phase P1: Light Load (8B) ---"
    for CONC in 1 32 64 128; do
        run_bench "p1_light" "$P1_MODEL" "$CONC"
        sleep 5
    done
fi

# P2: Medium Load — 14B model
if [[ "$PHASE" == "all" || "$PHASE" == "p2" ]]; then
    P2_MODEL="${SGLANG_P2_MODEL:-${PROJECT_ROOT}/models/hf/qwen2.5-14b}"
    echo "--- Phase P2: Medium Load (14B) ---"
    for CONC in 1 16 32 64; do
        run_bench "p2_medium" "$P2_MODEL" "$CONC"
        sleep 10
    done
fi

# P3: Heavy Stress — 32B AWQ
if [[ "$PHASE" == "all" || "$PHASE" == "p3" ]]; then
    P3_MODEL="${SGLANG_P3_MODEL:-$MODEL}"
    echo "--- Phase P3: Heavy Stress (32B AWQ) ---"
    for CONC in 1 4 8 16; do
        run_bench "p3_heavy" "$P3_MODEL" "$CONC"
        sleep 15
    done
fi

echo "=== SGLang benchmarks complete ==="
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
