#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BASE_URL="${VLLM_BENCH_URL:-http://localhost:8000}"
MODEL="${VLLM_BENCH_MODEL:-}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model specified. Set VLLM_BENCH_MODEL env var." >&2
    exit 1
fi
RESULTS_DIR="${VLLM_RESULTS_DIR:-${PROJECT_ROOT}/results/vllm}"
DATASET="${VLLM_BENCH_DATASET:-sharegpt}"
DATASET_PATH="${VLLM_BENCH_DATASET_PATH:-}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm 2>/dev/null || echo "${PROJECT_ROOT}/.venv/bin/vllm")}"
MAX_SEQS="${VLLM_BENCH_MAX_SEQS:-512}"

CONC_BASE="${VLLM_CONC_BASE:-1}"
CONC_STEP="${VLLM_CONC_STEP:-100}"
CONC_MAX="${VLLM_CONC_MAX:-2000}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Incremental stress benchmark for vLLM.
Starts at --conc-base, increases by --conc-step each round,
stops when a concurrency level fails or --conc-max is reached.

OPTIONS:
  -u, --url URL           Server base URL (default: http://localhost:8000)
  -m, --model NAME        Model name for request
  -o, --output DIR        Results directory (default: ./results/vllm)
  -d, --dataset NAME      Dataset: sharegpt, random (default: sharegpt)
  -dp, --dataset-path P   Dataset file path
  --conc-base N           Starting concurrency (default: 1)
  --conc-step N           Increment per round (default: 100)
  --conc-max N            Maximum concurrency (default: 2000)
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                   # defaults: 1 → 100 → 200 → ... → 2000
  $(basename "$0") --conc-base 50 --conc-step 50     # 50 → 100 → 150 → ...
  $(basename "$0") --conc-max 500 --conc-step 25     # 1 → 26 → 51 → ... → 500
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)           BASE_URL="$2"; shift 2 ;;
        -m|--model)         MODEL="$2"; shift 2 ;;
        -o|--output)        RESULTS_DIR="$2"; shift 2 ;;
        -d|--dataset)       DATASET="$2"; shift 2 ;;
        -dp|--dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --conc-base)        CONC_BASE="$2"; shift 2 ;;
        --conc-step)        CONC_STEP="$2"; shift 2 ;;
        --conc-max)         CONC_MAX="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown: $1" >&2; usage ;;
    esac
done

mkdir -p "$RESULTS_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${GREEN}${BOLD}=== vLLM Incremental Stress Benchmark ===${NC}"
echo "  Base URL     $BASE_URL"
echo "  Model        $MODEL"
echo "  Dataset      $DATASET"
echo "  Conc range   ${CONC_BASE} → ${CONC_MAX} (step ${CONC_STEP})"
echo "  Results      $RESULTS_DIR"
echo ""

run_bench() {
    local conc=$1
    # Scale num_prompts: at least conc*8, but cap at MAX_SEQS to match server capacity
    local num_prompts=$((conc * 8))
    if [[ "$num_prompts" -gt "$MAX_SEQS" ]]; then
        num_prompts="$MAX_SEQS"
    fi
    local outfile="${RESULTS_DIR}/stress_conc${conc}.json"

    echo -e "[$(date +%H:%M:%S)] ${CYAN}conc=${conc}${NC} num_prompts=${num_prompts}"

    if $VLLM_BIN bench serve \
        --backend openai-chat \
        --base-url "$BASE_URL" \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --dataset-name "$DATASET" \
        ${DATASET_PATH:+--dataset-path "$DATASET_PATH"} \
        --num-prompts "$num_prompts" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "stress_conc${conc}.json" \
        2>&1; then
        echo -e "  ${GREEN}✓${NC} conc=${conc} passed — Saved: $outfile"
        return 0
    else
        echo -e "  ${RED}✗${NC} conc=${conc} FAILED"
        return 1
    fi
}

CONC="$CONC_BASE"
PASSED=0
FAILED_CONC=0

while [[ "$CONC" -le "$CONC_MAX" ]]; do
    echo -e "--- ${BOLD}Round: conc=${CONC}${NC} ---"

    if run_bench "$CONC"; then
        PASSED=$((PASSED + 1))
        CONC=$((CONC + CONC_STEP))
    else
        FAILED_CONC=$CONC
        echo -e "\n${RED}${BOLD}Failed at conc=${CONC}. Stopping stress test.${NC}\n"
        break
    fi

    echo ""
done

echo -e "${GREEN}${BOLD}=== Stress benchmark complete ===${NC}"
echo "  Rounds passed:   $PASSED"
if [[ $FAILED_CONC -gt 0 ]]; then
    echo -e "  ${RED}Failed at:       conc=${FAILED_CONC}${NC}"
fi
echo "  Results:         $RESULTS_DIR/"
echo ""
ls -la "$RESULTS_DIR/"
