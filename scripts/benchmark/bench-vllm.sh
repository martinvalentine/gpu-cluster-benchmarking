#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists (does not override existing env vars)
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

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark server using vllm bench serve.
Measures TTFT, TPOT, ITL, throughput across phases.

OPTIONS:
  -u, --url URL           Server base URL (default: http://localhost:8000)
  -m, --model NAME        Model name for request (default: models/hf/qwen2.5-0.6b)
  -o, --output DIR        Results directory (default: ./results/vllm)
  -d, --dataset NAME      Dataset: sharegpt, random (default: sharegpt)
  -dp, --dataset-path P   Dataset file path
  -p, --phase PHASE       Run phase: p1, p2, p3, all (default: all)
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0") -p p1
  $(basename "$0") -u http://localhost:8001 -m Qwen3.6-35B-A3B-UDT-Q5_K_XL_MTP -p p1
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

echo "=== vLLM Benchmark (vllm bench serve) ==="
echo "  Base URL     $BASE_URL"
echo "  Model        $MODEL"
echo "  Dataset      $DATASET"
echo "  Phase        $PHASE"
echo "  Results      $RESULTS_DIR"
echo ""

run_bench() {
    local phase_name=$1
    local model=$2
    local conc=$3
    # Scale num_prompts: at least conc*8, but cap at MAX_SEQS to match server capacity
    local num_prompts=$((conc * 8))
    if [[ "$num_prompts" -gt "$MAX_SEQS" ]]; then
        num_prompts="$MAX_SEQS"
    fi
    local outfile="${RESULTS_DIR}/${phase_name}_conc${conc}.json"

    echo "[$(date +%H:%M:%S)] ${phase_name} conc=${conc} num_prompts=${num_prompts}"

    $VLLM_BIN bench serve \
        --backend openai-chat \
        --base-url "$BASE_URL" \
        --endpoint /v1/chat/completions \
        --model "$model" \
        --dataset-name "$DATASET" \
        ${DATASET_PATH:+--dataset-path "$DATASET_PATH"} \
        --num-prompts "$num_prompts" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "${phase_name}_conc${conc}.json" \
        2>&1 || {
            echo "  ERROR: Benchmark failed for ${phase_name} conc=${conc}"
            echo "  Check: curl -sf ${BASE_URL}/v1/models"
            return 0
        }

    echo "  Saved: $outfile"
    echo ""
}

if [[ "$PHASE" == "all" || "$PHASE" == "p0" ]]; then
    echo "--- Phase P0: Smoke Test ---"
    for CONC in 1 4; do
        run_bench "p0_smoke" "$MODEL" "$CONC"
        sleep 3
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "p1" ]]; then
    echo "--- Phase P1: Light Load ---"
    for CONC in 1 32 64 128; do
        run_bench "p1_light" "$MODEL" "$CONC"
        sleep 5
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "p2" ]]; then
    echo "--- Phase P2: Medium Load ---"
    for CONC in 1 16 32 64; do
        run_bench "p2_medium" "$MODEL" "$CONC"
        sleep 10
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "p3" ]]; then
    echo "--- Phase P3: Heavy Stress ---"
    for CONC in 1 4 8 16; do
        run_bench "p3_heavy" "$MODEL" "$CONC"
        sleep 15
    done
fi

echo "=== vLLM benchmarks complete ==="
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
