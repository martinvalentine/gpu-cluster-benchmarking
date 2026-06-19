#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists (does not override existing env vars)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BASE_URL="${LITELLM_BENCH_URL:-http://localhost:4000}"
MODEL="${LITELLM_BENCH_MODEL:-qwen0.5b-llamacpp}"
RESULTS_DIR="${LITELLM_RESULTS_DIR:-${PROJECT_ROOT}/results/litellm}"
METHOD="${LITELLM_BENCH_METHOD:-async}"

CONCURRENCY="${LITELLM_BENCH_CONCURRENCY:-10}"
NUM_REQUESTS="${LITELLM_BENCH_NUM_REQUESTS:-100}"
MAX_TOKENS="${LITELLM_BENCH_MAX_TOKENS:-256}"
CACHE_RATIO="${LITELLM_BENCH_CACHE_RATIO:-60}"

LOCUST_USERS="${LITELLM_LOCUST_USERS:-50}"
LOCUST_SPAWN_RATE="${LITELLM_LOCUST_SPAWN_RATE:-5}"
LOCUST_RUN_TIME="${LITELLM_LOCUST_RUN_TIME:-5m}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark LiteLLM proxy with Redis semantic cache.
Measures TTFT, TPOT, ITL, throughput, and cache hit rate through the full
proxy stack: Gateway -> Cache -> Serving Engine.

OPTIONS:
  -u, --url URL           LiteLLM proxy URL (default: http://localhost:4000)
  -m, --model NAME        Model name as declared in litellm_config.yaml (default: qwen0.5b-llamacpp)
  -o, --output DIR        Results directory (default: ./results/litellm)
  -M, --method METHOD     Benchmark method: async, locust (default: async)
  -c, --concurrency N     Max concurrent requests for async method (default: 10)
  -n, --num-requests N    Total requests for async method (default: 100)
  --max-tokens N          Max tokens per response (default: 256)
  --cache-ratio PCT       Percentage of repeated prompts to test cache hits (default: 60)
  --users N               Locust virtual users (default: 50)
  --spawn-rate N          Locust spawn rate per second (default: 5)
  --run-time DURATION     Locust run duration (default: 5m)
  -h, --help              Show this help

ENV OVERRIDES:
  LITELLM_BENCH_URL, LITELLM_BENCH_MODEL, LITELLM_RESULTS_DIR,
  LITELLM_BENCH_METHOD, LITELLM_BENCH_CONCURRENCY, LITELLM_BENCH_NUM_REQUESTS,
  LITELLM_BENCH_MAX_TOKENS, LITELLM_BENCH_CACHE_RATIO,
  LITELLM_LOCUST_USERS, LITELLM_LOCUST_SPAWN_RATE, LITELLM_LOCUST_RUN_TIME

METHODS:
  async    Async Python script measuring TTFT/TPOT/ITL per request (detailed)
  locust   Locust-based load test with realistic user simulation (scalable)

EXAMPLES:
  $(basename "$0")                                    # Defaults (async, 10 concurrent)
  $(basename "$0") -M locust --users 100 --run-time 10m
  $(basename "$0") -c 32 -n 200 --cache-ratio 80     # Heavy cache test
  $(basename "$0") -u http://gpu-pod:4000 -m qwen7b-sglang
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)           BASE_URL="$2"; shift 2 ;;
        -m|--model)         MODEL="$2"; shift 2 ;;
        -o|--output)        RESULTS_DIR="$2"; shift 2 ;;
        -M|--method)        METHOD="$2"; shift 2 ;;
        -c|--concurrency)   CONCURRENCY="$2"; shift 2 ;;
        -n|--num-requests)  NUM_REQUESTS="$2"; shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2"; shift 2 ;;
        --cache-ratio)      CACHE_RATIO="$2"; shift 2 ;;
        --users)            LOCUST_USERS="$2"; shift 2 ;;
        --spawn-rate)       LOCUST_SPAWN_RATE="$2"; shift 2 ;;
        --run-time)         LOCUST_RUN_TIME="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown option: $1" >&2; usage ;;
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
echo -e "  ${GREEN}${BOLD}  LiteLLM Proxy Benchmark${NC}${DIM} — Cache + Latency${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Proxy URL"    "$BASE_URL"
row "Model"        "$MODEL"
row "Method"       "$METHOD"
sep
if [[ "$METHOD" == "async" ]]; then
    row "Concurrency"  "$CONCURRENCY"
    row "Requests"     "$NUM_REQUESTS"
    row "Max Tokens"   "$MAX_TOKENS"
    row "Cache Ratio"  "${CACHE_RATIO}% repeated"
else
    row "Users"        "$LOCUST_USERS"
    row "Spawn Rate"   "${LOCUST_SPAWN_RATE}/s"
    row "Run Time"     "$LOCUST_RUN_TIME"
fi
sep
row "Results"      "$RESULTS_DIR"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

echo "Checking proxy health at ${BASE_URL}/health ..."
if ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "WARNING: LiteLLM proxy not reachable at ${BASE_URL}" >&2
    echo "  Start proxy first: ./scripts/run/run-proxy.sh" >&2
    read -rp "  Continue anyway? [y/N] " yn
    [[ "$yn" =~ ^[Yy] ]] || exit 1
fi
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "$METHOD" in
    async)
        OUTFILE="${RESULTS_DIR}/litellm_async_${TIMESTAMP}.json"
        echo "Running async benchmark -> $OUTFILE"
        echo ""
        uv run python "${SCRIPT_DIR}/bench-litellm-async.py" \
            --base-url "$BASE_URL" \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY" \
            --num-requests "$NUM_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --cache-ratio "$CACHE_RATIO" \
            --output "$OUTFILE"
        ;;
    locust)
        CSV_PREFIX="${RESULTS_DIR}/litellm_locust_${TIMESTAMP}"
        echo "Running locust benchmark -> ${CSV_PREFIX}_*.csv"
        echo ""
        uv run locust \
            -f "${SCRIPT_DIR}/bench-litellm-locust.py" \
            --host "$BASE_URL" \
            --headless \
            --users "$LOCUST_USERS" \
            --spawn-rate "$LOCUST_SPAWN_RATE" \
            --run-time "$LOCUST_RUN_TIME" \
            --csv "$CSV_PREFIX" \
            --print-stats
        echo ""
        echo "Locust CSV results:"
        ls -la "${CSV_PREFIX}"_*.csv 2>/dev/null || true
        ;;
    *)
        echo "ERROR: Unknown method: $METHOD (use 'async' or 'locust')" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Benchmark complete ==="
echo "Results in: $RESULTS_DIR/"
