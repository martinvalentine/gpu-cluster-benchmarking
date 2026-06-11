#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BASE_URL="${LLAMA_BENCH_URL:-http://localhost:8001/v1}"
RESULTS_DIR="${LLAMA_RESULTS_DIR:-${PROJECT_ROOT}/results/llamacpp}"

CONC_BASE="${LLAMA_CONC_BASE:-1}"
CONC_STEP="${LLAMA_CONC_STEP:-4}"
CONC_MAX="${LLAMA_CONC_MAX:-64}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Incremental stress benchmark for llama.cpp server.
Starts at --conc-base, increases by --conc-step each round,
stops when a concurrency level fails or --conc-max is reached.

OPTIONS:
  -u, --url URL           Server base URL (default: http://localhost:8001/v1)
  -o, --output DIR        Results directory (default: ./results/llamacpp)
  --conc-base N           Starting concurrency (default: 1)
  --conc-step N           Increment per round (default: 4)
  --conc-max N            Maximum concurrency (default: 64)
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                 # 1 → 5 → 9 → ... → 64
  $(basename "$0") --conc-base 4 --conc-step 4     # 4 → 8 → 12 → ... → 64
  $(basename "$0") --conc-base 1 --conc-step 1 --conc-max 16  # 1 → 2 → ... → 16
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)           BASE_URL="$2"; shift 2 ;;
        -o|--output)        RESULTS_DIR="$2"; shift 2 ;;
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
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

sep()  { echo -e "${DIM}$(printf '%.0s━' {1..55})${NC}"; }
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }

gpu_vram() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "VRAM: %s/%s MiB | GPU: %s%% | Temp: %s°C", $1, $2, $3, $4}' || echo "GPU: n/a"
}

get_model_name() {
    curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    if name.endswith('.gguf'):
        name = name.rsplit('.gguf', 1)[0]
    print(name)
except: print('model')
" 2>/dev/null || echo "model"
}

run_concurrent() {
    local np=$1
    local prompt="Write an essay about the benefits of AI in education, 300 words."
    local max_tok=512
    local model
    model=$(get_model_name)
    local escaped_prompt
    escaped_prompt=$(python3 -c "import json; print(json.dumps('$prompt'))")
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local t_start t_end

    t_start=$(date +%s%3N)

    for i in $(seq 1 "$np"); do
        (
            local resp t0 t1
            t0=$(date +%s%3N)
            resp=$(curl -sf --max-time 600 "${BASE_URL}/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"$model\",
                    \"messages\": [{\"role\": \"user\", \"content\": $escaped_prompt}],
                    \"max_tokens\": $max_tok,
                    \"stream\": false,
                    \"temperature\": 0.6
                }" 2>/dev/null) || true
            t1=$(date +%s%3N)
            if [ -n "$resp" ]; then
                local c_tok elapsed
                c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)
                elapsed=$(( t1 - t0 ))
                echo "$c_tok $elapsed" > "$tmp_dir/res_$i"
            fi
        ) &
    done

    wait
    t_end=$(date +%s%3N)
    local wall=$(( t_end - t_start ))

    local total_tok=0 total_time=0 count=0
    for f in "$tmp_dir"/res_*; do
        [ -f "$f" ] || continue
        read -r c_tok elapsed < "$f"
        total_tok=$(( total_tok + c_tok ))
        total_time=$(( total_time + elapsed ))
        count=$(( count + 1 ))
    done

    rm -rf "$tmp_dir"

    local agg_tps="?" per_user_tps="?" avg_latency="?"
    if [ "$count" -gt 0 ] && [ "$wall" -gt 0 ]; then
        agg_tps=$(echo "scale=1; $total_tok * 1000 / $wall" | bc 2>/dev/null || echo "?")
        per_user_tps=$(echo "scale=1; $total_tok * 1000 / $total_time" | bc 2>/dev/null || echo "?")
        avg_latency=$(echo "scale=0; $total_time / $count" | bc 2>/dev/null || echo "?")
    fi

    echo "$count $total_tok $wall $agg_tps $per_user_tps $avg_latency"

    if [ "$count" -eq "$np" ]; then
        return 0
    else
        return 1
    fi
}

echo ""
echo -e "${GREEN}${BOLD}=== llama.cpp Incremental Stress Benchmark ===${NC}"
echo "  Server       ${BASE_URL}"
echo "  Conc range   ${CONC_BASE} → ${CONC_MAX} (step ${CONC_STEP})"
echo "  Results      ${RESULTS_DIR}"
echo "  GPU          $(gpu_vram)"
echo ""

OUT_FILE="${RESULTS_DIR}/stress_summary.tsv"
echo -e "conc\tcompleted\ttotal_tokens\twall_ms\tagg_tps\tper_user_tps\tavg_latency_ms\tstatus" > "$OUT_FILE"

CONC="$CONC_BASE"
PASSED=0
FAILED_CONC=0

while [[ "$CONC" -le "$CONC_MAX" ]]; do
    sep
    log "${BOLD}Round: conc=${CONC}${NC}"
    sep

    result=$(run_concurrent "$CONC") && status="pass" || status="fail"

    read -r completed total_tok wall agg_tps per_user_tps avg_latency <<< "$result"

    if [ "$status" = "pass" ]; then
        ok "conc=${CONC} — completed=${completed}/${CONC} total_tok=${total_tok} wall=${wall}ms agg=${agg_tps} t/s"
        echo "  GPU: $(gpu_vram)"
        PASSED=$((PASSED + 1))
        echo -e "$CONC\t$completed\t$total_tok\t$wall\t$agg_tps\t$per_user_tps\t$avg_latency\tpass" >> "$OUT_FILE"
        CONC=$((CONC + CONC_STEP))
    else
        FAILED_CONC=$CONC
        fail "conc=${CONC} — completed=${completed}/${CONC} total_tok=${total_tok}"
        echo -e "$CONC\t$completed\t$total_tok\t$wall\t$agg_tps\t$per_user_tps\t$avg_latency\tfail" >> "$OUT_FILE"
        echo ""
        echo -e "${RED}${BOLD}Failed at conc=${CONC}. Stopping stress test.${NC}"
        break
    fi

    echo ""
done

echo ""
sep
echo -e "${GREEN}${BOLD}=== Stress benchmark complete ===${NC}"
echo "  Rounds passed:   $PASSED"
if [[ $FAILED_CONC -gt 0 ]]; then
    echo -e "  ${RED}Failed at:       conc=${FAILED_CONC}${NC}"
fi
echo "  Results:         ${OUT_FILE}"
echo ""
sep
echo -e "${CYAN}$(basename "$OUT_FILE")${NC}"
column -t -s $'\t' "$OUT_FILE" 2>/dev/null | sed 's/^/  /'
echo ""
