#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists (does not override existing env vars)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BASE_URL="${LLAMA_BENCH_URL:-http://localhost:8001/v1}"
RESULTS_DIR="${LLAMA_RESULTS_DIR:-${PROJECT_ROOT}/results/llamacpp}"
CONC_LEVELS="${LLAMA_CONC_LEVELS:-1 4 8 16}"
CTX_SIZES="${LLAMA_CTX_SIZES:-1024 4096 16384}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark llama-server via direct HTTP calls.
Tests: single user, concurrent users, long context.

OPTIONS:
  -u, --url URL           Server base URL (default: http://localhost:8001/v1)
  -o, --output DIR        Results directory (default: ./results/llamacpp)
  -c, --concurrency LVL   Concurrency levels (default: "1 4 8 16")
  -x, --ctx-sizes SIZES   Context sizes for long context test (default: "1024 4096 16384")
  -h, --help              Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)          BASE_URL="$2"; shift 2 ;;
        -o|--output)       RESULTS_DIR="$2"; shift 2 ;;
        -c|--concurrency)  CONC_LEVELS="$2"; shift 2 ;;
        -x|--ctx-sizes)    CTX_SIZES="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown: $1" >&2; usage ;;
    esac
done

mkdir -p "$RESULTS_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }
sep()  { echo -e "${DIM}$(printf '%.0s━' {1..55})${NC}"; }

gpu_vram() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "VRAM: %s/%s MiB | GPU: %s%% | Temp: %s°C", $1, $2, $3, $4}' || echo "GPU: n/a"
}

# ─── Health check ───────────────────────────────────────────────────────────
wait_server() {
    log "Waiting for server at ${BASE_URL%/v1} ..."
    for i in $(seq 1 60); do
        if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
            local model
            model=$(curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['data'][0]['id'])
except: print('unknown')
" 2>/dev/null || echo "unknown")
            ok "Server ready — model: $model"
            return 0
        fi
        printf "  [%02d/60] polling...\r" "$i"; sleep 2
    done
    fail "Server not ready after 120s"; exit 1
}

# ─── Single request ─────────────────────────────────────────────────────────
chat_request() {
    local prompt="$1" max_tokens="${2:-512}" stream="${3:-false}"
    local model
    model=$(curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    if name.endswith('.gguf'):
        name = name.rsplit('.gguf', 1)[0]
    print(name)
except: print('model')
" 2>/dev/null || echo "model")

    curl -sf --max-time 600 "${BASE_URL}/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('$prompt'))")}],
            \"max_tokens\": $max_tokens,
            \"stream\": $stream,
            \"temperature\": 0.6,
            \"top_p\": 0.95
        }"
}

# ─── TEST 1: Single user latency ────────────────────────────────────────────
bench_single() {
    sep
    log "${BOLD}TEST 1: Single user — latency & tokens/s${NC}"
    sep

    local prompts=(
        "Write a short greeting."
        "Explain transformer architecture in 3 sentences."
        "Write Python code to read a CSV and compute the mean of a column."
        "Analyze the pros and cons of MoE vs dense transformer in 10 sentences."
    )
    local max_tokens=(32 128 256 512)
    local labels=("short" "medium" "code" "long")

    local out_file="$RESULTS_DIR/single_user.tsv"
    echo -e "label\tprompt_tokens\tcompletion_tokens\tttft_ms\tdecode_tps\ttotal_s" > "$out_file"

    for i in "${!prompts[@]}"; do
        local label="${labels[$i]}"
        local prompt="${prompts[$i]}"
        local max_tok="${max_tokens[$i]}"

        log "  [$((i+1))/${#prompts[@]}] $label (max_tokens=$max_tok)..."

        local t_start t_end elapsed
        t_start=$(date +%s%3N)

        local resp
        resp=$(chat_request "$prompt" "$max_tok" "false" 2>/dev/null) || true
        t_end=$(date +%s%3N)
        elapsed=$(( t_end - t_start ))

        if [ -z "$resp" ]; then
            warn "    No response"; continue
        fi

        local p_tok c_tok finish
        p_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null || echo "?")
        c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
        finish=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['finish_reason'])" 2>/dev/null || echo "?")

        local decode_tps="?"
        if [[ "$c_tok" =~ ^[0-9]+$ ]] && [ "$c_tok" -gt 0 ]; then
            decode_tps=$(awk "BEGIN{printf \"%.1f\", $c_tok * 1000 / $elapsed}" 2>/dev/null || echo "?")
        fi

        ok "    prompt=$p_tok tok | completion=$c_tok tok | total=${elapsed}ms | decode=$decode_tps t/s | finish=$finish"
        echo -e "$label\t$p_tok\t$c_tok\t${elapsed}\t$decode_tps\t$(awk "BEGIN{printf \"%.2f\", $elapsed/1000}")" >> "$out_file"
        echo "  GPU: $(gpu_vram)"
    done
}

# ─── TEST 2: Concurrent users ───────────────────────────────────────────────
bench_concurrent() {
    local np="$1"
    sep
    log "${BOLD}TEST 2: Concurrent $np users — aggregate throughput${NC}"
    sep

    local prompt="Write an essay about the benefits of AI in education, 300 words."
    local max_tok=512
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local t_start t_end

    t_start=$(date +%s%3N)

    # Get model name once
    local model
    model=$(curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    if name.endswith('.gguf'):
        name = name.rsplit('.gguf', 1)[0]
    print(name)
except: print('model')
" 2>/dev/null || echo "model")

    local escaped_prompt
    escaped_prompt=$(python3 -c "import json; print(json.dumps('$prompt'))")

    # Spawn np requests in parallel
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

    # Aggregate results
    local total_tok=0 total_time=0 count=0
    for f in "$tmp_dir"/res_*; do
        [ -f "$f" ] || continue
        read -r c_tok elapsed < "$f"
        total_tok=$(( total_tok + c_tok ))
        total_time=$(( total_time + elapsed ))
        count=$(( count + 1 ))
    done

    local agg_tps per_user_tps avg_latency
    if [ "$count" -gt 0 ] && [ "$wall" -gt 0 ]; then
        agg_tps=$(awk "BEGIN{printf \"%.1f\", $total_tok * 1000 / $wall}" 2>/dev/null || echo "?")
        per_user_tps=$(awk "BEGIN{printf \"%.1f\", $total_tok * 1000 / $total_time}" 2>/dev/null || echo "?")
        avg_latency=$(awk "BEGIN{printf \"%.0f\", $total_time / $count}" 2>/dev/null || echo "?")
    fi

    ok "  Completed: $count/$np requests"
    ok "  Total tokens: $total_tok"
    ok "  Wall time: ${wall}ms | Aggregate: $agg_tps tok/s"
    ok "  Per-user avg: $per_user_tps tok/s | Avg latency: ${avg_latency}ms"
    echo "  GPU: $(gpu_vram)"

    local out_file="$RESULTS_DIR/concurrent_${np}.tsv"
    echo -e "np\tcompleted\ttotal_tokens\twall_ms\tagg_tps\tper_user_tps\tavg_latency_ms" > "$out_file"
    echo -e "$np\t$count\t$total_tok\t$wall\t$agg_tps\t$per_user_tps\t$avg_latency" >> "$out_file"

    rm -rf "$tmp_dir"
}

# ─── TEST 3: Long context ───────────────────────────────────────────────────
bench_long_context() {
    sep
    log "${BOLD}TEST 3: Long context — prefill speed${NC}"
    sep

    local base_text="This is a long passage used to test model prefill speed. "
    local out_file="$RESULTS_DIR/long_context.tsv"
    echo -e "ctx_tokens\tprefill_ms\tprefill_tps\tdecode_tps" > "$out_file"

    for ctx in $CTX_SIZES; do
        # Build prompt of ~ctx tokens (~4 chars/token)
        local repeat=$(( ctx * 4 / ${#base_text} + 1 ))
        local long_prompt=""
        for _ in $(seq 1 "$repeat"); do
            long_prompt+="$base_text"
        done
        long_prompt="${long_prompt:0:$(( ctx * 4 ))}"
        long_prompt="${long_prompt//\"/\\\"} Summarize the above in 1 sentence."

        log "  Testing ctx ~${ctx} tokens..."

        local t_start t_end elapsed
        t_start=$(date +%s%3N)

        local resp
        resp=$(curl -sf --max-time 600 "${BASE_URL}/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$(curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'].rsplit('.gguf',1)[0])" 2>/dev/null || echo "model")\",
                \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('''$long_prompt'''))")}],
                \"max_tokens\": 64,
                \"stream\": false,
                \"temperature\": 0.1
            }" 2>/dev/null) || true

        t_end=$(date +%s%3N)
        elapsed=$(( t_end - t_start ))

        if [ -z "$resp" ]; then
            warn "    No response (context may exceed slot size)"
            echo -e "$ctx\ttimeout\t?\t?" >> "$out_file"
            continue
        fi

        local p_tok c_tok
        p_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null || echo 0)
        c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)

        local prefill_tps="?" decode_tps="?"
        if [[ "$c_tok" =~ ^[0-9]+$ ]] && [ "$c_tok" -gt 0 ]; then
            decode_tps=$(awk "BEGIN{printf \"%.1f\", $c_tok * 1000 / $elapsed}" 2>/dev/null || echo "?")
        fi
        if [[ "$p_tok" =~ ^[0-9]+$ ]] && [ "$p_tok" -gt 0 ]; then
            prefill_tps=$(awk "BEGIN{printf \"%.0f\", $p_tok * 1000 / $elapsed}" 2>/dev/null || echo "?")
        fi

        ok "    p_tok=$p_tok | c_tok=$c_tok | total=${elapsed}ms | prefill~$prefill_tps t/s | decode~$decode_tps t/s"
        echo -e "$p_tok\t$elapsed\t$prefill_tps\t$decode_tps" >> "$out_file"
    done
}

# ─── TEST 4: Server metrics ────────────────────────────────────────────────
snap_metrics() {
    sep
    log "${BOLD}TEST 4: Server /metrics snapshot${NC}"
    sep

    local metrics
    metrics=$(curl -sf "${BASE_URL%/v1}/metrics" 2>/dev/null || echo "")
    if [ -z "$metrics" ]; then
        warn "Cannot reach /metrics (start llama-server with --metrics to enable)"
        return
    fi

    local slots_idle slots_proc kv_ratio tps
    slots_idle=$(echo "$metrics" | grep '^llamacpp:slots_idle ' | awk '{print $2}')
    slots_proc=$(echo "$metrics" | grep '^llamacpp:slots_processing ' | awk '{print $2}')
    kv_ratio=$(echo "$metrics" | grep '^llamacpp:kv_cache_usage_ratio ' | awk '{print $2}')
    tps=$(echo "$metrics" | grep '^llamacpp:tokens_per_second ' | awk '{print $2}')

    echo "  Slots idle:       ${slots_idle:-N/A}"
    echo "  Slots processing: ${slots_proc:-N/A}"
    echo "  KV cache usage:   $(awk "BEGIN{printf \"%.1f\", ${kv_ratio:-0} * 100}")%"
    echo "  Tokens/sec:       ${tps:-N/A}"
    echo "  GPU: $(gpu_vram)"

    echo "$metrics" > "$RESULTS_DIR/metrics_snapshot.txt"
    ok "Full metrics saved to $RESULTS_DIR/metrics_snapshot.txt"
}

# ─── Summary ────────────────────────────────────────────────────────────────
print_summary() {
    sep
    log "${BOLD}SUMMARY — Results: $RESULTS_DIR${NC}"
    sep
    for f in "$RESULTS_DIR"/*.tsv; do
        [ -f "$f" ] || continue
        echo -e "${CYAN}$(basename "$f")${NC}"
        column -t -s $'\t' "$f" 2>/dev/null | sed 's/^/  /' || cat "$f" | sed 's/^/  /'
        echo ""
    done
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    sep
    echo -e "${BOLD}  llama-server Benchmark — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "  Server: $BASE_URL"
    echo -e "  GPU: $(gpu_vram)"
    sep

    wait_server
    snap_metrics
    bench_single

    for conc in $CONC_LEVELS; do
        bench_concurrent "$conc"
    done

    bench_long_context
    print_summary

    sep
    ok "Benchmark complete! Results: $RESULTS_DIR"
    sep
}

main "$@" || true
