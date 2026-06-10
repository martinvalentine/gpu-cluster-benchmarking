#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

_cleanup() { rm -rf "${_BENCH_TMP_DIRS[@]}" 2>/dev/null || true; }
declare -a _BENCH_TMP_DIRS=()
trap _cleanup EXIT

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }
sep()  { echo -e "${DIM}$(printf '%.0s━' {1..55})${NC}"; }

RESULTS_DIR="${PROJECT_ROOT}/results"
mkdir -p "$RESULTS_DIR"/{vllm,llamacpp,litellm,embed}

wait_health() {
    local url="$1" name="$2"
    log "Waiting for $name at $url ..."
    for i in $(seq 1 60); do
        if curl -sf "$url" >/dev/null 2>&1; then
            ok "$name ready"
            return 0
        fi
        printf "  [%02d/60] polling...\r" "$i"; sleep 2
    done
    fail "$name not ready after 120s"
    return 1
}

echo ""
sep
echo -e "${BOLD}  Benchmark Suite — Qwen2.5-0.5B${NC}"
echo -e "  $(date '+%Y-%m-%d %H:%M:%S')"
sep
echo ""

wait_health "http://localhost:8000/health" "vLLM"
wait_health "http://localhost:8001/health" "llama.cpp"

# ─── vLLM Benchmark ───────────────────────────────────────────────────────
echo ""
sep
echo -e "${BOLD}  [1/3] vLLM Benchmark${NC}"
sep

VLLM_MODEL="${PROJECT_ROOT}/models/hf/qwen2.5-0.6b"
VLLM_URL="http://localhost:8000"
VLLM_DIR="$RESULTS_DIR/vllm"

for CONC in 1 4 8; do
    NUM_PROMPTS=$((CONC * 8))
    log "vLLM: conc=$CONC num_prompts=$NUM_PROMPTS"
    uv run vllm bench serve \
        --backend openai-chat \
        --base-url "$VLLM_URL" \
        --model "$VLLM_MODEL" \
        --dataset-name sharegpt \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$VLLM_DIR" \
        --result-filename "p0_conc${CONC}.json" \
        2>&1 || warn "vLLM bench failed for conc=$CONC"
    sleep 3
done
ok "vLLM benchmarks complete → $VLLM_DIR/"

# ─── llama.cpp Benchmark ──────────────────────────────────────────────────
echo ""
sep
echo -e "${BOLD}  [2/3] llama.cpp Benchmark${NC}"
sep

LLAMA_URL="http://localhost:8001/v1"
LLAMA_DIR="$RESULTS_DIR/llamacpp"

chat_request() {
    local prompt="$1" max_tokens="${2:-512}"
    local model
    model=$(curl -sf "${LLAMA_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    if name.endswith('.gguf'):
        name = name.rsplit('.gguf', 1)[0]
    print(name)
except: print('model')
" 2>/dev/null || echo "model")

    local escaped
    escaped=$(python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" <<< "$prompt")

    curl -sf --max-time 600 "${LLAMA_URL}/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": $escaped}],
            \"max_tokens\": $max_tokens,
            \"stream\": false,
            \"temperature\": 0.6,
            \"top_p\": 0.95
        }"
}

# Single user
log "llama.cpp: single user latency"
SINGLE_OUT="$LLAMA_DIR/single_user.tsv"
echo -e "label\tprompt_tokens\tcompletion_tokens\tttft_ms\tdecode_tps\ttotal_s" > "$SINGLE_OUT"

PROMPTS=("Write a short greeting." "Explain transformers in 3 sentences." "Write Python code to read a CSV." "Analyze MoE vs dense transformer.")
MAX_TOKS=(32 128 256 512)
LABELS=("short" "medium" "code" "long")

for i in "${!PROMPTS[@]}"; do
    label="${LABELS[$i]}"
    prompt="${PROMPTS[$i]}"
    max_tok="${MAX_TOKS[$i]}"

    t_start=$(date +%s%3N)
    resp=$(chat_request "$prompt" "$max_tok" 2>/dev/null) || true
    t_end=$(date +%s%3N)
    elapsed=$(( t_end - t_start ))

    if [ -n "$resp" ]; then
        p_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null || echo "?")
        c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
        decode_tps="?"
        if [[ "$c_tok" =~ ^[0-9]+$ ]] && [ "$c_tok" -gt 0 ]; then
            decode_tps=$(echo "scale=1; $c_tok * 1000 / $elapsed" | bc 2>/dev/null || echo "?")
        fi
        ok "  $label: prompt=$p_tok tok | completion=$c_tok tok | ${elapsed}ms | decode=$decode_tps t/s"
        echo -e "$label\t$p_tok\t$c_tok\t${elapsed}\t$decode_tps\t$(echo "scale=2; $elapsed/1000" | bc)" >> "$SINGLE_OUT"
    else
        warn "  $label: no response"
    fi
done

# Concurrent
for CONC in 1 4 8; do
    log "llama.cpp: concurrent $CONC users"
    OUT="$LLAMA_DIR/concurrent_${CONC}.tsv"
    echo -e "np\tcompleted\ttotal_tokens\twall_ms\tagg_tps\tper_user_tps\tavg_latency_ms" > "$OUT"

    PROMPT="Write an essay about AI in education, 300 words."
    MAX_TOK=512
    TMP_DIR=$(mktemp -d)
    _BENCH_TMP_DIRS+=("$TMP_DIR")

    t_start=$(date +%s%3N)
    for j in $(seq 1 "$CONC"); do
        (
            t0=$(date +%s%3N)
            resp=$(chat_request "$PROMPT" "$MAX_TOK" 2>/dev/null) || true
            t1=$(date +%s%3N)
            if [ -n "$resp" ]; then
                c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)
                echo "$c_tok $(( t1 - t0 ))" > "$TMP_DIR/res_$j"
            fi
        ) &
    done
    wait
    t_end=$(date +%s%3N)
    wall=$(( t_end - t_start ))

    total_tok=0 total_time=0 count=0
    for f in "$TMP_DIR"/res_*; do
        [ -f "$f" ] || continue
        read -r c_tok elapsed < "$f"
        total_tok=$(( total_tok + c_tok ))
        total_time=$(( total_time + elapsed ))
        count=$(( count + 1 ))
    done

    agg_tps="?" per_user_tps="?" avg_latency="?"
    if [ "$count" -gt 0 ] && [ "$wall" -gt 0 ]; then
        agg_tps=$(echo "scale=1; $total_tok * 1000 / $wall" | bc 2>/dev/null || echo "?")
        per_user_tps=$(echo "scale=1; $total_tok * 1000 / $total_time" | bc 2>/dev/null || echo "?")
        avg_latency=$(echo "scale=0; $total_time / $count" | bc 2>/dev/null || echo "?")
    fi

    ok "  $count/$CONC done | total=$total_tok tok | wall=${wall}ms | agg=$agg_tps t/s"
    echo -e "$CONC\t$count\t$total_tok\t$wall\t$agg_tps\t$per_user_tps\t$avg_latency" >> "$OUT"
    rm -rf "$TMP_DIR"
    sleep 2
done
ok "llama.cpp benchmarks complete → $LLAMA_DIR/"

# ─── LiteLLM Proxy Benchmark ──────────────────────────────────────────────
echo ""
sep
echo -e "${BOLD}  [3/3] LiteLLM Proxy Benchmark${NC}"
sep

LITELLM_URL="http://localhost:4000"
LITELLM_DIR="$RESULTS_DIR/litellm"

if curl -sf "$LITELLM_URL/health" >/dev/null 2>&1; then
    log "LiteLLM proxy: testing via proxy"

    for CONC in 1 4; do
        log "LiteLLM: conc=$CONC"
        OUT="$LITELLM_DIR/proxy_conc${CONC}.tsv"
        echo -e "np\tcompleted\ttotal_tokens\twall_ms\tagg_tps\tavg_latency_ms" > "$OUT"

        TMP_DIR=$(mktemp -d)
        _BENCH_TMP_DIRS+=("$TMP_DIR")
        t_start=$(date +%s%3N)

        for j in $(seq 1 "$CONC"); do
            (
                t0=$(date +%s%3N)
                resp=$(curl -sf --max-time 600 "$LITELLM_URL/v1/chat/completions" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: Bearer EMPTY" \
                    -d '{
                        "model": "qwen0.5b-vllm",
                        "messages": [{"role": "user", "content": "Write a short greeting."}],
                        "max_tokens": 64,
                        "stream": false
                    }' 2>/dev/null) || true
                t1=$(date +%s%3N)
                if [ -n "$resp" ]; then
                    c_tok=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
                    echo "$c_tok $(( t1 - t0 ))" > "$TMP_DIR/res_$j"
                fi
            ) &
        done
        wait
        t_end=$(date +%s%3N)
        wall=$(( t_end - t_start ))

        total_tok=0 total_time=0 count=0
        for f in "$TMP_DIR"/res_*; do
            [ -f "$f" ] || continue
            read -r c_tok elapsed < "$f"
            total_tok=$(( total_tok + c_tok ))
            total_time=$(( total_time + elapsed ))
            count=$(( count + 1 ))
        done

        agg_tps="?" avg_latency="?"
        if [ "$count" -gt 0 ] && [ "$wall" -gt 0 ]; then
            agg_tps=$(echo "scale=1; $total_tok * 1000 / $wall" | bc 2>/dev/null || echo "?")
            avg_latency=$(echo "scale=0; $total_time / $count" | bc 2>/dev/null || echo "?")
        fi

        ok "  $count/$CONC done | wall=${wall}ms | agg=$agg_tps t/s"
        echo -e "$CONC\t$count\t$total_tok\t$wall\t$agg_tps\t$avg_latency" >> "$OUT"
        rm -rf "$TMP_DIR"
        sleep 2
    done
    ok "LiteLLM benchmarks complete → $LITELLM_DIR/"
else
    warn "LiteLLM proxy not reachable at $LITELLM_URL — skipping"
fi

# ─── Report ────────────────────────────────────────────────────────────────
echo ""
sep
echo -e "${BOLD}  Generating Report${NC}"
sep

cd "$PROJECT_ROOT"
uv run python scripts/report.py --results-dir "$RESULTS_DIR"

echo ""
ok "All benchmarks complete!"
echo -e "  Results: $RESULTS_DIR/"
echo ""