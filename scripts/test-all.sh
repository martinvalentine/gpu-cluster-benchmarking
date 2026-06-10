#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

_TEST_TMP_FILES=()
_cleanup_test() { rm -f "${_TEST_TMP_FILES[@]}" 2>/dev/null || true; }
trap _cleanup_test EXIT

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
LLAMA_URL="${LLAMA_URL:-http://localhost:8001}"
EMBED_URL="${EMBED_URL:-http://localhost:8003}"
SGLANG_URL="${SGLANG_URL:-http://localhost:8002}"
PROXY_URL="${PROXY_URL:-http://localhost:4000}"
TIMEOUT=10

passed=0 failed=0 total=0

pass() { total=$((total + 1)); passed=$((passed + 1)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { total=$((total + 1)); failed=$((failed + 1)); echo -e "  ${RED}FAIL${NC} $1"; }
skip() { echo -e "  ${DIM}SKIP${NC} $1"; }
sep() { echo -e "${DIM}$(printf '%.0s─' {1..55})${NC}"; }

echo ""
echo -e "  ${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${BOLD}  Service Health Check${NC}"
echo -e "  ${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

# ── 1. System dependencies ────────────────────────────────────
echo -e "  ${CYAN}System Dependencies${NC}"
sep

echo -n "  uv... "
if command -v uv &>/dev/null; then pass "found"; else fail "not found (install: curl -LsSf https://astral.sh/uv/install.sh | sh)"; fi

echo -n "  redis-cli... "
if command -v redis-cli &>/dev/null; then pass "found"; else fail "not found (install: apt install redis-server)"; fi

echo -n "  tmux... "
if command -v tmux &>/dev/null; then pass "found"; else fail "not found (install: apt install tmux)"; fi

echo -n "  nvidia-smi... "
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    pass "${gpu_count}x ${gpu_name}"
else
    fail "not found (no GPU driver?)"
fi

echo ""

# ── 2. Redis ──────────────────────────────────────────────────
echo -e "  ${CYAN}Redis${NC}"
sep

echo -n "  redis-cli ping... "
if redis-cli ping &>/dev/null 2>&1; then
    pass "PONG"
else
    echo -e "  ${YELLOW}Starting Redis...${NC}"
    redis-server --daemonize yes --maxmemory 8gb --maxmemory-policy allkeys-lru --logfile /tmp/redis.log 2>/dev/null || true
    sleep 1
    if redis-cli ping &>/dev/null 2>&1; then
        pass "started"
    else
        fail "cannot connect"
    fi
fi

echo ""

# ── 3. Serving engines ────────────────────────────────────────
echo -e "  ${CYAN}Serving Engines${NC}"
sep

check_endpoint() {
    local name="$1" url="$2" models_path="$3"
    echo -n "  ${name}... "
    if curl -sf --max-time "$TIMEOUT" "${url}${models_path}" >/dev/null 2>&1; then
        local model_id
        model_id=$(curl -sf --max-time "$TIMEOUT" "${url}${models_path}" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    data = d.get('data', d.get('models', []))
    if data:
        print(data[0].get('id', data[0].get('name', '?')))
    else:
        print('no models')
except: print('parse error')
" 2>/dev/null || echo "error")
        pass "running → ${model_id}"
    else
        fail "unreachable at ${url}"
    fi
}

check_endpoint "vLLM (:8000)"     "$VLLM_URL"    "/v1/models"
check_endpoint "llama.cpp (:8001)" "$LLAMA_URL"   "/v1/models"
check_endpoint "Embed (:8003)"     "$EMBED_URL"   "/health"
check_endpoint "SGLang (:8002)"    "$SGLANG_URL"  "/v1/models"

echo ""

# ── 4. LiteLLM Proxy ──────────────────────────────────────────
echo -e "  ${CYAN}LiteLLM Proxy${NC}"
sep

echo -n "  proxy health... "
if curl -sf --max-time "$TIMEOUT" "$PROXY_URL/health" >/dev/null 2>&1; then
    health=$(curl -sf --max-time "$TIMEOUT" "$PROXY_URL/health" 2>/dev/null)
    healthy=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('healthy_count',0))" 2>/dev/null || echo "?")
    unhealthy=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('unhealthy_count',0))" 2>/dev/null || echo "?")
    if [[ "$unhealthy" == "0" ]]; then
        pass "healthy (${healthy} endpoints)"
    else
        fail "unhealthy (${healthy} ok, ${unhealthy} failed)"
    fi
else
    fail "unreachable at $PROXY_URL"
fi

echo -n "  proxy models... "
if curl -sf --max-time "$TIMEOUT" "$PROXY_URL/v1/models" >/dev/null 2>&1; then
    model_list=$(curl -sf --max-time "$TIMEOUT" "$PROXY_URL/v1/models" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
names = [m['id'] for m in d.get('data', [])]
print(', '.join(names) if names else 'none')
" 2>/dev/null || echo "error")
    pass "${model_list}"
else
    fail "cannot list models"
fi

echo ""

# ── 5. Chat completion smoke test ─────────────────────────────
echo -e "  ${CYAN}Smoke Test (Chat via Proxy)${NC}"
sep

# Auto-detect first available model
FIRST_MODEL=$(curl -sf --max-time "$TIMEOUT" "$PROXY_URL/v1/models" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('data', [])
# Prefer vllm model, then any
for m in models:
    if 'vllm' in m['id']:
        print(m['id']); sys.exit(0)
if models:
    print(models[0]['id'])
else:
    print('')
" 2>/dev/null || echo "")

if [[ -z "$FIRST_MODEL" ]]; then
    skip "no models available via proxy"
else
    echo -n "  chat (${FIRST_MODEL})... "
    resp_file=$(mktemp /tmp/test_all_XXXXXX.json)
    _TEST_TMP_FILES+=("$resp_file")
    http_code=$(curl -sf --max-time 30 -o "$resp_file" -w '%{http_code}' \
        -X POST "$PROXY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "{
            \"model\": \"$FIRST_MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" 2>/dev/null) || true

    if [[ "$http_code" == "200" ]]; then
        content=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
print(d['choices'][0]['message']['content'].strip())
" "$resp_file" 2>/dev/null || echo "?")
        pass "\"${content}\""
    else
        fail "HTTP ${http_code:-timeout}"
    fi
    rm -f "$resp_file"
fi

echo ""

# ── 6. Cache test ─────────────────────────────────────────────
if [[ -n "$FIRST_MODEL" ]]; then
    echo -e "  ${CYAN}Cache Test${NC}"
    sep

    PAYLOAD="{\"model\":\"$FIRST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 1+1?\"}],\"max_tokens\":8,\"temperature\":0}"

    # First request (cache miss)
    echo -n "  cache miss... "
    t_start=$(date +%s%N)
    curl -sf --max-time 30 "$PROXY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "$PAYLOAD" >/dev/null 2>&1 || true
    t_end=$(date +%s%N)
    miss_ms=$(( (t_end - t_start) / 1000000 ))
    pass "${miss_ms}ms"

    # Second request (cache hit)
    echo -n "  cache hit...  "
    t_start=$(date +%s%N)
    curl -sf --max-time 30 "$PROXY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "$PAYLOAD" >/dev/null 2>&1 || true
    t_end=$(date +%s%N)
    hit_ms=$(( (t_end - t_start) / 1000000 ))
    pass "${hit_ms}ms"

    if [[ $miss_ms -gt 0 ]] && [[ $hit_ms -lt $((miss_ms / 2)) ]]; then
        echo -e "  ${GREEN}Cache is working!${NC} (hit=${hit_ms}ms < miss/2=${miss_ms}ms)"
    elif [[ $hit_ms -lt 100 ]]; then
        echo -e "  ${GREEN}Cache is working!${NC} (hit=${hit_ms}ms)"
    fi
    echo ""
fi

# ── 7. tmux sessions ──────────────────────────────────────────
echo -e "  ${CYAN}tmux Sessions${NC}"
sep

echo -n "  llm-servers... "
if tmux has-session -t llm-servers 2>/dev/null; then
    windows=$(tmux list-windows -t llm-servers -F '#W' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
    pass "active (${windows})"
else
    skip "not running (start: ./scripts/start-all-tmux.sh)"
fi

echo ""

# ── Summary ────────────────────────────────────────────────────
sep
if [[ $failed -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}All checks passed${NC} (${passed}/${total})"
else
    echo -e "  ${RED}${BOLD}${failed} check(s) failed${NC} (${passed}/${total} passed)"
fi
sep
echo ""

[[ $failed -eq 0 ]] && exit 0 || exit 1
