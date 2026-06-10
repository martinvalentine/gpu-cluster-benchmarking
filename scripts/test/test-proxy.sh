#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

PROXY_URL="${LITELLM_PROXY_URL:-http://localhost:4000}"
MODEL="${LITELLM_MODEL:-qwen35b-llamacpp}"
TIMEOUT=60
MAX_TOKENS=64

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }

echo "=== LiteLLM Proxy Test ==="
echo "  Proxy:  $PROXY_URL"
echo "  Model:  $MODEL"
echo ""

passed=0 total=0

# 1. Health check
echo -n "1. Health check... "
total=$((total + 1))
if curl -sf --max-time 5 "$PROXY_URL/health" >/dev/null 2>&1; then
    pass "proxy reachable"
    passed=$((passed + 1))
else
    fail "proxy unreachable at $PROXY_URL"
    echo "   Start proxy: ./scripts/run/run-proxy.sh"
    echo ""
    echo "=== Results: $passed/$total passed ==="
    exit 1
fi

# 2. Models endpoint
echo -n "2. Models endpoint... "
total=$((total + 1))
resp=$(curl -sf --max-time 5 "$PROXY_URL/v1/models" 2>/dev/null)
if [[ $? -eq 0 ]]; then
    models=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('data', []):
    print(f\"    {m['id']}\")
" 2>/dev/null)
    pass "models listed"
    echo -e "  ${DIM}$models${NC}"
    passed=$((passed + 1))
else
    fail "/v1/models failed"
fi

# 3. Chat completion (first request — cache miss)
echo -n "3. Chat completion (cache miss)... "
total=$((total + 1))
payload=$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': 'What is 2+2? Answer in one word.'}],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0
}))")

resp_file=$(mktemp /tmp/proxy_test_XXXXXX.json)
http_code=$(curl -sf --max-time "$TIMEOUT" -o "$resp_file" -w '%{http_code}' \
    -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null) || true

if [[ "$http_code" == "200" ]]; then
    content=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
text = d['choices'][0]['message']['content']
usage = d.get('usage', {})
print(f'    Response: {text[:200]}')
print(f'    Tokens:   {usage.get(\"prompt_tokens\",\"?\")}/{usage.get(\"completion_tokens\",\"?\")}/{usage.get(\"total_tokens\",\"?\")}')
" "$resp_file" 2>/dev/null)
    pass "HTTP 200"
    echo -e "  ${CYAN}$content${NC}"
    passed=$((passed + 1))
else
    fail "HTTP ${http_code:-timeout}"
    cat "$resp_file" 2>/dev/null | head -3 >&2
fi
rm -f "$resp_file"

# 4. Same request again — test cache hit
echo -n "4. Chat completion (cache hit)... "
total=$((total + 1))
resp_file2=$(mktemp /tmp/proxy_test_XXXXXX.json)
start_time=$(date +%s%N)
http_code2=$(curl -sf --max-time "$TIMEOUT" -o "$resp_file2" -w '%{http_code}' \
    -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null) || true
end_time=$(date +%s%N)
elapsed_ms=$(( (end_time - start_time) / 1000000 ))

if [[ "$http_code2" == "200" ]]; then
    pass "HTTP 200 (${elapsed_ms}ms)"
    passed=$((passed + 1))
else
    fail "HTTP ${http_code2:-timeout}"
fi
rm -f "$resp_file2"

# 5. Unique request (cache miss path)
echo -n "5. Unique request (no cache)... "
total=$((total + 1))
unique_payload=$(python3 -c "
import json, uuid
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': f'Unique test {uuid.uuid4()}'}],
    'max_tokens': 16,
    'temperature': 0
}))")
resp_file3=$(mktemp /tmp/proxy_test_XXXXXX.json)
http_code3=$(curl -sf --max-time "$TIMEOUT" -o "$resp_file3" -w '%{http_code}' \
    -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$unique_payload" 2>/dev/null) || true

if [[ "$http_code3" == "200" ]]; then
    pass "HTTP 200"
    passed=$((passed + 1))
else
    fail "HTTP ${http_code3:-timeout}"
fi
rm -f "$resp_file3"

echo ""
echo "=== Results: $passed/$total passed ==="
[[ $passed -eq $total ]] && exit 0 || exit 1
