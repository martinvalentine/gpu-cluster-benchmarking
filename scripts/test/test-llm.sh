#!/usr/bin/env bash
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [PROMPT]

Test LLM serving endpoints (health, models, chat completion).

OPTIONS:
  -s, --server NAME     Test specific server: vllm, llamacpp, sglang, llamacpp-turbo (default: all)
  -t, --timeout SEC     Completion timeout in seconds (default: 60)
  -m, --max-tokens N    Max tokens for completion (default: 64)
  -h, --help            Show this help

EXAMPLES:
  $(basename "$0")                                    # Test all 3, default prompt
  $(basename "$0") -s llamacpp "What is 2+2?"         # Test only llamacpp
  $(basename "$0") -s vllm -t 120 -m 128 "Explain TCP"
  $(basename "$0") --server sglang --max-tokens 256

SERVERS:
  vllm            http://localhost:8000
  llamacpp        http://localhost:8001
  sglang          http://localhost:8002
  llamacpp-turbo  http://localhost:8003
EOF
    exit 0
}

SERVER_FILTER=""
TIMEOUT=60
MAX_TOKENS=64
PROMPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--server)   SERVER_FILTER="$2"; shift 2 ;;
        -t|--timeout)  TIMEOUT="$2"; shift 2 ;;
        -m|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        -h|--help)     usage ;;
        -*)            echo "Unknown option: $1" >&2; usage ;;
        *)             PROMPT="$1"; shift ;;
    esac
done

PROMPT="${PROMPT:-Hello, respond with one word.}"

declare -A SERVERS=(
    [vllm]="http://localhost:8000"
    [llamacpp]="http://localhost:8001"
    [sglang]="http://localhost:8002"
    [llamacpp-turbo]="http://localhost:8003"
)

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }

test_health() {
    local name=$1 url=$2
    curl -sf --max-time 5 "$url/health" >/dev/null 2>&1 || { fail "health unreachable"; return 1; }
    pass "health OK"
}

test_models() {
    local name=$1 url=$2
    local resp
    resp=$(curl -sf --max-time 5 "$url/v1/models" 2>/dev/null) || { fail "/v1/models unreachable"; return 1; }
    local model_id
    model_id=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null) || { fail "parse models failed"; return 1; }
    pass "model: $model_id"
}

test_completion() {
    local name=$1 url=$2
    local payload resp_file http_code content

    resp_file=$(mktemp /tmp/llm_test_XXXXXX.json)
    trap "rm -f '$resp_file'" RETURN

    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'default',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
    'temperature': 0
}))" "$PROMPT" "$MAX_TOKENS")

    http_code=$(curl -sf --max-time "$TIMEOUT" -o "$resp_file" -w '%{http_code}' \
        -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null) || { fail "completion timeout (${TIMEOUT}s)"; return 1; }

    if [[ "$http_code" != "200" ]]; then
        fail "completion HTTP $http_code"
        cat "$resp_file" 2>/dev/null | head -5 >&2
        return 1
    fi

    content=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
msg = d['choices'][0]['message']
text = msg.get('content', '') or msg.get('reasoning_content', '')
usage = d.get('usage', {})
print(f\"Response: {text[:500]}\")
if usage:
    print(f\"Tokens:   prompt={usage.get('prompt_tokens','?')} completion={usage.get('completion_tokens','?')} total={usage.get('total_tokens','?')}\")
" "$resp_file" 2>/dev/null) || { fail "parse completion failed"; return 1; }

    pass "completion OK"
    echo -e "  ${CYAN}${content}${NC}"
}

echo "=== LLM Connection Test ==="
echo "Prompt:   \"$PROMPT\""
echo "Timeout:  ${TIMEOUT}s"
echo "Max tok:  ${MAX_TOKENS}"
echo ""

targets=()
if [[ -n "$SERVER_FILTER" ]]; then
    if [[ -z "${SERVERS[$SERVER_FILTER]+x}" ]]; then
        echo "ERROR: Unknown server '$SERVER_FILTER'. Choose: vllm, llamacpp, sglang, llamacpp-turbo" >&2
        exit 1
    fi
    targets=("$SERVER_FILTER")
else
    targets=(vllm llamacpp sglang llamacpp-turbo)
fi

total=0 passed=0

for name in "${targets[@]}"; do
    url="${SERVERS[$name]}"
    echo "[$name] $url"
    total=$((total + 3))

    if test_health "$name" "$url"; then passed=$((passed + 1)); fi
    if test_models "$name" "$url"; then passed=$((passed + 1)); fi
    if test_completion "$name" "$url"; then passed=$((passed + 1)); fi
    echo ""
done

echo "=== Results: $passed/$total passed ==="
[[ $passed -eq $total ]] && exit 0 || exit 1
