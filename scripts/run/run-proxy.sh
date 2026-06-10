#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

DEFAULT_CONFIG="${PROJECT_ROOT}/litellm_config.yaml"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start LiteLLM proxy with Redis-backed semantic cache.

OPTIONS:
  -c, --config PATH       LiteLLM config YAML (default: litellm_config.yaml)
  -p, --port PORT         Proxy port (default: 4000)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -d, --detailed-debug    Enable detailed debug logging
  -h, --help              Show this help

ENV OVERRIDES (lower priority than flags):
  LITELLM_PORT, LITELLM_HOST, LITELLM_CONFIG

PREREQUISITES:
  - Redis running on localhost:6379 (redis-cli ping → PONG)
  - At least one serving engine running (vLLM :8000, llama.cpp :8001, SGLang :8002)

EXAMPLES:
  $(basename "$0")                                    # Defaults
  $(basename "$0") -p 4000 -d                        # Debug mode
  $(basename "$0") -c /path/to/custom_config.yaml    # Custom config
EOF
    exit 0
}

CONFIG="${LITELLM_CONFIG:-$DEFAULT_CONFIG}"
PORT="${LITELLM_PORT:-4000}"
HOST="${LITELLM_HOST:-0.0.0.0}"
DEBUG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)        CONFIG="$2"; shift 2 ;;
        -p|--port)          PORT="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -d|--detailed-debug) DEBUG="--detailed_debug"; shift ;;
        -h|--help)          usage ;;
        -*)                 echo "Unknown option: $1" >&2; usage ;;
        *)                  echo "Unexpected argument: $1" >&2; usage ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

if ! command -v litellm &>/dev/null; then
    echo "ERROR: litellm not installed. Run: pip install 'litellm[proxy]>=1.40.0'" >&2
    exit 1
fi

# Check Redis connectivity
if command -v redis-cli &>/dev/null; then
    if ! redis-cli ping &>/dev/null; then
        echo "WARNING: Redis not reachable on localhost:6379 — semantic cache will fail" >&2
    fi
fi

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

row() { printf "  ${CYAN}%-16s${NC} %s\n" "$1" "$2"; }
sep() { echo -e "  ${DIM}$(printf '%.0s─' {1..48})${NC}"; }

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  LiteLLM Proxy${NC}${DIM} — Semantic Cache Gateway${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Config"       "$CONFIG"
row "Endpoint"     "http://${HOST}:${PORT}"
sep
row "Cache"        "Redis Semantic (cosine ≥ 0.95)"
row "TTL"          "3600s"
row "Router"       "least-busy"
row "Callbacks"    "prometheus"
sep
row "Backends"     "vLLM :8000 | llama.cpp :8001 | SGLang :8002"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

exec litellm \
    --config "$CONFIG" \
    --port "$PORT" \
    --host "$HOST" \
    $DEBUG
