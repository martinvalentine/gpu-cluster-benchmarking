#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}  ✓${NC} $*"; }
fail(){ echo -e "${RED}  ✗${NC} $*"; }
sep() { echo -e "${DIM}  $(printf '%.0s─' {1..50})${NC}"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Full pipeline: download models → run benchmarks → parse results.

OPTIONS:
  --skip-download         Skip model download
  --skip-bench            Skip benchmark execution
  --only-download         Only download models, no benchmark
  --phase PHASE           Benchmark phase: p1, p2, p3, all (default: all)
  --model NAME            Specific model to benchmark (default: all)
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                          # Full pipeline
  $(basename "$0") --only-download          # Download all models only
  $(basename "$0") --skip-download --phase p1  # Benchmark P1 only
  $(basename "$0") --model qwen7b-gguf      # Download + benchmark specific model
EOF
    exit 0
}

SKIP_DOWNLOAD=0
SKIP_BENCH=0
ONLY_DOWNLOAD=0
PHASE="all"
MODEL_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)  SKIP_DOWNLOAD=1; shift ;;
        --skip-bench)     SKIP_BENCH=1; shift ;;
        --only-download)  ONLY_DOWNLOAD=1; shift ;;
        --phase)          PHASE="$2"; shift 2 ;;
        --model)          MODEL_FILTER="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown: $1" >&2; usage ;;
    esac
done

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  GPU Cluster Benchmark — Full Pipeline${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""

# ── Step 1: Setup ────────────────────────────────────────────
log "Step 1: Checking environment..."
cd "$PROJECT_ROOT"

if ! command -v uv &>/dev/null; then
    fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
ok "uv found"

if ! command -v redis-cli &>/dev/null; then
    fail "redis-cli not found"
    exit 1
fi

if redis-cli ping &>/dev/null 2>&1; then
    ok "Redis running"
else
    log "Starting Redis..."
    redis-server --daemonize yes \
        --maxmemory 8gb \
        --maxmemory-policy allkeys-lru \
        --logfile /tmp/redis.log 2>/dev/null || true
    sleep 1
    redis-cli ping &>/dev/null && ok "Redis started" || fail "Redis failed to start"
fi

sep

# ── Step 2: Download models ──────────────────────────────────
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    log "Step 2: Downloading models..."
    if [[ -n "$MODEL_FILTER" ]]; then
        uv run python scripts/download-models.py --only "$MODEL_FILTER"
    else
        uv run python scripts/download-models.py --skip-existing
    fi
    ok "Models ready"
else
    log "Step 2: Skipping download (--skip-download)"
fi

sep

if [[ "$ONLY_DOWNLOAD" -eq 1 ]]; then
    echo ""
    echo -e "  ${GREEN}${BOLD}Download complete!${NC}"
    echo -e "  Models in: ${CYAN}$(uv run python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['base_dir'])")${NC}"
    echo ""
    exit 0
fi

# ── Step 3: Check running servers ────────────────────────────
log "Step 3: Checking servers..."
SERVERS_UP=0

for port in 8001 8000 8002; do
    if curl -sf "http://localhost:${port}/v1/models" &>/dev/null; then
        ok "Server on port $port"
        SERVERS_UP=$((SERVERS_UP + 1))
    fi
done

if [[ $SERVERS_UP -eq 0 ]]; then
    fail "No servers running on ports 8000-8002"
    echo ""
    echo -e "  ${YELLOW}Start a server first:${NC}"
    echo "    uv run ./scripts/run/run-llamacpp.sh    # llama-cpp-turboquant (port 8001)"
    echo "    uv run ./scripts/run/run-vllm.sh         # vLLM (port 8000)"
    echo "    uv run ./scripts/run/run-sglang.sh       # SGLang (port 8002)"
    echo ""
    exit 1
fi

sep

# ── Step 4: Run benchmarks ───────────────────────────────────
log "Step 4: Running benchmarks (phase: $PHASE)..."

# llama.cpp
if curl -sf "http://localhost:8001/v1/models" &>/dev/null; then
    log "  → llama-cpp-turboquant (port 8001)"
    uv run ./scripts/benchmark/bench-llamacpp.sh || fail "llama.cpp benchmark failed"
fi

# vLLM
if curl -sf "http://localhost:8000/v1/models" &>/dev/null; then
    log "  → vLLM (port 8000)"
    uv run ./scripts/benchmark/bench-vllm.sh -p "$PHASE" || fail "vLLM benchmark failed"
fi

# SGLang
if curl -sf "http://localhost:8002/v1/models" &>/dev/null; then
    log "  → SGLang (port 8002)"
    uv run ./scripts/benchmark/bench-sglang.sh -p "$PHASE" || fail "SGLang benchmark failed"
fi

sep

# ── Step 5: Generate report ─────────────────────────────────
if [[ "$SKIP_BENCH" -eq 0 ]]; then
    log "Step 5: Generating report..."
    uv run python scripts/report.py --results-dir results/ 2>&1 | tee -a "$LOG_FILE"
fi

sep

# ── Summary ──────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Pipeline complete!${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$ONLY_DOWNLOAD" -eq 1 ]]; then
    echo "  Downloaded models in: $(python3 -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['base_dir'])" 2>/dev/null || echo '/workspace/models')"
else
    echo "  Results: results/"
    ls results/*.csv 2>/dev/null || echo "  (no CSV results yet)"
fi
echo ""
