#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }
sep()  { echo -e "${DIM}  $(printf '%.0s─' {1..55})${NC}"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [COMMAND] [OPTIONS]

Master orchestrator for the GPU cluster benchmark pipeline.

COMMANDS:
    setup       Bootstrap a fresh machine (install deps, download models, start servers)
    download    Download models (respects configs/models.yaml)
    start       Start all servers in tmux
    stop        Stop all servers
    test        Run health checks on all services
    benchmark   Run benchmarks and generate report
    status      Show current system status
    full        Full pipeline: download → start → test → benchmark

OPTIONS:
    --phase PHASE       Benchmark phase: p0, p1, p2, p3, all (default: all)
    --model NAME        Specific model name to download (can repeat)
    --skip-download     Skip model download step
    --skip-bench        Skip benchmark execution
    --only-download     Only download models, exit after
    --results-dir DIR   Custom results directory (default: results/)
    -h, --help          Show this help

EXAMPLES:
    $(basename "$0") setup                              # Fresh machine setup
    $(basename "$0") download                           # Download enabled models
    $(basename "$0") download --model qwen0.5b          # Download specific model
    $(basename "$0") start                              # Start all servers
    $(basename "$0") test                               # Health check
    $(basename "$0") benchmark                          # Run benchmarks
    $(basename "$0") full                               # Everything
    $(basename "$0") full --phase p0 --skip-download    # Benchmark P0 only
EOF
    exit 0
}

# ── Parse args ────────────────────────────────────────────────
COMMAND="${1:-}"
shift 2>/dev/null || true

# Handle --help/-h as first arg
if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
    usage
fi

PHASE="all"
MODELS=()
SKIP_DOWNLOAD=0
SKIP_BENCH=0
ONLY_DOWNLOAD=0
RESULTS_DIR="${PROJECT_ROOT}/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)          PHASE="$2"; shift 2 ;;
        --model)          MODELS+=("$2"); shift 2 ;;
        --skip-download)  SKIP_DOWNLOAD=1; shift ;;
        --skip-bench)     SKIP_BENCH=1; shift ;;
        --only-download)  ONLY_DOWNLOAD=1; shift ;;
        --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown: $1" >&2; usage ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    usage
fi

cd "$PROJECT_ROOT"

# ── Shared dependency helpers ───────────────────────────────
ensure_uv() {
    if ! command -v uv &>/dev/null; then
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    ok "uv $(uv --version 2>/dev/null || echo 'installed')"
}

ensure_redis() {
    if redis-cli ping &>/dev/null 2>&1; then
        ok "Redis already running"
    else
        redis-server --daemonize yes --maxmemory 8gb --maxmemory-policy allkeys-lru --logfile /tmp/redis.log 2>/dev/null
        sleep 1
        redis-cli ping &>/dev/null 2>&1 && ok "Redis started" || fail "Redis failed"
    fi
}

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  GPU Cluster Benchmark — $(echo "$COMMAND" | tr '[:lower:]' '[:upper:]')${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""

# ── SETUP ─────────────────────────────────────────────────────
do_setup() {
    log "Step 1: Checking system dependencies..."

    # uv
    ensure_uv

    # Redis
    if ! command -v redis-cli &>/dev/null; then
        log "Installing Redis..."
        sudo apt-get update -y && sudo apt-get install -y redis-server
    fi
    ok "redis-cli found"

    # tmux
    if ! command -v tmux &>/dev/null; then
        log "Installing tmux..."
        sudo apt-get install -y tmux
    fi
    ok "tmux found"

    # GPU
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        ok "${gpu_count}x ${gpu_name}"
    else
        warn "nvidia-smi not found (no GPU?)"
    fi

    sep

    log "Step 2: Installing Python dependencies..."
    uv sync --group all --group vllm 2>&1 | tail -3
    ok "Python dependencies installed"

    sep

    log "Step 3: Starting Redis..."
    ensure_redis

    sep

    log "Step 4: Downloading models..."
    uv run python scripts/download-models.py --skip-existing
    ok "Models ready"

    sep

    log "Step 5: Starting servers..."
    bash scripts/start-all-tmux.sh

    sep

    log "Step 6: Waiting for services..."
    sleep 15

    log "Step 7: Running health checks..."
    bash scripts/test-all.sh
}

# ── DOWNLOAD ──────────────────────────────────────────────────
do_download() {
    local args=()
    args+=("--skip-existing")

    if [[ ${#MODELS[@]} -gt 0 ]]; then
        args+=("--only" "${MODELS[@]}")
    fi

    if [[ "$PHASE" != "all" ]]; then
        args+=("--phase" "$PHASE")
    fi

    uv run python scripts/download-models.py "${args[@]}"
}

# ── START ─────────────────────────────────────────────────────
do_start() {
    bash scripts/start-all-tmux.sh
}

# ── STOP ──────────────────────────────────────────────────────
do_stop() {
    bash scripts/stop-all.sh
}

# ── TEST ──────────────────────────────────────────────────────
do_test() {
    bash scripts/test-all.sh
}

# ── BENCHMARK ─────────────────────────────────────────────────
do_benchmark() {
    mkdir -p "$RESULTS_DIR"/{vllm,sglang,llamacpp,litellm}

    # Build args
    local args=()
    if [[ "$PHASE" != "all" ]]; then
        args+=("-p" "$PHASE")
    fi
    if [[ ${#MODELS[@]} -gt 0 ]]; then
        for m in "${MODELS[@]}"; do
            args+=("-m" "$m")
        done
    fi

    # Run per-model benchmarks (manages server lifecycle per model)
    bash scripts/bench-models.sh "${args[@]+"${args[@]}"}"

    # Generate report
    log "Generating report..."
    uv run python scripts/report.py --results-dir "$RESULTS_DIR"
}

# ── STATUS ────────────────────────────────────────────────────
do_status() {
    echo -e "  ${CYAN}System${NC}"
    sep

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null | while IFS=', ' read -r idx name used total util; do
            echo -e "  GPU $idx: $name | $used/$total | $util"
        done
    else
        echo -e "  ${DIM}No GPU detected${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Servers${NC}"
    sep

    for port in 8000 8001 8002 8003 4000; do
        local name=""
        case $port in
            8000) name="vLLM" ;;
            8001) name="llama.cpp" ;;
            8002) name="SGLang" ;;
            8003) name="Embedding" ;;
            4000) name="LiteLLM" ;;
        esac
        if curl -sf --max-time 3 "http://localhost:${port}/v1/models" &>/dev/null 2>&1 || \
           curl -sf --max-time 3 "http://localhost:${port}/health" &>/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} ${name} (:${port})"
        else
            echo -e "  ${DIM}○ ${name} (:${port}) not running${NC}"
        fi
    done

    echo ""
    echo -e "  ${CYAN}Redis${NC}"
    sep

    if redis-cli ping &>/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Redis running"
    else
        echo -e "  ${DIM}○ Redis not running${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}tmux${NC}"
    sep

    if tmux has-session -t llm-servers 2>/dev/null; then
        windows=$(tmux list-windows -t llm-servers -F '#W' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        echo -e "  ${GREEN}✓${NC} llm-servers session: ${windows}"
    else
        echo -e "  ${DIM}○ No llm-servers session${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Models${NC}"
    sep

    if [[ -d "${PROJECT_ROOT}/models" ]]; then
        find "${PROJECT_ROOT}/models" -maxdepth 3 -type f \( -name "*.gguf" -o -name "*.safetensors" \) 2>/dev/null | while read -r f; do
            local size
            size=$(du -sh "$f" 2>/dev/null | cut -f1)
            echo -e "  ${DIM}$(basename "$f")${NC} ($size)"
        done
    else
        echo -e "  ${DIM}No models directory${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Results${NC}"
    sep

    if [[ -d "$RESULTS_DIR" ]]; then
        local count
        count=$(find "$RESULTS_DIR" -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.tsv" \) 2>/dev/null | wc -l)
        echo -e "  ${count} result files in ${RESULTS_DIR}/"
        if [[ -f "$RESULTS_DIR/report.md" ]]; then
            echo -e "  ${GREEN}✓${NC} report.md exists"
        fi
    else
        echo -e "  ${DIM}No results yet${NC}"
    fi

    echo ""
}

# ── FULL PIPELINE ─────────────────────────────────────────────
do_full() {
    log "Step 1: System setup & dependencies..."
    ensure_uv
    ensure_redis

    sep

    # Download models
    if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
        log "Step 2: Downloading models..."
        do_download
        ok "Models ready"
    else
        log "Step 2: Skipping download"
    fi

    # Only download mode
    if [[ "$ONLY_DOWNLOAD" -eq 1 ]]; then
        echo ""
        ok "Download complete! Models in: ${PROJECT_ROOT}/models/"
        echo ""
        exit 0
    fi

    sep

    # Start servers (start-all-tmux.sh generates litellm_config.yaml internally)
    log "Step 3: Starting servers..."
    do_start

    sep

    # Wait for services
    log "Step 4: Waiting for services to initialize..."
    sleep 20

    # Health check
    log "Step 5: Health check..."
    bash scripts/test-all.sh || warn "Some checks failed"

    sep

    # Benchmark
    if [[ "$SKIP_BENCH" -eq 0 ]]; then
        log "Step 6: Running benchmarks..."
        do_benchmark
    else
        log "Step 6: Skipping benchmarks"
    fi

    sep

    echo ""
    echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
    echo -e "  ${GREEN}${BOLD}  Pipeline complete!${NC}"
    echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Results: ${CYAN}${RESULTS_DIR}/${NC}"
    echo -e "  Logs:    ${CYAN}/tmp/{vllm,llama,embed,proxy}.log${NC}"
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────
case "$COMMAND" in
    setup)     do_setup ;;
    download)  do_download ;;
    start)     do_start ;;
    stop)      do_stop ;;
    test)      do_test ;;
    benchmark) do_benchmark ;;
    status)    do_status ;;
    full)      do_full ;;
    *)         echo "Unknown command: $COMMAND" >&2; usage ;;
esac
