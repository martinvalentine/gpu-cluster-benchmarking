#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="${BENCH_RESULTS_DIR:-${PROJECT_ROOT}/results}"
LOG_FILE="${RESULTS_DIR}/benchmark_run_$(date +%Y%m%d_%H%M%S).log"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Master benchmark runner — executes all frameworks sequentially (benchmark_plan.md F.1).
Runs vLLM, SGLang, llama.cpp, and LiteLLM benchmarks across all phases.

OPTIONS:
  -o, --output DIR        Results root directory (default: ./results)
  -f, --framework FW      Run specific framework: vllm, sglang, llamacpp, litellm, all (default: all)
  -p, --phase PHASE       Run specific phase: p1, p2, p3, all (default: all)
  --use-llama-benchy      Use new llama-benchy based scripts instead of native benchmark tools
  --skip-health-check     Skip pre-flight health checks
  -y, --yes               Auto-accept prompts (for CI / non-interactive use)
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                    # Everything
  $(basename "$0") -f vllm -p p1                     # vLLM P1 only
  $(basename "$0") -f sglang -f llamacpp              # SGLang + llama.cpp
  $(basename "$0") --use-llama-benchy                 # Use llama-benchy for all
  $(basename "$0") --skip-health-check                # Skip connectivity checks
EOF
    exit 0
}

FRAMEWORK="all"
PHASE="all"
SKIP_HEALTH=0
AUTO_YES=0
USE_LLAMA_BENCHY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)            RESULTS_DIR="$2"; shift 2 ;;
        -f|--framework)         FRAMEWORK="$2"; shift 2 ;;
        -p|--phase)             PHASE="$2"; shift 2 ;;
        --use-llama-benchy)     USE_LLAMA_BENCHY=1; shift ;;
        --skip-health-check)    SKIP_HEALTH=1; shift ;;
        -y|--yes)               AUTO_YES=1; shift ;;
        -h|--help)              usage ;;
        *)                      echo "Unknown: $1" >&2; usage ;;
    esac
done

mkdir -p "$RESULTS_DIR"/{vllm,sglang,llamacpp,litellm}

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
header() { echo -e "\n${GREEN}${BOLD}═══ $* ═══${NC}\n" | tee -a "$LOG_FILE"; }

check_health() {
    local name=$1 url=$2
    if curl -sf "$url" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $name reachable at $url"
        return 0
    else
        echo -e "  ${RED}✗${NC} $name NOT reachable at $url"
        return 1
    fi
}

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  GPU Cluster Benchmark — Master Runner${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Framework${NC}    $FRAMEWORK"
echo -e "  ${CYAN}Phase${NC}        $PHASE"
echo -e "  ${CYAN}Results${NC}      $RESULTS_DIR"
echo -e "  ${CYAN}Log${NC}          $LOG_FILE"
echo ""

if [[ "$SKIP_HEALTH" -eq 0 ]]; then
    echo -e "  ${DIM}Pre-flight health checks...${NC}"
    FAILED=0
    check_health "vLLM"    "http://localhost:8000/v1/models"    || FAILED=$((FAILED+1))
    check_health "llama.cpp" "http://localhost:8001/v1/models"  || FAILED=$((FAILED+1))
    check_health "Embedding" "http://localhost:8003/health"     || FAILED=$((FAILED+1))
    check_health "SGLang"  "http://localhost:8002/v1/models"    || FAILED=$((FAILED+1))
    check_health "LiteLLM" "http://localhost:4000/health"       || FAILED=$((FAILED+1))
    check_health "Redis"   "http://localhost:6379"              || FAILED=$((FAILED+1))

    if [[ $FAILED -gt 0 ]]; then
        echo ""
        echo -e "  ${YELLOW}WARNING: $FAILED service(s) not reachable.${NC}"
        if [[ $AUTO_YES -eq 0 && -t 0 ]]; then
            read -rp "  Continue anyway? [y/N] " yn
            [[ "$yn" =~ ^[Yy] ]] || exit 1
        fi
    fi
    echo ""
fi

log "Benchmark run started: framework=$FRAMEWORK phase=$PHASE"

if [[ "$FRAMEWORK" == "all" || "$FRAMEWORK" == "vllm" ]]; then
    header "vLLM Benchmarks"
    if [[ "$USE_LLAMA_BENCHY" -eq 1 ]]; then
        "${PROJECT_ROOT}/scripts/benchmark/bench.sh" -b vllm \
            2>&1 | tee -a "$LOG_FILE"
    else
        "${PROJECT_ROOT}/scripts/benchmark/vllm_bench.sh" -o "$RESULTS_DIR/vllm" \
            2>&1 | tee -a "$LOG_FILE"
    fi
fi

if [[ "$FRAMEWORK" == "all" || "$FRAMEWORK" == "sglang" ]]; then
    header "SGLang Benchmarks"
    if [[ "$USE_LLAMA_BENCHY" -eq 1 ]]; then
        "${PROJECT_ROOT}/scripts/benchmark/bench.sh" -b sglang \
            2>&1 | tee -a "$LOG_FILE"
    else
        "${PROJECT_ROOT}/scripts/benchmark/sglang_bench.sh" -o "$RESULTS_DIR/sglang" \
            2>&1 | tee -a "$LOG_FILE"
    fi
fi

if [[ "$FRAMEWORK" == "all" || "$FRAMEWORK" == "llamacpp" ]]; then
    header "llama.cpp Benchmarks"
    if [[ "$USE_LLAMA_BENCHY" -eq 1 ]]; then
        "${PROJECT_ROOT}/scripts/benchmark/bench.sh" -b llamacpp \
            2>&1 | tee -a "$LOG_FILE"
    else
        "${PROJECT_ROOT}/scripts/benchmark/llamacpp_bench.sh" -o "$RESULTS_DIR/llamacpp" \
            2>&1 | tee -a "$LOG_FILE"
    fi
fi

if [[ "$FRAMEWORK" == "all" || "$FRAMEWORK" == "litellm" ]]; then
    header "LiteLLM Proxy Benchmarks"
    LITELLM_RESULTS_DIR="$RESULTS_DIR/litellm" \
    "${PROJECT_ROOT}/scripts/benchmark/bench-litellm.sh" \
        2>&1 | tee -a "$LOG_FILE"
fi

header "GPU Summary"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  All benchmarks complete!${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Results: ${CYAN}$RESULTS_DIR/${NC}"
echo -e "  Log:     ${CYAN}$LOG_FILE${NC}"
echo ""

if [[ -f "${PROJECT_ROOT}/scripts/parse_bench.py" ]]; then
    log "Parsing results..."
    python3 "${PROJECT_ROOT}/scripts/parse_bench.py" "$RESULTS_DIR" --all \
        2>&1 | tee -a "$LOG_FILE"
fi
