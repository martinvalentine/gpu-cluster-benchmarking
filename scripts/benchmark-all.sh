#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark all enabled models from configs/models.yaml.

Delegates to bench-models.sh which loops over each model:
  1. Start server → benchmark → stop → next model
  2. Results saved to: results/run-N/{vllm,llamacpp}/

OPTIONS:
  -o, --output DIR        Results root directory (default: ./results)
  -p, --phase PHASE       Run specific phase: p0, p1, p2, p3, all (default: all)
  -m, --model NAME        Run specific model (can repeat)
  --skip-health-check     Skip pre-flight health checks
  -y, --yes               Auto-accept prompts
  --dry-run               Preview without executing
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                    # All enabled models
  $(basename "$0") -p p0                              # P0 phase only
  $(basename "$0") -m qwen32b-awq                     # Specific model
  $(basename "$0") --dry-run                          # Preview

NOTE: This script delegates to bench-models.sh which manages server
lifecycle (start/bench/stop) per model. For manual server management,
use bench-vllm.sh / bench-llamacpp.sh directly.
EOF
    exit 0
}

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)            EXTRA_ARGS+=("-o" "$2"); shift 2 ;;
        -p|--phase)             EXTRA_ARGS+=("-p" "$2"); shift 2 ;;
        -m|--model)             EXTRA_ARGS+=("-m" "$2"); shift 2 ;;
        --skip-health-check)    EXTRA_ARGS+=("--skip-health-check"); shift ;;
        -y|--yes)               EXTRA_ARGS+=("--yes"); shift ;;
        --dry-run)              EXTRA_ARGS+=("--dry-run"); shift ;;
        -h|--help)              usage ;;
        *)                      echo "Unknown: $1" >&2; usage ;;
    esac
done

echo ""
sep
echo -e "${BOLD}  Benchmark Suite — configs/models.yaml${NC}"
echo -e "  $(date '+%Y-%m-%d %H:%M:%S')"
sep
echo ""

# Delegate to bench-models.sh (reads YAML config, loops models, manages servers)
exec bash "${PROJECT_ROOT}/scripts/bench-models.sh" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
