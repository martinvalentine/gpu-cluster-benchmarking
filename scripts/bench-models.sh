#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="${BENCH_RESULTS_DIR:-${PROJECT_ROOT}/results}"
YAML_CONFIG="${PROJECT_ROOT}/configs/models.yaml"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Per-model benchmark orchestrator — run-based structure.

For each enabled model:
  1. Start vLLM with that model → bench → save → stop
  2. Start llama.cpp with that model → bench → save → stop
  3. Results saved to: results/run-N/{vllm,llamacpp}/

OPTIONS:
  -o, --output DIR        Results root directory (default: ./results)
  -p, --phase PHASE       Run specific phase: p0, p1, p2, p3, all (default: all)
  -m, --model NAME        Run specific model by name (can repeat)
  --skip-health-check     Skip pre-flight health checks
  -y, --yes               Auto-accept prompts (for CI / non-interactive use)
  --dry-run               Show what would be run without executing
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                    # All enabled models
  $(basename "$0") -p p0                              # P0 phase models only
  $(basename "$0") -m qwen0.5b                        # Specific model (all backends)
  $(basename "$0") --dry-run                          # Preview actions

OUTPUT STRUCTURE:
  results/
    run-1/
      vllm/                     # vLLM benchmark results
        p1_light_conc1.json
        p1_light_conc32.json
      llamacpp/                 # llama.cpp benchmark results
        p1_light.json
    run-2/
      vllm/
        ...
      llamacpp/
        ...
    summary.csv                 # Aggregated results
EOF
    exit 0
}

PHASE="all"
AUTO_YES=0
SKIP_HEALTH=0
DRY_RUN=0
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)            RESULTS_DIR="$2"; shift 2 ;;
        -p|--phase)             PHASE="$2"; shift 2 ;;
        -m|--model)             SELECTED_MODELS+=("$2"); shift 2 ;;
        --skip-health-check)    SKIP_HEALTH=1; shift ;;
        -y|--yes)               AUTO_YES=1; shift ;;
        --dry-run)              DRY_RUN=1; shift ;;
        -h|--help)              usage ;;
        *)                      echo "Unknown: $1" >&2; usage ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
header() { echo -e "\n${GREEN}${BOLD}═══ $* ═══${NC}\n" | tee -a "$LOG_FILE"; }
ok()     { echo -e "  ${GREEN}✓${NC} $*"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $*"; }
fail()   { echo -e "  ${RED}✗${NC} $*"; }

# ── Parse models from YAML ──────────────────────────────────────
parse_models() {
    local yaml="$1" phase="$2"
    shift 2
    local selected=("$@")

    uv run python - "$yaml" "$phase" "${selected[@]+"${selected[@]}"}" <<'PYEOF'
import sys, yaml
from pathlib import Path

yaml_path = sys.argv[1]
phase_filter = sys.argv[2]
selected = sys.argv[3:] if len(sys.argv) > 3 else []

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

base_dir = cfg.get("base_dir", "models")
models = cfg.get("models", [])

for m in models:
    if not m.get("enabled", True):
        continue
    name = m.get("name", "")
    backend = m.get("backend", "vllm")
    local_dir = m.get("local_dir", "")
    m_phase = m.get("phase", "")
    include = m.get("include", "*.gguf")

    if phase_filter != "all" and m_phase != phase_filter:
        continue
    if m_phase == "embedding":
        continue
    if selected and name not in selected:
        continue

    if backend == "vllm":
        model_path = f"{base_dir}/{local_dir}"
    else:
        gguf_dir = Path(base_dir) / local_dir
        if gguf_dir.exists():
            from fnmatch import fnmatch
            matches = sorted(f for f in gguf_dir.iterdir() if fnmatch(f.name, include))
            model_path = str(matches[0]) if matches else str(gguf_dir / f"{Path(local_dir).name}.gguf")
        else:
            model_path = str(gguf_dir / f"{Path(local_dir).name}.gguf")

    print(f"{name}|{backend}|{model_path}|{m_phase}")
PYEOF
}

# ── Wait for server ─────────────────────────────────────────────
wait_for_server() {
    local name=$1 url=$2 timeout=${3:-60}
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

# ── Kill server on port ─────────────────────────────────────────
kill_server_on_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# ── Start vLLM server ───────────────────────────────────────────
start_vllm() {
    local model_path=$1 port=$2
    log "Starting vLLM: model=$model_path port=$port"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would start: vllm serve $model_path --port $port"
        return 0
    fi

    kill_server_on_port "$port"

    tmux new-session -d -s "bench-vllm" -n "serve" \
        "cd $PROJECT_ROOT && .venv/bin/vllm serve $model_path \
            --host 0.0.0.0 --port $port \
            --tensor-parallel-size ${VLLM_TP:-1} \
            --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.15} \
            --max-model-len ${VLLM_MAX_MODEL_LEN:-4096} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS:-64} \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --enforce-eager 2>&1 | tee /tmp/bench_vllm.log; sleep infinity"

    log "  Waiting for vLLM to be ready..."
    if wait_for_server "vLLM" "http://localhost:$port/v1/models" 120; then
        ok "vLLM ready on port $port"
        return 0
    else
        fail "vLLM failed to start on port $port"
        return 1
    fi
}

# ── Start llama.cpp server ──────────────────────────────────────
start_llamacpp() {
    local model_path=$1 port=$2
    log "Starting llama.cpp: model=$model_path port=$port"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would start: llama-server $model_path --port $port"
        return 0
    fi

    kill_server_on_port "$port"

    local llama_bin="${PROJECT_ROOT}/third_party/llama-cpp-turboquant/build/bin/llama-server"
    if [[ ! -f "$llama_bin" ]]; then
        llama_bin=$(command -v llama-server 2>/dev/null || true)
    fi
    if [[ -z "$llama_bin" || ! -f "$llama_bin" ]]; then
        fail "llama-server binary not found"
        return 1
    fi

    tmux new-session -d -s "bench-llama" -n "serve" \
        "cd $PROJECT_ROOT && $llama_bin \
            -m $model_path \
            --host 0.0.0.0 --port $port \
            -n 4 -c 4096 -ng all \
            2>&1 | tee /tmp/bench_llama.log; sleep infinity"

    log "  Waiting for llama.cpp to be ready..."
    if wait_for_server "llama.cpp" "http://localhost:$port/health" 120; then
        ok "llama.cpp ready on port $port"
        return 0
    else
        fail "llama.cpp failed to start on port $port"
        return 1
    fi
}

# ── Stop server ─────────────────────────────────────────────────
stop_server() {
    local fw=$1 port=$2
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would stop $fw on port $port"
        return 0
    fi

    log "  Stopping $fw on port $port..."
    kill_server_on_port "$port"

    local session="bench-${fw}"
    tmux kill-session -t "$session" 2>/dev/null || true
    ok "$fw stopped"
}

# ── Run benchmark ───────────────────────────────────────────────
run_benchmark() {
    local fw=$1 model_name=$2 model_path=$3 port=$4 results_dir=$5
    local bench_script="${PROJECT_ROOT}/scripts/benchmark/bench-${fw}.sh"

    if [[ ! -f "$bench_script" ]]; then
        warn "Benchmark script not found: $bench_script"
        return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would run: $bench_script → $results_dir/"
        return 0
    fi

    log "  Running $fw benchmark for $model_name..."

    mkdir -p "$results_dir"

    if [[ "$fw" == "vllm" ]]; then
        VLLM_RESULTS_DIR="$results_dir" \
        VLLM_BENCH_URL="http://localhost:$port" \
        VLLM_BENCH_MODEL="$model_path" \
        "$bench_script" -p "$PHASE" 2>&1 | tee -a "$LOG_FILE"
    elif [[ "$fw" == "llamacpp" ]]; then
        LLAMA_RESULTS_DIR="$results_dir" \
        LITELLM_BENCH_URL="http://localhost:$port" \
        LITELLM_BENCH_MODEL="$model_name" \
        "$bench_script" -p "$PHASE" 2>&1 | tee -a "$LOG_FILE"
    fi
}

# ── Main ────────────────────────────────────────────────────────
LOG_FILE="${RESULTS_DIR}/bench_models_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RESULTS_DIR"

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Per-Model Benchmark Orchestrator${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Phase${NC}        $PHASE"
echo -e "  ${CYAN}Results${NC}      $RESULTS_DIR"
echo -e "  ${CYAN}Config${NC}       $YAML_CONFIG"
[[ $DRY_RUN -eq 1 ]] && echo -e "  ${YELLOW}Mode${NC}         DRY-RUN"
echo ""

# Parse models
mapfile -t MODEL_LINES < <(parse_models "$YAML_CONFIG" "$PHASE" "${SELECTED_MODELS[@]+"${SELECTED_MODELS[@]}"}")

if [[ ${#MODEL_LINES[@]} -eq 0 ]]; then
    fail "No models found for phase=$PHASE"
    exit 1
fi

log "Found ${#MODEL_LINES[@]} model(s) to benchmark:"
for line in "${MODEL_LINES[@]}"; do
    IFS='|' read -r name backend model_path m_phase <<< "$line"
    log "  - $name (backend=$backend, phase=$m_phase)"
done
echo ""

# Benchmark each model as a separate run
RUN_NUM=0
TOTAL=0
PASSED=0
FAILED_LIST=()

for line in "${MODEL_LINES[@]}"; do
    IFS='|' read -r name backend model_path m_phase <<< "$line"
    RUN_NUM=$((RUN_NUM + 1))
    TOTAL=$((TOTAL + 1))

    RUN_DIR="${RESULTS_DIR}/run-${RUN_NUM}"
    header "Run $RUN_NUM: $name ($backend, $m_phase)"

    log "  Results: $RUN_DIR/"

    PORT=8000
    if [[ "$backend" == "llamacpp" ]]; then
        PORT=8001
    fi

    # Start server
    if [[ "$backend" == "vllm" ]]; then
        if ! start_vllm "$model_path" "$PORT"; then
            FAILED_LIST+=("$name")
            continue
        fi
    elif [[ "$backend" == "llamacpp" ]]; then
        if ! start_llamacpp "$model_path" "$PORT"; then
            FAILED_LIST+=("$name")
            continue
        fi
    else
        warn "Unknown backend: $backend, skipping"
        FAILED_LIST+=("$name")
        continue
    fi

    # Run benchmark
    BENCH_DIR="${RUN_DIR}/${backend}"
    if run_benchmark "$backend" "$name" "$model_path" "$PORT" "$BENCH_DIR"; then
        PASSED=$((PASSED + 1))
        ok "Run $RUN_NUM complete: $name"
    else
        FAILED_LIST+=("$name")
        fail "Run $RUN_NUM failed: $name"
    fi

    # Stop server
    stop_server "$backend" "$PORT"
    echo ""
done

# ── Summary ─────────────────────────────────────────────────────
header "Summary"
echo -e "  Total runs: ${CYAN}$TOTAL${NC}"
echo -e "  Passed:     ${GREEN}$PASSED${NC}"
echo -e "  Failed:     ${RED}$((TOTAL - PASSED))${NC}"

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo ""
    echo -e "  ${RED}Failed models:${NC}"
    for m in "${FAILED_LIST[@]}"; do
        echo -e "    - $m"
    done
fi

echo ""
echo -e "  Results: ${CYAN}$RESULTS_DIR/${NC}"
echo ""
echo -e "  ${DIM}Structure:${NC}"
for d in "$RESULTS_DIR"/run-*/; do
    [[ -d "$d" ]] || continue
    run_name=$(basename "$d")
    echo -e "    ${CYAN}$run_name/${NC}"
    for fw_dir in "$d"*/; do
        [[ -d "$fw_dir" ]] || continue
        fw_name=$(basename "$fw_dir")
        count=$(find "$fw_dir" -name "*.json" 2>/dev/null | wc -l)
        echo -e "      $fw_name/ ($count files)"
    done
done
echo ""
echo -e "  Log: ${CYAN}$LOG_FILE${NC}"
echo ""

# Parse results
if [[ $DRY_RUN -eq 0 && -f "${PROJECT_ROOT}/scripts/parse-results.py" ]]; then
    log "Parsing results..."
    python3 "${PROJECT_ROOT}/scripts/parse-results.py" --results-dir "$RESULTS_DIR" \
        2>&1 | tee -a "$LOG_FILE"
fi
