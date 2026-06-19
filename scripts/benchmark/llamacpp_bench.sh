#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

LLAMA_BIN="${PROJECT_ROOT}/.venv/bin/llama-benchy"
if [[ ! -f "$LLAMA_BIN" ]]; then
    LLAMA_BIN=$(command -v llama-benchy 2>/dev/null || echo "")
fi
if [[ -z "$LLAMA_BIN" || ! -f "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-benchy not found. Install via: uv sync --group benchmark" >&2
    exit 1
fi

BASE_URL="${LLAMA_BENCH_URL:-http://localhost:8001/v1}"
RESULTS_DIR="${LLAMA_RESULTS_DIR:-${PROJECT_ROOT}/results/llamacpp}"
OUTPUT_FORMAT="json"
CCU_MODE="mul"
CCU_START=1
CCU_MAX=256
CCU_STEP=2
PROMPT_START=1
PROMPT_MAX=16384
PROMPT_STEP="mul"
TG=128
RUNS=3
MODEL=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark an OpenAI-compatible endpoint using llama-benchy.

OPTIONS:
  -u, --url URL               Endpoint URL (default: http://localhost:8001/v1)
  -o, --output DIR            Results directory (default: ./results/llamacpp)
  --format FMT                Output format: json, csv, md (default: json)

  Sweep 1: Concurrency ladder
  --ccu-mode MODE             CCU step mode: mul, add (default: mul)
  --ccu-start N               Starting CCU (default: 1)
  --ccu-max N                 Maximum CCU (default: 256 for mul, 64 for add)
  --ccu-step N                Step size (default: 2 for mul, 4 for add)

  Sweep 2: Prompt length ladder
  --prompt-start N            Starting prompt tokens (default: 1)
  --prompt-max N              Maximum prompt tokens (default: 16384)

  llama-benchy pass-through
  --tg N                      Token generation count (default: 128)
  --runs N                    Runs per test (default: 3)
  -m, --model MODEL           HF model name (e.g. Qwen/Qwen2.5-0.5B-Instruct).
                                Required when server returns GGUF filenames.

  -h, --help                  Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)           BASE_URL="$2"; shift 2 ;;
        -o|--output)        RESULTS_DIR="$2"; shift 2 ;;
        --format)           OUTPUT_FORMAT="$2"; shift 2 ;;
        --ccu-mode)         CCU_MODE="$2"; shift 2 ;;
        --ccu-start)        CCU_START="$2"; shift 2 ;;
        --ccu-max)          CCU_MAX="$2"; shift 2 ;;
        --ccu-step)         CCU_STEP="$2"; shift 2 ;;
        --prompt-start)     PROMPT_START="$2"; shift 2 ;;
        --prompt-max)       PROMPT_MAX="$2"; shift 2 ;;
        --tg)               TG="$2"; shift 2 ;;
        --runs)             RUNS="$2"; shift 2 ;;
        -m|--model)         MODEL="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown: $1" >&2; usage ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }

source "${SCRIPT_DIR}/_embed_params.sh"

get_model_name() {
    curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    if name.endswith('.gguf'):
        name = name.rsplit('.gguf', 1)[0]
    print(name)
except: print('unknown')
" 2>/dev/null || echo "unknown"
}

detect_model_short() {
    local full="$1"
    local short
    short=$(echo "$full" | grep -oiP 'qwen[\d.]+-\d+\.?\d*b' | head -1)
    if [[ -z "$short" ]]; then
        short=$(echo "$full" | grep -oiP 'llama[\d.]+-\d+\.?\d*b' | head -1)
    fi
    if [[ -z "$short" ]]; then
        short=$(basename "$full" .gguf | head -c 20)
    fi
    echo "$short"
}

# Create timestamped session directory
TIMESTAMP="$(date +%Y-%m-%d_%Hh%M)"
MODEL_FULL=$(get_model_name)
MODEL_SHORT=$(detect_model_short "$MODEL_FULL")
SESSION_DIR="${RESULTS_DIR}/${TIMESTAMP}_${MODEL_SHORT}_llamacpp"
mkdir -p "$SESSION_DIR"

# Capture server params snapshot for this run
PARAMS_FILE="${RESULTS_DIR}/_active_params.json"
PARAMS_SNAPSHOT="${SESSION_DIR}/params.json"
if [[ -f "$PARAMS_FILE" ]]; then
    cp "$PARAMS_FILE" "$PARAMS_SNAPSHOT"
    log "Server params snapshot: $PARAMS_SNAPSHOT"
    if command -v jq &>/dev/null; then
        log "  model:    $(jq -r '.server.model' "$PARAMS_SNAPSHOT")"
        log "  endpoint: $(jq -r '.server.endpoint' "$PARAMS_SNAPSHOT")"
        log "  ctk/ctv:  $(jq -r '.server.cache_key' "$PARAMS_SNAPSHOT")/$(jq -r '.server.cache_val' "$PARAMS_SNAPSHOT")"
        log "  fa:       $(jq -r '.server.flash_attn' "$PARAMS_SNAPSHOT")"
        log "  n_parallel: $(jq -r '.server.n_parallel' "$PARAMS_SNAPSHOT")"
        log "  gpu:      $(jq -r '.hardware.gpu_count' "$PARAMS_SNAPSHOT")x $(jq -r '.hardware.gpu_name' "$PARAMS_SNAPSHOT")"
        log "  cuda:     $(jq -r '.hardware.cuda_version' "$PARAMS_SNAPSHOT")"
        log "  commit:   $(jq -r '.system.git_commit' "$PARAMS_SNAPSHOT")"
    else
        warn "jq not found; server params logged to $PARAMS_SNAPSHOT only"
    fi
else
    warn "No _active_params.json at $PARAMS_FILE — run a run-*.sh script first. Params will be missing from sweep results."
fi

log "llama-benchy benchmark for llama.cpp"
log "  URL:      $BASE_URL"
log "  Model:    $MODEL_FULL"
log "  Output:   $SESSION_DIR"
MODEL_ARG=()
if [[ -n "$MODEL" ]]; then
    MODEL_ARG=(--model "$MODEL")
fi

log "Model arg: ${MODEL_ARG[*]:-(auto-detect)}"
echo ""

# GPU monitor
GPU_MON_PID=""
gpu_log="$SESSION_DIR/gpu_util.log"
echo -e "timestamp\tgpu\tsm_util\tmem_util\tenc_util\tdec_util\tmem_used\tmem_total\ttemp\tpower" > "$gpu_log"

start_gpu_monitor() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi dmon -s u -d 1 -o DT >> "$gpu_log" 2>/dev/null &
        GPU_MON_PID=$!
    fi
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MON_PID" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
    fi
    GPU_MON_PID=""
}

log "Sweep 1: Concurrency ladder (fixed pp=2048, tg=$TG)"

ccu_list=()
ccu=$CCU_START
_ccu_iter=0
if [[ "$CCU_MODE" == "mul" ]]; then
    while [[ $ccu -le $CCU_MAX ]] && [[ $_ccu_iter -lt 100 ]]; do
        ccu_list+=($ccu)
        ccu=$((ccu * CCU_STEP))
        [[ ${#ccu_list[@]} -gt 1 && "$ccu" -eq "$CCU_START" ]] && break
        _ccu_iter=$((_ccu_iter + 1))
    done
else
    while [[ $ccu -le $CCU_MAX ]] && [[ $_ccu_iter -lt 100 ]]; do
        ccu_list+=($ccu)
        ccu=$((ccu + CCU_STEP))
        _ccu_iter=$((_ccu_iter + 1))
    done
fi

log "  CCU levels: ${ccu_list[*]}"

start_gpu_monitor

timeout 300 "$LLAMA_BIN" \
    --base-url "$BASE_URL" \
    --pp 2048 \
    --tg "$TG" \
    --concurrency "${ccu_list[@]}" \
    --format "$OUTPUT_FORMAT" \
    --save-result "$SESSION_DIR/ccu_sweep.json" \
    --runs "$RUNS" \
    "${MODEL_ARG[@]}" \
    2>&1 && ok "Concurrency sweep saved to $SESSION_DIR/ccu_sweep.json" || warn "Concurrency sweep had errors"
embed_params_in_sweep "$SESSION_DIR/ccu_sweep.json"

stop_gpu_monitor

echo ""

log "Sweep 2: Prompt length ladder (fixed ccu=1, tg=$TG)"

prompt_sizes=()
ps=$PROMPT_START
_ps_iter=0
while [[ $ps -le $PROMPT_MAX ]] && [[ $_ps_iter -lt 100 ]] && [[ $ps -gt 0 ]]; do
    prompt_sizes+=($ps)
    ps=$((ps * 2))
    _ps_iter=$((_ps_iter + 1))
done

log "  Prompt sizes: ${prompt_sizes[*]}"

start_gpu_monitor

log "  Testing pp=${prompt_sizes[*]}..."
timeout 300 "$LLAMA_BIN" \
    --base-url "$BASE_URL" \
    --pp "${prompt_sizes[@]}" \
    --tg "$TG" \
    --concurrency 1 \
    --format "$OUTPUT_FORMAT" \
    --save-result "$SESSION_DIR/prompt_sweep.json" \
    --runs "$RUNS" \
    "${MODEL_ARG[@]}" \
    2>&1 && ok "Prompt sweep saved to $SESSION_DIR/prompt_sweep.json" || warn "Prompt sweep had errors"
embed_params_in_sweep "$SESSION_DIR/prompt_sweep.json"

stop_gpu_monitor

echo ""
ok "Benchmark complete. Results: $SESSION_DIR/"
ls -la "$SESSION_DIR/"
