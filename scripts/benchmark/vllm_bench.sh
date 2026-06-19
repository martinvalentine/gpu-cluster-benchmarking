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

VLLM_BIN="${VLLM_BIN:-$(command -v vllm 2>/dev/null || echo "${PROJECT_ROOT}/.venv/bin/vllm")}"

BASE_URL="${VLLM_BENCH_URL:-http://localhost:8000/v1}"
MODEL="${VLLM_BENCH_MODEL:-}"
RESULTS_DIR="${VLLM_RESULTS_DIR:-${PROJECT_ROOT}/results/vllm}"
DATASET="${VLLM_BENCH_DATASET:-sharegpt}"
DATASET_PATH="${VLLM_BENCH_DATASET_PATH:-}"
OUTPUT_FORMAT="json"
CCU_MODE="mul"
CCU_START=1
CCU_MAX=256
CCU_STEP=2
PROMPT_START=1
PROMPT_MAX=16384
CROSS_SWEEP=0
EARLY_EXIT=0
TG=128
RUNS=3
NATIVE=0
LLAMA_BENCHY_MODEL=""
NATIVE_MAX_CONC=128
NATIVE_NUM_PROMPTS=512

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark an OpenAI-compatible endpoint using llama-benchy.

OPTIONS:
  -u, --url URL               Endpoint URL (default: http://localhost:8000/v1)
  -m, --model NAME            HF model name for tokenization (optional, auto-detected)
  -o, --output DIR            Results directory (default: ./results/vllm)
  --format FMT                Output format: json, csv, md (default: json)

  Sweep 1: Concurrency ladder
  --ccu-mode MODE             CCU step mode: mul, add (default: mul)
  --ccu-start N               Starting CCU (default: 1)
  --ccu-max N                 Maximum CCU (default: 256 for mul, 64 for add)
  --ccu-step N                Step size (default: 2 for mul, 4 for add)

  Sweep 2: Prompt length ladder
  --prompt-start N            Starting prompt tokens (default: 1)
  --prompt-max N              Maximum prompt tokens (default: 16384)

  Sweep 3: Cross-sweep (CCU ladder at each prompt size)
  --cross-sweep               Run CCU ladder at each prompt size (instead of separate sweeps)
  --early-exit                Stop CCU ladder at first hard error per prompt size

  llama-benchy pass-through
  --tg N                      Token generation count (default: 128)
  --runs N                    Runs per test (default: 3)

  Native benchmark (vllm bench serve)
  --native                    Also run native dataset-driven benchmark
  --full                      Run both llama-benchy and native
  --native-max-conc N         Native max concurrency (default: 128)
  --native-num-prompts N      Native num prompts (default: 512)

  -h, --help                  Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)               BASE_URL="$2"; shift 2 ;;
        -m|--model)             LLAMA_BENCHY_MODEL="$2"; shift 2 ;;
        -o|--output)            RESULTS_DIR="$2"; shift 2 ;;
        --format)               OUTPUT_FORMAT="$2"; shift 2 ;;
        --ccu-mode)             CCU_MODE="$2"; shift 2 ;;
        --ccu-start)            CCU_START="$2"; shift 2 ;;
        --ccu-max)              CCU_MAX="$2"; shift 2 ;;
        --ccu-step)             CCU_STEP="$2"; shift 2 ;;
        --prompt-start)         PROMPT_START="$2"; shift 2 ;;
        --prompt-max)           PROMPT_MAX="$2"; shift 2 ;;
        --cross-sweep)          CROSS_SWEEP=1; shift ;;
        --early-exit)           EARLY_EXIT=1; shift ;;
        --tg)                   TG="$2"; shift 2 ;;
        --runs)                 RUNS="$2"; shift 2 ;;
        --native)               NATIVE=1; shift ;;
        --full)                 NATIVE=1; shift ;;
        --native-max-conc)      NATIVE_MAX_CONC="$2"; shift 2 ;;
        --native-num-prompts)   NATIVE_NUM_PROMPTS="$2"; shift 2 ;;
        -h|--help)              usage ;;
        *)                      echo "Unknown: $1" >&2; usage ;;
    esac
done

# vllm bench serve uses base URL without /v1
VLLM_BASE_URL="${BASE_URL%/v1}"

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

get_model_name() {
    curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    name = d['data'][0]['id']
    print(name)
except: print('unknown')
" 2>/dev/null || echo "unknown"
}

MODEL_FULL=$(get_model_name)

HF_FORMAT_PATTERN='^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
LLAMA_BENCHY_ARGS=()
if [[ "$MODEL_FULL" =~ $HF_FORMAT_PATTERN ]]; then
    LLAMA_BENCHY_ARGS+=(--model "$MODEL_FULL")
elif [[ -n "$LLAMA_BENCHY_MODEL" ]]; then
    LLAMA_BENCHY_ARGS+=(--model "$LLAMA_BENCHY_MODEL" --served-model-name "$MODEL_FULL")
else
    echo "ERROR: Server model '$MODEL_FULL' is not an HF model name (org/model)." >&2
    echo "  Pass -m <HF_MODEL_NAME> to specify a valid HF model for tokenization." >&2
    echo "  Example: -m Qwen/Qwen2.5-0.5B-Instruct" >&2
    exit 1
fi

detect_model_short() {
    local full="$1"
    local short
    short=$(echo "$full" | grep -oiP 'qwen[\d.]+-\d+\.?\d*b' | head -1)
    if [[ -z "$short" ]]; then
        short=$(echo "$full" | grep -oiP 'llama[\d.]+-\d+\.?\d*b' | head -1)
    fi
    if [[ -z "$short" ]]; then
        short=$(basename "$full" | head -c 20)
    fi
    echo "$short"
}

TIMESTAMP="$(date +%Y-%m-%d_%Hh%M)"
MODEL_SHORT=$(detect_model_short "$MODEL_FULL")
SESSION_DIR="${RESULTS_DIR}/${TIMESTAMP}_${MODEL_SHORT}_vllm"
mkdir -p "$SESSION_DIR"

# Auto-detect max model length from endpoint to cap prompt ladder
MAX_MODEL_LEN=$(curl -sf "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for m in d.get('data', []):
        ml = m.get('max_model_len', 0)
        if ml: print(ml); break
except: pass
" 2>/dev/null || echo "0")
if [[ "$MAX_MODEL_LEN" -gt 0 && "$MAX_MODEL_LEN" -lt "$PROMPT_MAX" ]]; then
    PROMPT_MAX=$((MAX_MODEL_LEN > TG ? MAX_MODEL_LEN - TG : MAX_MODEL_LEN))
    log "  Capping prompt ladder to $PROMPT_MAX (model max=$MAX_MODEL_LEN, tg=$TG)"
fi

log "llama-benchy benchmark for vLLM"
log "  URL:      $BASE_URL"
log "  Model:    $MODEL_FULL"
log "  Output:   $SESSION_DIR"
log "  Format:   $OUTPUT_FORMAT"
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

prompt_sizes=()
ps=$PROMPT_START
_ps_iter=0
while [[ $ps -le $PROMPT_MAX ]] && [[ $_ps_iter -lt 100 ]] && [[ $ps -gt 0 ]]; do
    prompt_sizes+=($ps)
    ps=$((ps * 2))
    _ps_iter=$((_ps_iter + 1))
done

if [[ "$CROSS_SWEEP" -eq 1 ]]; then
    log "Cross-sweep: CCU ladder at each prompt size (tg=$TG)"
    log "  Prompt sizes: ${prompt_sizes[*]}"
    [[ "$EARLY_EXIT" -eq 1 ]] && log "  Early-exit: enabled (stop CCU ladder on first hard error per prompt size)"

    start_gpu_monitor

    for pp in "${prompt_sizes[@]}"; do
        log "  Testing pp=$pp with CCU: ${ccu_list[*]}..."

        if [[ "$EARLY_EXIT" -eq 1 ]]; then
            # Run each CCU level individually for early-exit support
            for ccu in "${ccu_list[@]}"; do
                log "    ccu=$ccu..."
                if timeout 300 "$LLAMA_BIN" \
                    --base-url "$BASE_URL" \
                    "${LLAMA_BENCHY_ARGS[@]}" \
                    --pp "$pp" \
                    --tg "$TG" \
                    --concurrency "$ccu" \
                    --format "$OUTPUT_FORMAT" \
                    --save-result "$SESSION_DIR/cross_sweep_pp${pp}_ccu${ccu}.json" \
                    --runs "$RUNS" \
                    2>&1; then
                    ok "    ccu=$ccu saved"
                else
                    warn "    ccu=$ccu failed — stopping CCU ladder for pp=$pp"
                    break
                fi
            done
        else
            timeout 300 "$LLAMA_BIN" \
                --base-url "$BASE_URL" \
                "${LLAMA_BENCHY_ARGS[@]}" \
                --pp "$pp" \
                --tg "$TG" \
                --concurrency "${ccu_list[@]}" \
                --format "$OUTPUT_FORMAT" \
                --save-result "$SESSION_DIR/cross_sweep_pp${pp}.json" \
                --runs "$RUNS" \
                2>&1 && ok "  pp=$pp saved" || warn "  pp=$pp had errors"
        fi
    done

    stop_gpu_monitor
else
    log "Sweep 1: Concurrency ladder (fixed pp=2048, tg=$TG)"

    start_gpu_monitor

    timeout 300 "$LLAMA_BIN" \
        --base-url "$BASE_URL" \
        "${LLAMA_BENCHY_ARGS[@]}" \
        --pp 2048 \
        --tg "$TG" \
        --concurrency "${ccu_list[@]}" \
        --format "$OUTPUT_FORMAT" \
        --save-result "$SESSION_DIR/ccu_sweep.json" \
        --runs "$RUNS" \
        2>&1 && ok "Concurrency sweep saved to $SESSION_DIR/ccu_sweep.json" || warn "Concurrency sweep had errors"

    stop_gpu_monitor

    echo ""

    log "Sweep 2: Prompt length ladder (fixed ccu=1, tg=$TG)"
    log "  Prompt sizes: ${prompt_sizes[*]}"

    start_gpu_monitor

    timeout 300 "$LLAMA_BIN" \
        --base-url "$BASE_URL" \
        "${LLAMA_BENCHY_ARGS[@]}" \
        --pp "${prompt_sizes[@]}" \
        --tg "$TG" \
        --concurrency 1 \
        --format "$OUTPUT_FORMAT" \
        --save-result "$SESSION_DIR/prompt_sweep.json" \
        --runs "$RUNS" \
        2>&1 && ok "Prompt sweep saved to $SESSION_DIR/prompt_sweep.json" || warn "Prompt sweep had errors"

    stop_gpu_monitor
fi


if [[ "$NATIVE" -eq 1 ]]; then
    echo ""
    log "Native vLLM benchmark (vllm bench serve)"

    NATIVE_MODEL_RESOLVED="$LLAMA_BENCHY_MODEL"
    if [[ -z "$NATIVE_MODEL_RESOLVED" ]]; then
        NATIVE_MODEL_RESOLVED="$MODEL"
    fi
    if [[ -z "$NATIVE_MODEL_RESOLVED" ]]; then
        NATIVE_MODEL_RESOLVED="$MODEL_FULL"
    fi

    if [[ ! -x "$VLLM_BIN" ]] && ! command -v vllm &>/dev/null; then
        warn "vllm binary not found — skipping native benchmark"
    else
        start_gpu_monitor

        for CONC in 1 "$NATIVE_MAX_CONC"; do
            local num_prompts=$((CONC * 8))
            if [[ "$num_prompts" -gt "$NATIVE_NUM_PROMPTS" ]]; then
                num_prompts="$NATIVE_NUM_PROMPTS"
            fi

            log "  Native conc=$CONC num_prompts=$num_prompts"
            "$VLLM_BIN" bench serve \
                --backend openai-chat \
                --base-url "$VLLM_BASE_URL" \
                --endpoint /v1/chat/completions \
                --model "$NATIVE_MODEL_RESOLVED" \
                --dataset-name "$DATASET" \
                ${DATASET_PATH:+--dataset-path "$DATASET_PATH"} \
                --num-prompts "$num_prompts" \
                --max-concurrency "$CONC" \
                --request-rate inf \
                --percentile-metrics ttft,tpot,itl,e2el \
                --save-result \
                --result-dir "$SESSION_DIR" \
                --result-filename "native_conc${CONC}.json" \
                2>&1 && ok "  native conc=$CONC saved" || warn "  native conc=$CONC failed"
            sleep 3
        done

        stop_gpu_monitor
    fi
fi

echo ""
ok "Benchmark complete. Results: $SESSION_DIR/"
ls -la "$SESSION_DIR/"
