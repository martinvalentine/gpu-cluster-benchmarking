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
import sys, yaml, subprocess
from pathlib import Path

yaml_path = sys.argv[1]
phase_filter = sys.argv[2]
selected = sys.argv[3:] if len(sys.argv) > 3 else []

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

# Resolve GPU count
cluster = cfg.get("cluster", {})
gpu_count = cluster.get("gpu_count", 0)
if not gpu_count:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        gpu_count = len(out.strip().splitlines()) if out.strip() else 1
    except Exception:
        gpu_count = 1

# Cluster defaults
vllm_defaults = cluster.get("vllm", {})
llamacpp_defaults = cluster.get("llamacpp", {})

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

    # Per-model vLLM config (override cluster defaults)
    vllm_tp = m.get("vllm_tp", vllm_defaults.get("tp", 1))
    vllm_gpu_mem = m.get("vllm_gpu_mem", vllm_defaults.get("gpu_mem_util", "0.87"))
    vllm_quant = m.get("vllm_quant", vllm_defaults.get("quant", "none"))
    vllm_max_seqs = m.get("vllm_max_seqs", vllm_defaults.get("max_num_seqs", 64))
    vllm_max_model_len = vllm_defaults.get("max_model_len", 4096)
    vllm_max_batched_tokens = vllm_defaults.get("max_batched_tokens", 8192)
    vllm_swap_space = vllm_defaults.get("swap_space", 4)

    # Per-model llama.cpp config (override cluster defaults)
    llama_np = m.get("llamacpp_n_parallel", llamacpp_defaults.get("n_parallel", 4))
    llama_ctx = m.get("llamacpp_ctx_size", llamacpp_defaults.get("ctx_size", 4096))
    llama_ngl = m.get("llamacpp_n_gpu_layers", llamacpp_defaults.get("n_gpu_layers", "all"))
    llama_batch = m.get("llamacpp_batch", llamacpp_defaults.get("batch", 2048))
    llama_ubatch = m.get("llamacpp_ubatch", llamacpp_defaults.get("ubatch", 512))
    llama_threads = m.get("llamacpp_threads", llamacpp_defaults.get("threads", 0))
    llama_ctk = m.get("llamacpp_cache_key", llamacpp_defaults.get("cache_key", "q8_0"))
    llama_ctv = m.get("llamacpp_cache_val", llamacpp_defaults.get("cache_val", "turbo4"))
    llama_fa = m.get("llamacpp_flash_attn", llamacpp_defaults.get("flash_attn", "on"))
    if isinstance(llama_fa, bool):
        llama_fa = "on" if llama_fa else "off"
    llama_cp = m.get("llamacpp_cache_prompt", llamacpp_defaults.get("cache_prompt", True))
    if isinstance(llama_cp, bool):
        llama_cp = str(llama_cp).lower()

    # Skip if model needs more GPUs than available
    skip = ""
    if backend == "vllm" and int(vllm_tp) > gpu_count:
        skip = f"SKIP:tp>{gpu_count}"

    # Output: name|backend|model_path|phase|vllm_tp|vllm_gpu_mem|vllm_quant|vllm_max_seqs|gpu_count|skip|max_model_len|max_batched_tokens|swap_space|llama_np|llama_ctx|llama_ngl|llama_batch|llama_ubatch|llama_threads|llama_ctk|llama_ctv|llama_fa|llama_cp
    print(f"{name}|{backend}|{model_path}|{m_phase}|{vllm_tp}|{vllm_gpu_mem}|{vllm_quant}|{vllm_max_seqs}|{gpu_count}|{skip}|{vllm_max_model_len}|{vllm_max_batched_tokens}|{vllm_swap_space}|{llama_np}|{llama_ctx}|{llama_ngl}|{llama_batch}|{llama_ubatch}|{llama_threads}|{llama_ctk}|{llama_ctv}|{llama_fa}|{llama_cp}")

# Output dataset path as first line
dataset_cfg = cfg.get("dataset", {})
sharegpt_path = dataset_cfg.get("sharegpt", "/workspace/datasets/sharegpt.json")
print(f"DATASET|{sharegpt_path}")
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
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            # Kill entire process tree (parent + children like VLLM::EngineCore)
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        done
        sleep 2
        # Force kill anything still on the port
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        for pid in $pids; do
            kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        done
    fi
    # Also kill any VLLM::EngineCore processes
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
}

# ── Start vLLM server ───────────────────────────────────────────
start_vllm() {
    local model_path=$1 port=$2 tp=${3:-1} gpu_mem=${4:-0.87} quant=${5:-none} max_seqs=${6:-64}
    local max_model_len=${7:-4096} max_batched_tokens=${8:-8192} swap_space=${9:-4}
    log "Starting vLLM: model=$model_path port=$port tp=$tp gpu_mem=$gpu_mem quant=$quant max_seqs=$max_seqs"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would start: vllm serve $model_path --port $port --tp $tp --gpu-mem $gpu_mem --quant $quant --max-seqs $max_seqs --max-model-len $max_model_len --max-batched-tokens $max_batched_tokens"
        return 0
    fi

    kill_server_on_port "$port"

    local vllm_bin="${PROJECT_ROOT}/.venv/bin/vllm"
    if [[ ! -f "$vllm_bin" ]]; then
        vllm_bin=$(command -v vllm 2>/dev/null || true)
    fi
    if [[ -z "$vllm_bin" || ! -f "$vllm_bin" ]]; then
        fail "vllm binary not found (checked .venv/bin/vllm and PATH)"
        return 1
    fi

    local quant_args=()
    if [[ "$quant" != "none" ]]; then
        quant_args=(--quantization "$quant")
    fi

    tmux new-session -d -s "bench-vllm" -n "serve" \
        "VLLM_USE_FLASHINFER_SAMPLER=0 $vllm_bin serve $model_path \
            --host 0.0.0.0 --port $port \
            --tensor-parallel-size $tp \
            --gpu-memory-utilization $gpu_mem \
            --max-model-len $max_model_len \
            --max-num-seqs $max_seqs \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-num-batched-tokens $max_batched_tokens \
            --trust-remote-code \
            --enforce-eager ${quant_args[@]:-} 2>&1 | tee /tmp/bench_vllm.log; sleep infinity"

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
    local np=${3:-4} ctx=${4:-4096} ngl=${5:-all} batch=${6:-2048} ubatch=${7:-512}
    local threads=${8:-0} ctk=${9:-q8_0} ctv=${10:-turbo4} fa=${11:-on} cache_prompt=${12:-true}
    log "Starting llama.cpp: model=$model_path port=$port np=$np ctx=$ctx ngl=$ngl batch=$batch ubatch=$ubatch ctk=$ctk ctv=$ctv fa=$fa cache_prompt=$cache_prompt"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would start: llama-server -m $model_path --port $port -np $np -c $ctx -ngl $ngl -b $batch -ub $ubatch -t $threads -ctk $ctk -ctv $ctv -fa $fa --cache-prompt"
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

    # Resolve threads: 0 = auto (nproc)
    local threads_args=()
    if [[ "$threads" -gt 0 ]]; then
        threads_args=(-t "$threads")
    fi

    # Resolve ngl: "all" → -ngl 999
    local ngl_value="$ngl"
    if [[ "$ngl" == "all" ]]; then
        ngl_value=999
    fi

    # Resolve cache prompt flag
    local cp_flag="--cache-prompt"
    if [[ "$cache_prompt" == "false" || "$cache_prompt" == "False" || "$cache_prompt" == "0" ]]; then
        cp_flag="--no-cache-prompt"
    fi

    tmux new-session -d -s "bench-llama" -n "serve" \
        "cd $PROJECT_ROOT && $llama_bin \
            -m $model_path \
            --host 0.0.0.0 --port $port \
            -np $np -c $ctx -ngl $ngl_value \
            -b $batch -ub $ubatch \
            ${threads_args[@]:-} \
            -ctk $ctk -ctv $ctv -fa $fa \
            $cp_flag \
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
        VLLM_BENCH_DATASET_PATH="$DATASET_PATH" \
        "$bench_script" -p "$PHASE" 2>&1 | tee -a "$LOG_FILE"
    elif [[ "$fw" == "llamacpp" ]]; then
        LLAMA_RESULTS_DIR="$results_dir" \
        LLAMA_BENCH_URL="http://localhost:$port/v1" \
        "$bench_script" 2>&1 | tee -a "$LOG_FILE"
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
mapfile -t ALL_LINES < <(parse_models "$YAML_CONFIG" "$PHASE" "${SELECTED_MODELS[@]+"${SELECTED_MODELS[@]}"}")

# Extract dataset path (first line)
DATASET_PATH="/workspace/datasets/sharegpt.json"
MODEL_LINES=()
for line in "${ALL_LINES[@]}"; do
    if [[ "$line" == DATASET\|* ]]; then
        DATASET_PATH="${line#DATASET|}"
    else
        MODEL_LINES+=("$line")
    fi
done
log "Dataset: $DATASET_PATH"

if [[ ${#MODEL_LINES[@]} -eq 0 ]]; then
    fail "No models found for phase=$PHASE"
    exit 1
fi

log "Found ${#MODEL_LINES[@]} model(s) to benchmark:"
for line in "${MODEL_LINES[@]}"; do
    IFS='|' read -r name backend model_path m_phase vllm_tp vllm_gpu_mem vllm_quant vllm_max_seqs gpu_count skip vllm_max_model_len vllm_max_batched_tokens vllm_swap_space llama_np llama_ctx llama_ngl llama_batch llama_ubatch llama_threads llama_ctk llama_ctv llama_fa llama_cp <<< "$line"
    if [[ -n "$skip" ]]; then
        warn "  - $name (backend=$backend, phase=$m_phase, tp=$vllm_tp) — $skip (pod has $gpu_count GPU(s))"
    elif [[ "$backend" == "vllm" ]]; then
        log "  - $name (vllm): tp=$vllm_tp gpu_mem=$vllm_gpu_mem quant=$vllm_quant max_seqs=$vllm_max_seqs max_model_len=$vllm_max_model_len"
    else
        log "  - $name (llamacpp): np=$llama_np ctx=$llama_ctx ngl=$llama_ngl batch=$llama_batch ubatch=$llama_ubatch ctk=$llama_ctk ctv=$llama_ctv fa=$llama_fa cache_prompt=$llama_cp"
    fi
done
echo ""

# Benchmark each model as a separate run
RUN_NUM=0
TOTAL=0
PASSED=0
SKIPPED=0
FAILED_LIST=()

for line in "${MODEL_LINES[@]}"; do
    IFS='|' read -r name backend model_path m_phase vllm_tp vllm_gpu_mem vllm_quant vllm_max_seqs gpu_count skip vllm_max_model_len vllm_max_batched_tokens vllm_swap_space llama_np llama_ctx llama_ngl llama_batch llama_ubatch llama_threads llama_ctk llama_ctv llama_fa llama_cp <<< "$line"
    RUN_NUM=$((RUN_NUM + 1))
    TOTAL=$((TOTAL + 1))

    # Skip models that need more GPUs than available
    if [[ -n "$skip" ]]; then
        warn "Skipping $name — $skip"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    RUN_DIR="${RESULTS_DIR}/run-${RUN_NUM}"
    header "Run $RUN_NUM: $name ($backend, $m_phase)"

    log "  Results: $RUN_DIR/"

    PORT=8000
    if [[ "$backend" == "llamacpp" ]]; then
        PORT=8001
    fi

    # Start server
    if [[ "$backend" == "vllm" ]]; then
        if ! start_vllm "$model_path" "$PORT" "$vllm_tp" "$vllm_gpu_mem" "$vllm_quant" "$vllm_max_seqs" "$vllm_max_model_len" "$vllm_max_batched_tokens" "$vllm_swap_space"; then
            FAILED_LIST+=("$name")
            continue
        fi
    elif [[ "$backend" == "llamacpp" ]]; then
        if ! start_llamacpp "$model_path" "$PORT" "$llama_np" "$llama_ctx" "$llama_ngl" "$llama_batch" "$llama_ubatch" "$llama_threads" "$llama_ctk" "$llama_ctv" "$llama_fa" "$llama_cp"; then
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
echo -e "  Total models: ${CYAN}$TOTAL${NC}"
echo -e "  Passed:       ${GREEN}$PASSED${NC}"
[[ $SKIPPED -gt 0 ]] && echo -e "  Skipped:      ${YELLOW}$SKIPPED${NC} (need more GPUs)"
echo -e "  Failed:       ${RED}$((TOTAL - PASSED - SKIPPED))${NC}"

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
