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
  -p, --phase PHASE       Run specific model phase: p0, p1, p2, p3, all (default: all)
  -b, --backend BACKEND   Filter by backend: vllm, llamacpp, sglang (can repeat, default: all)
  -m, --model NAME        Run specific model by name (can repeat)
  --bench-phase PHASE     Benchmark load level: p0, p1, p2, p3, all (default: all)
                          p0=smoke, p1=light, p2=medium, p3=heavy
  --bench-type TYPE       Benchmark type: phases, stress (default: phases)
                          phases = fixed concurrency per phase
                          stress = incremental concurrency ramp
  --conc-base N           Stress mode starting concurrency (default: 1)
  --conc-step N           Stress mode increment per round (default: 100)
  --conc-max N            Stress mode maximum concurrency (default: 2000)
  --skip-health-check     Skip pre-flight health checks
  -y, --yes               Auto-accept prompts (for CI / non-interactive use)
  --dry-run               Show what would be run without executing
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                    # All models, all bench phases
  $(basename "$0") -p p0                              # P0 models only
  $(basename "$0") -b vllm                            # Only vLLM backends
  $(basename "$0") -b llamacpp                        # Only llama.cpp backends
  $(basename "$0") -m qwen0.5b                        # Specific model (all backends)
  $(basename "$0") --bench-phase p1                   # All models, light load only
  $(basename "$0") -p p1 --bench-phase p1             # P1 models, light load only
  $(basename "$0") --bench-type stress                # Incremental stress: 1→100→200→...→2000
  $(basename "$0") --bench-type stress --conc-base 50 --conc-step 50 --conc-max 500
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
BACKEND_FILTER=()
BENCH_PHASE="all"
BENCH_TYPE="phases"
CONC_BASE=1
CONC_STEP=100
CONC_MAX=2000
AUTO_YES=0
SKIP_HEALTH=0
DRY_RUN=0
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)            RESULTS_DIR="$2"; shift 2 ;;
        -p|--phase)             PHASE="$2"; shift 2 ;;
        -b|--backend)           BACKEND_FILTER+=("$2"); shift 2 ;;
        -m|--model)             SELECTED_MODELS+=("$2"); shift 2 ;;
        --bench-phase)          BENCH_PHASE="$2"; shift 2 ;;
        --bench-type)           BENCH_TYPE="$2"; shift 2 ;;
        --conc-base)            CONC_BASE="$2"; shift 2 ;;
        --conc-step)            CONC_STEP="$2"; shift 2 ;;
        --conc-max)             CONC_MAX="$2"; shift 2 ;;
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
ok()     { echo -e "  ${GREEN}✓${NC} $*" | tee -a "$LOG_FILE"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $*" | tee -a "$LOG_FILE"; }
fail()   { echo -e "  ${RED}✗${NC} $*" | tee -a "$LOG_FILE"; }

# ── Parse models from YAML ──────────────────────────────────────
parse_models() {
    local yaml="$1" phase="$2"
    shift 2
    local selected=("$@")

    "$PROJECT_ROOT/.venv/bin/python" - "$yaml" "$phase" "${selected[@]+"${selected[@]}"}" <<'PYEOF'
import sys, yaml, subprocess, os
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

vllm_defaults = cluster.get("vllm", {})
llamacpp_defaults = cluster.get("llamacpp", {})
sglang_defaults = cluster.get("sglang", {})

base_dir = os.environ.get("BENCH_BASE_DIR", cfg.get("base_dir", "models"))
models = cfg.get("models", [])

for m in models:
    if not m.get("enabled", True):
        continue
    name = m.get("name", "")
    repo_id = m.get("repo_id", "")
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

    vllm_tp = m.get("vllm_tp", vllm_defaults.get("tp", 1))
    vllm_gpu_mem = m.get("vllm_gpu_mem", vllm_defaults.get("gpu_mem_util", "0.92"))
    vllm_quant = m.get("vllm_quant", vllm_defaults.get("quant", "none"))
    vllm_max_seqs = m.get("vllm_max_seqs", vllm_defaults.get("max_num_seqs", 512))
    vllm_max_model_len = m.get("vllm_max_model_len", vllm_defaults.get("max_model_len", 4096))
    vllm_max_batched_tokens = m.get("vllm_max_batched_tokens", vllm_defaults.get("max_batched_tokens", 16384))
    vllm_block_size = m.get("vllm_block_size", vllm_defaults.get("block_size", 16))
    vllm_dtype = m.get("vllm_dtype", vllm_defaults.get("dtype", "auto"))
    vllm_swap_space = m.get("vllm_swap_space", vllm_defaults.get("swap_space", 4))
    vllm_distributed_executor = m.get("vllm_distributed_executor", vllm_defaults.get("distributed_executor", ""))
    vllm_enforce_eager = m.get("vllm_enforce_eager", vllm_defaults.get("enforce_eager", False))

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
    llama_ts = m.get("llamacpp_tensor_split", llamacpp_defaults.get("tensor_split", ""))

    sglang_mem = m.get("sglang_mem_fraction", sglang_defaults.get("mem_fraction", "0.85"))
    sglang_max_len = m.get("sglang_max_model_len", sglang_defaults.get("max_model_len", 4096))

    skip = ""
    if backend == "vllm" and int(vllm_tp) > gpu_count:
        skip = "SKIP:tp>{gpu_count}"

    # Output: name|repo_id|backend|model_path|phase|vllm_tp|vllm_gpu_mem|vllm_quant|vllm_max_seqs|gpu_count|skip|max_model_len|max_batched_tokens|block_size|dtype|llama_np|llama_ctx|llama_ngl|llama_batch|llama_ubatch|llama_threads|llama_ctk|llama_ctv|llama_fa|llama_cp|vllm_swap_space|vllm_distributed_executor|vllm_enforce_eager|llama_ts|sglang_mem_fraction|sglang_max_model_len
    print(f"{name}|{repo_id}|{backend}|{model_path}|{m_phase}|{vllm_tp}|{vllm_gpu_mem}|{vllm_quant}|{vllm_max_seqs}|{gpu_count}|{skip}|{vllm_max_model_len}|{vllm_max_batched_tokens}|{vllm_block_size}|{vllm_dtype}|{llama_np}|{llama_ctx}|{llama_ngl}|{llama_batch}|{llama_ubatch}|{llama_threads}|{llama_ctk}|{llama_ctv}|{llama_fa}|{llama_cp}|{vllm_swap_space}|{vllm_distributed_executor}|{vllm_enforce_eager}|{llama_ts}|{sglang_mem}|{sglang_max_len}")

# Output dataset path as first line
dataset_cfg = cfg.get("dataset", {})
sharegpt_path = os.environ.get("BENCH_DATASET_PATH", dataset_cfg.get("sharegpt", "/workspace/datasets/sharegpt.json"))
print(f"DATASET|{sharegpt_path}")
PYEOF
}

# ── Wait for server ─────────────────────────────────────────────
wait_for_server() {
    local name=$1 url=$2 timeout=${3:-60} proc_pattern=${4:-}
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -n "$proc_pattern" ]]; then
            local found=0
            while IFS= read -r pid; do
                local cmd
                cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
                if [[ "$cmd" != *"tmux"* ]]; then
                    found=1
                    break
                fi
            done < <(pgrep -f "$proc_pattern" 2>/dev/null)
            if [[ $found -eq 0 ]]; then
                log "  $name process died — check /tmp/bench_*.log for details"
                return 1
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "  $name timed out after ${timeout}s"
    return 1
}

# ── Kill server processes ──────────────────────────────────────
kill_process_tree() {
    local pid=$1
    local sig=${2:-TERM}
    local children
    children=$(ps -o pid= --ppid "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child" "$sig"
    done
    kill -"$sig" "$pid" 2>/dev/null || true
}

get_tmux_pane_pids() {
    local session=$1
    tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null || true
}

kill_tmux_session() {
    local session=$1
    local pids
    pids=$(get_tmux_pane_pids "$session")
    if [[ -n "$pids" ]]; then
        log "  Killing process trees for tmux session '$session': $pids"
        for pid in $pids; do
            kill_process_tree "$pid" TERM
        done
        sleep 1
        # Force kill any survivors
        pids=$(get_tmux_pane_pids "$session")
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                kill_process_tree "$pid" KILL
            done
        fi
    fi
    tmux kill-session -t "$session" 2>/dev/null || true
}

kill_server_by_name() {
    local fw=$1
    if [[ "$fw" == "vllm" ]]; then
        pkill -f "VLLM::EngineCore" 2>/dev/null || true
        pkill -f "vllm serve " 2>/dev/null || true
    elif [[ "$fw" == "llamacpp" ]]; then
        pkill -f "llama-server" 2>/dev/null || true
    elif [[ "$fw" == "sglang" ]]; then
        pkill -f "sglang.launch_server" 2>/dev/null || true
    fi
}

wait_for_process_exit() {
    local name=$1 timeout=${2:-5}
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if ! pgrep -f "$name" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

force_kill_by_name() {
    local pattern=$1
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "  Force-killing processes matching '$pattern': $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
}

# ── Wait for GPU memory to be released ─────────────────────────
wait_gpu_free() {
    local min_free_mb=${1:-5000}
    local timeout=${2:-30}
    local elapsed=0

    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi

    while [[ $elapsed -lt $timeout ]]; do
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
            | tr -d ' ' | sort -n | head -1 || echo "0")
        if [[ "$free_mb" -ge "$min_free_mb" ]]; then
            return 0
        fi
        log "  GPU memory: ${free_mb} MiB free (need ${min_free_mb} MiB), waiting..."
        sleep 2
        elapsed=$((elapsed + 2))
    done

    warn "GPU memory still low after ${timeout}s (${free_mb} MiB free), proceeding anyway"
    return 0
}

# ── GPU utilization logger ─────────────────────────────────────
GPU_MON_PID=""

start_gpu_monitor() {
    local log_file=$1
    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi
    nvidia-smi dmon -s u -d 1 -o DT >> "$log_file" 2>/dev/null &
    GPU_MON_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MON_PID" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
    fi
    GPU_MON_PID=""
}

# ── Start vLLM server ───────────────────────────────────────────
start_vllm() {
    local model_path=$1 port=$2 tp=${3:-6} gpu_mem=${4:-0.92} quant=${5:-none} max_seqs=${6:-512}
    local max_model_len=${7:-4096} max_batched_tokens=${8:-16384} block_size=${9:-16} dtype=${10:-auto}
    local swap_space=${11:-4} distributed_executor=${12:-} enforce_eager=${13:-false}
    log "Starting vLLM: model=$model_path port=$port tp=$tp gpu_mem=$gpu_mem quant=$quant max_seqs=$max_seqs block=$block_size dtype=$dtype swap=$swap_space exec=$distributed_executor eager=$enforce_eager"

    if [[ $DRY_RUN -eq 1 ]]; then
        local _eager_flag=""
        if [[ "$enforce_eager" == "true" || "$enforce_eager" == "True" ]]; then _eager_flag="--enforce-eager"; fi
        local _exec_flag=""
        if [[ -n "$distributed_executor" ]]; then _exec_flag="--distributed-executor-backend $distributed_executor"; fi
        local _swap_flag=""
        if [[ -n "$swap_space" && "$swap_space" -gt 0 ]]; then _swap_flag="--swap-space $swap_space"; fi
        log "  [DRY-RUN] Would start: vllm serve $model_path --port $port --tp $tp --gpu-mem $gpu_mem --quant $quant --max-seqs $max_seqs --max-model-len $max_model_len --max-batched-tokens $max_batched_tokens --block-size $block_size --dtype $dtype $_swap_flag $_exec_flag $_eager_flag"
        return 0
    fi

    if [[ ! -d "$model_path" ]]; then
        fail "Model directory not found: $model_path"
        return 1
    fi

    kill_tmux_session "bench-vllm"
    kill_server_by_name "vllm"
    if ! wait_for_process_exit "vllm serve " 3; then
        force_kill_by_name "VLLM::EngineCore"
        force_kill_by_name "vllm serve "
    fi

    local vllm_bin="${PROJECT_ROOT}/.venv/bin/vllm"
    if [[ ! -f "$vllm_bin" ]]; then
        vllm_bin=$(command -v vllm 2>/dev/null || true)
    fi
    if [[ -z "$vllm_bin" || ! -f "$vllm_bin" ]]; then
        fail "vllm binary not found (checked .venv/bin/vllm and PATH)"
        return 1
    fi

    local quant_str=""
    if [[ "$quant" != "none" ]]; then
        quant_str="--quantization $quant"
    fi

    local swap_str=""
    if [[ -n "$swap_space" && "$swap_space" -gt 0 ]]; then
        if $vllm_bin serve --help=all 2>/dev/null | grep -q "swap-space"; then
            swap_str="--swap-space $swap_space"
        else
            warn "  --swap-space not supported by this vLLM version, ignoring"
        fi
    fi

    local exec_str=""
    if [[ -n "$distributed_executor" ]]; then
        exec_str="--distributed-executor-backend $distributed_executor"
    fi

    local eager_str=""
    if [[ "$enforce_eager" == "true" || "$enforce_eager" == "True" ]]; then
        eager_str="--enforce-eager"
    fi

    wait_gpu_free 5000 30

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
            --block-size $block_size \
            --dtype $dtype \
            --trust-remote-code \
            $quant_str $swap_str $exec_str $eager_str 2>&1 | tee /tmp/bench_vllm.log"

    log "  Waiting for vLLM to be ready..."
    if wait_for_server "vLLM" "http://localhost:$port/v1/models" 300 "vllm serve"; then
        ok "vLLM ready on port $port"
        return 0
    else
        fail "vLLM failed to start on port $port"
        if [[ -f /tmp/bench_vllm.log ]]; then
            echo -e "  ${DIM}Last 10 lines of /tmp/bench_vllm.log:${NC}"
            tail -10 /tmp/bench_vllm.log | sed 's/^/    /'
        fi
        stop_server "vllm"
        return 1
    fi
}

# ── Start llama.cpp server ──────────────────────────────────────
start_llamacpp() {
    local model_path=$1 port=$2
    local np=${3:-4} ctx=${4:-4096} ngl=${5:-all} batch=${6:-2048} ubatch=${7:-512}
    local threads=${8:-0} ctk=${9:-q8_0} ctv=${10:-turbo4} fa=${11:-on} cache_prompt=${12:-true}
    local tensor_split=${13:-}
    log "Starting llama.cpp: model=$model_path port=$port np=$np ctx=$ctx ngl=$ngl batch=$batch ubatch=$ubatch ctk=$ctk ctv=$ctv fa=$fa cache_prompt=$cache_prompt ts=$tensor_split"

    if [[ $DRY_RUN -eq 1 ]]; then
        local _ts_flag=""
        if [[ -n "$tensor_split" ]]; then _ts_flag="--tensor-split $tensor_split"; fi
        log "  [DRY-RUN] Would start: llama-server -m $model_path --port $port -np $np -c $ctx -ngl $ngl -b $batch -ub $ubatch -t $threads -ctk $ctk -ctv $ctv -fa $fa --cache-prompt $_ts_flag"
        return 0
    fi

    if [[ ! -f "$model_path" ]]; then
        fail "Model file not found: $model_path"
        return 1
    fi

    kill_tmux_session "bench-llamacpp"
    kill_tmux_session "bench-llama"
    kill_server_by_name "llamacpp"
    if ! wait_for_process_exit "llama-server" 3; then
        force_kill_by_name "llama-server"
    fi

    local llama_bin="${PROJECT_ROOT}/third_party/llama-cpp-turboquant/build/bin/llama-server"
    if [[ ! -f "$llama_bin" ]]; then
        llama_bin=$(command -v llama-server 2>/dev/null || true)
    fi
    if [[ -z "$llama_bin" || ! -f "$llama_bin" ]]; then
        fail "llama-server binary not found"
        return 1
    fi

    # Resolve threads: 0 = auto (nproc)
    local threads_str=""
    if [[ "$threads" -gt 0 ]]; then
        threads_str="-t $threads"
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

    # Resolve tensor-split flag — only from explicit YAML config, no auto-detect
    # (orchestrator runs many models; auto-detect would force multi-GPU on tiny models)
    # For standalone use, see run-llamacpp.sh which auto-detects.
    local ts_flag=""
    if [[ -n "$tensor_split" ]]; then
        ts_flag="--tensor-split $tensor_split"
    fi

    wait_gpu_free 5000 30

    tmux new-session -d -s "bench-llamacpp" -n "serve" \
        "cd $PROJECT_ROOT && $llama_bin \
            -m $model_path \
            --host 0.0.0.0 --port $port \
            -np $np -c $ctx -ngl $ngl_value \
            -b $batch -ub $ubatch \
            $threads_str \
            -ctk $ctk -ctv $ctv -fa $fa \
            $cp_flag \
            $ts_flag \
            --metrics \
            2>&1 | tee /tmp/bench_llama.log"

    log "  Waiting for llama.cpp to be ready..."
    if wait_for_server "llama.cpp" "http://localhost:$port/health" 300 "llama-server"; then
        ok "llama.cpp ready on port $port"
        return 0
    else
        fail "llama.cpp failed to start on port $port"
        if [[ -f /tmp/bench_llama.log ]]; then
            echo -e "  ${DIM}Last 10 lines of /tmp/bench_llama.log:${NC}"
            tail -10 /tmp/bench_llama.log | sed 's/^/    /'
        fi
        stop_server "llamacpp"
        return 1
    fi
}

# ── Start SGLang server ──────────────────────────────────────────
start_sglang() {
    local model_path=$1 port=$2
    local mem_fraction=${3:-0.85} max_model_len=${4:-4096} gpu_count=${5:-1}
    log "Starting SGLang: model=$model_path port=$port mem_fraction=$mem_fraction max_model_len=$max_model_len gpu_count=$gpu_count"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would start: sglang.launch_server --model $model_path --host 0.0.0.0 --port $port --mem-fraction-static $mem_fraction --context-length $max_model_len --tp-size $gpu_count --attention-backend flashinfer --chunked-prefill-size 8192 --max-running-requests 2048 --enable-torch-compile --trust-remote-code"
        return 0
    fi

    if [[ ! -d "$model_path" ]]; then
        fail "Model directory not found: $model_path"
        return 1
    fi

    kill_tmux_session "bench-sglang"
    kill_server_by_name "sglang"
    if ! wait_for_process_exit "sglang.launch_server" 3; then
        force_kill_by_name "sglang.launch_server"
    fi

    local sglang_python="/opt/venv-sglang/bin/python"
    if [[ ! -f "$sglang_python" ]]; then
        sglang_python="${PROJECT_ROOT}/.venv/bin/python"
    fi
    if [[ ! -f "$sglang_python" ]]; then
        sglang_python=$(command -v python3 2>/dev/null || true)
    fi

    wait_gpu_free 5000 30

    tmux new-session -d -s "bench-sglang" -n "serve" \
        "$sglang_python -m sglang.launch_server \
            --model $model_path \
            --host 0.0.0.0 --port $port \
            --mem-fraction-static $mem_fraction \
            --context-length $max_model_len \
            --tp-size $gpu_count \
            --attention-backend flashinfer \
            --chunked-prefill-size 8192 \
            --max-running-requests 2048 \
            --enable-torch-compile \
            --trust-remote-code \
            2>&1 | tee /tmp/bench_sglang.log"

    log "  Waiting for SGLang to be ready..."
    if wait_for_server "SGLang" "http://localhost:$port/v1/models" 300 "sglang.launch_server"; then
        ok "SGLang ready on port $port"
        return 0
    else
        fail "SGLang failed to start on port $port"
        if [[ -f /tmp/bench_sglang.log ]]; then
            echo -e "  ${DIM}Last 10 lines of /tmp/bench_sglang.log:${NC}"
            tail -10 /tmp/bench_sglang.log | sed 's/^/    /'
        fi
        stop_server "sglang"
        return 1
    fi
}

# ── Stop server ─────────────────────────────────────────────────
stop_server() {
    local fw=$1
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would stop $fw"
        return 0
    fi

    log "  Stopping $fw..."

    local session="bench-${fw}"

    kill_tmux_session "$session"
    if [[ "$fw" == "llamacpp" ]]; then
        kill_tmux_session "bench-llama"
    fi

    kill_server_by_name "$fw"

    if ! wait_for_process_exit "bench-${fw}" 3; then
        if [[ "$fw" == "vllm" ]]; then
            force_kill_by_name "VLLM::EngineCore"
            force_kill_by_name "vllm serve "
        elif [[ "$fw" == "llamacpp" ]]; then
            force_kill_by_name "llama-server"
        elif [[ "$fw" == "sglang" ]]; then
            force_kill_by_name "sglang.launch_server"
        fi
    fi

    sleep 1
    ok "$fw stopped"

    wait_gpu_free 5000 60
}

# ── Run benchmark ───────────────────────────────────────────────
run_benchmark() {
    local fw=$1 model_name=$2 model_path=$3 port=$4 results_dir=$5 repo_id=$6
    local bench_script="${PROJECT_ROOT}/scripts/benchmark/${fw}_bench.sh"

    if [[ ! -f "$bench_script" ]]; then
        warn "Benchmark script not found: $bench_script"
        return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] Would run: $bench_script -u http://localhost:$port/v1 -o $results_dir --full"
        return 0
    fi

    log "  Running $fw benchmark for $model_name..."

    mkdir -p "$results_dir"

    # Build model flag: use HF repo_id for tokenization if available
    local model_flag=()
    if [[ -n "$repo_id" ]]; then
        model_flag=(-m "$repo_id")
    fi

    # Run the per-backend benchmark script with --full (llama-benchy + native dataset)
    # The script handles its own session directory creation
    "$bench_script" \
        -u "http://localhost:$port/v1" \
        -o "$results_dir" \
        "${model_flag[@]}" \
        --full 2>&1 | tee -a "$LOG_FILE"
}

# ── Main ────────────────────────────────────────────────────────
# Auto-create session folder: YYYYMMDD_benchmark
# Structure: results/YYYYMMDD_benchmark/{vllm,llamacpp}_run/run-N/
if [[ -z "${BENCH_RESULTS_DIR:-}" ]]; then
    SESSION_STAMP="$(date +%Y%m%d_%H%M%S)"
    RESULTS_DIR="${PROJECT_ROOT}/results/${SESSION_STAMP}_benchmark"
fi
LOG_FILE="${RESULTS_DIR}/bench.log"
mkdir -p "$RESULTS_DIR"

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Per-Model Benchmark Orchestrator${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Phase${NC}        $PHASE"
echo -e "  ${CYAN}Bench Phase${NC} $BENCH_PHASE"
echo -e "  ${CYAN}Bench Type${NC}  $BENCH_TYPE"
if [[ "$BENCH_TYPE" == "stress" ]]; then
echo -e "  ${CYAN}Conc range${NC}  ${CONC_BASE} → ${CONC_MAX} (step ${CONC_STEP})"
fi
echo -e "  ${CYAN}Results${NC}      $RESULTS_DIR"
echo -e "  ${CYAN}Config${NC}       $YAML_CONFIG"
[[ $DRY_RUN -eq 1 ]] && echo -e "  ${YELLOW}Mode${NC}         DRY-RUN"
echo ""

# Parse models
mapfile -t ALL_LINES < <(parse_models "$YAML_CONFIG" "$PHASE" "${SELECTED_MODELS[@]+"${SELECTED_MODELS[@]}"}")

# Extract dataset path (first line)
DATASET_PATH="${BENCH_DATASET_PATH:-/workspace/datasets/sharegpt.json}"
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
    IFS='|' read -r name repo_id backend model_path m_phase vllm_tp vllm_gpu_mem vllm_quant vllm_max_seqs gpu_count skip vllm_max_model_len vllm_max_batched_tokens vllm_block_size vllm_dtype llama_np llama_ctx llama_ngl llama_batch llama_ubatch llama_threads llama_ctk llama_ctv llama_fa llama_cp vllm_swap_space vllm_distributed_executor vllm_enforce_eager llama_ts sglang_mem sglang_max_len <<< "$line"
    if [[ ${#BACKEND_FILTER[@]} -gt 0 ]]; then
        match=0
        for bf in "${BACKEND_FILTER[@]}"; do
            if [[ "$backend" == "$bf" ]]; then match=1; break; fi
        done
        [[ $match -eq 0 ]] && continue
    fi
    if [[ -n "$skip" ]]; then
        warn "  - $name (backend=$backend, phase=$m_phase, tp=$vllm_tp) — $skip (pod has $gpu_count GPU(s))"
    elif [[ "$backend" == "vllm" ]]; then
        log "  - $name (vllm): repo=$repo_id tp=$vllm_tp gpu_mem=$vllm_gpu_mem quant=$vllm_quant max_seqs=$vllm_max_seqs block=$vllm_block_size dtype=$vllm_dtype swap=$vllm_swap_space exec=$vllm_distributed_executor eager=$vllm_enforce_eager"
    elif [[ "$backend" == "sglang" ]]; then
        log "  - $name (sglang): repo=$repo_id mem_fraction=$sglang_mem max_model_len=$sglang_max_len"
    else
        log "  - $name (llamacpp): repo=$repo_id np=$llama_np ctx=$llama_ctx ngl=$llama_ngl batch=$llama_batch ubatch=$llama_ubatch ctk=$llama_ctk ctv=$llama_ctv fa=$llama_fa cache_prompt=$llama_cp ts=$llama_ts"
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
    IFS='|' read -r name repo_id backend model_path m_phase vllm_tp vllm_gpu_mem vllm_quant vllm_max_seqs gpu_count skip vllm_max_model_len vllm_max_batched_tokens vllm_block_size vllm_dtype llama_np llama_ctx llama_ngl llama_batch llama_ubatch llama_threads llama_ctk llama_ctv llama_fa llama_cp vllm_swap_space vllm_distributed_executor vllm_enforce_eager llama_ts sglang_mem sglang_max_len <<< "$line"

    if [[ ${#BACKEND_FILTER[@]} -gt 0 ]]; then
        match=0
        for bf in "${BACKEND_FILTER[@]}"; do
            if [[ "$backend" == "$bf" ]]; then
                match=1
                break
            fi
        done
        if [[ $match -eq 0 ]]; then
            continue
        fi
    fi

    TOTAL=$((TOTAL + 1))

    if [[ -n "$skip" ]]; then
        warn "Skipping $name — $skip"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    RUN_NUM=$((RUN_NUM + 1))
    RUN_DIR="${RESULTS_DIR}/${backend}_run/run-${RUN_NUM}"
    header "Run $RUN_NUM: $name ($backend, $m_phase)"

    log "  Results: $RUN_DIR/"
    mkdir -p "$RUN_DIR"

    PORT=8000
    if [[ "$backend" == "llamacpp" ]]; then
        PORT=8001
    elif [[ "$backend" == "sglang" ]]; then
        PORT=8002
    fi

    if [[ "$backend" == "vllm" ]]; then
        if ! start_vllm "$model_path" "$PORT" "$vllm_tp" "$vllm_gpu_mem" "$vllm_quant" "$vllm_max_seqs" "$vllm_max_model_len" "$vllm_max_batched_tokens" "$vllm_block_size" "$vllm_dtype" "$vllm_swap_space" "$vllm_distributed_executor" "$vllm_enforce_eager"; then
            echo "FAILED: server did not start" > "$RUN_DIR/FAILED"
            [[ ! " ${FAILED_LIST[*]:-} " =~ " ${name} " ]] && FAILED_LIST+=("$name")
            continue
        fi
    elif [[ "$backend" == "llamacpp" ]]; then
        if ! start_llamacpp "$model_path" "$PORT" "$llama_np" "$llama_ctx" "$llama_ngl" "$llama_batch" "$llama_ubatch" "$llama_threads" "$llama_ctk" "$llama_ctv" "$llama_fa" "$llama_cp" "$llama_ts"; then
            echo "FAILED: server did not start" > "$RUN_DIR/FAILED"
            [[ ! " ${FAILED_LIST[*]:-} " =~ " ${name} " ]] && FAILED_LIST+=("$name")
            continue
        fi
    elif [[ "$backend" == "sglang" ]]; then
        if ! start_sglang "$model_path" "$PORT" "$sglang_mem" "$sglang_max_len" "$gpu_count"; then
            echo "FAILED: server did not start" > "$RUN_DIR/FAILED"
            [[ ! " ${FAILED_LIST[*]:-} " =~ " ${name} " ]] && FAILED_LIST+=("$name")
            continue
        fi
    else
        warn "Unknown backend: $backend, skipping"
        echo "FAILED: unknown backend" > "$RUN_DIR/FAILED"
        [[ ! " ${FAILED_LIST[*]:-} " =~ " ${name} " ]] && FAILED_LIST+=("$name")
        continue
    fi

    BENCH_DIR="$RUN_DIR"
    gpu_log="$RUN_DIR/gpu_util.log"
    echo -e "timestamp\tgpu\tsm_util\tmem_util\tenc_util\tdec_util\tmem_used\tmem_total\ttemp\tpower" > "$gpu_log"
    start_gpu_monitor "$gpu_log"
    if run_benchmark "$backend" "$name" "$model_path" "$PORT" "$BENCH_DIR" "$repo_id"; then
        PASSED=$((PASSED + 1))
        ok "Run $RUN_NUM complete: $name"
    else
        echo "FAILED: benchmark error" > "$RUN_DIR/FAILED"
        [[ ! " ${FAILED_LIST[*]:-} " =~ " ${name} " ]] && FAILED_LIST+=("$name")
        fail "Run $RUN_NUM failed: $name"
    fi
    stop_gpu_monitor

    stop_server "$backend"
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
for backend_dir in "$RESULTS_DIR"/*_run/; do
    [[ -d "$backend_dir" ]] || continue
    backend_name=$(basename "$backend_dir")
    echo -e "    ${CYAN}$backend_name/${NC}"
    for run_dir in "$backend_dir"run-*/; do
        [[ -d "$run_dir" ]] || continue
        run_name=$(basename "$run_dir")
        if [[ -f "$run_dir/FAILED" ]]; then
            echo -e "      ${RED}$run_name/ (FAILED)${NC}"
        else
            count=$(find "$run_dir" -name "*.tsv" -o -name "*.json" 2>/dev/null | wc -l)
            echo -e "      $run_name/ ($count files)"
        fi
    done
done
echo ""
echo -e "  Log: ${CYAN}$LOG_FILE${NC}"
echo ""

# Parse results
if [[ $DRY_RUN -eq 0 ]]; then
    local parse_script="${PROJECT_ROOT}/scripts/parse_bench.py"
    if [[ -f "$parse_script" ]]; then
        log "Parsing results..."
        python3 "$parse_script" "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"
    fi
    local viz_script="${PROJECT_ROOT}/scripts/visualize_cross_sweep.py"
    if [[ -f "$viz_script" ]]; then
        log "Generating visualizations..."
        for backend_dir in "$RESULTS_DIR"/*_run/; do
            [[ -d "$backend_dir" ]] || continue
            python3 "$viz_script" "$backend_dir" 2>&1 | tee -a "$LOG_FILE" || true
        done
    fi
fi
