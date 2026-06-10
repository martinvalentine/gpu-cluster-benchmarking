#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SESSION="llm-servers"

resolve_model_paths() {
    local yaml_path="${PROJECT_ROOT}/configs/models.yaml"
    if [[ ! -f "$yaml_path" ]]; then
        echo "ERROR: Config not found: $yaml_path" >&2
        return 1
    fi

    local resolved
    resolved=$(uv run python - "$yaml_path" <<'PYEOF'
import sys, yaml
from pathlib import Path

config_path = Path(sys.argv[1])
with open(config_path) as f:
    cfg = yaml.safe_load(f)

base_dir = Path(cfg.get("base_dir", "models"))
models = cfg.get("models", [])

vllm_path = ""
llama_path = ""
embed_path = ""

for m in models:
    if not m.get("enabled", True):
        continue
    backend = m.get("backend", "vllm")
    local_dir = m.get("local_dir", "")
    phase = m.get("phase", "")

    if backend == "vllm" and not vllm_path:
        vllm_path = str(base_dir / local_dir)
    elif backend == "llamacpp" and phase == "embedding" and not embed_path:
        embed_path = str(base_dir / local_dir)
    elif backend == "llamacpp" and phase != "embedding" and not llama_path:
        include = m.get("include", "*.gguf")
        gguf_dir = base_dir / local_dir
        if gguf_dir.exists():
            from fnmatch import fnmatch
            matches = sorted(f for f in gguf_dir.iterdir() if fnmatch(f.name, include))
            if matches:
                llama_path = str(matches[0])
            else:
                llama_path = str(gguf_dir / f"{Path(local_dir).name}.gguf")
        else:
            llama_path = str(gguf_dir / f"{Path(local_dir).name}.gguf")

if vllm_path:
    print(f"VLLM_MODEL={vllm_path}")
if llama_path:
    print(f"LLAMA_MODEL={llama_path}")
if embed_path:
    print(f"EMBED_MODEL={embed_path}")
PYEOF
    ) || return 0

    while IFS='=' read -r key val; do
        case "$key" in
            VLLM_MODEL)  VLLM_MODEL="$val" ;;
            LLAMA_MODEL) LLAMA_MODEL="$val" ;;
            EMBED_MODEL) EMBED_MODEL="$val" ;;
        esac
    done <<< "$resolved"
}

resolve_model_paths

VLLM_MODEL="${VLLM_MODEL:-${PROJECT_ROOT}/models/hf/qwen2.5-0.6b}"
LLAMA_MODEL="${LLAMA_MODEL:-${PROJECT_ROOT}/models/gguf/qwen2.5-0.6b/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
EMBED_MODEL="${EMBED_MODEL:-${PROJECT_ROOT}/models/gguf/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf}"

TP=${VLLM_TP:-1}
GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.15}
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}

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

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Starting all LLM servers in tmux${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────
echo -e "  ${DIM}Pre-flight checks...${NC}"

MISSING=0

check_model() {
    local path="$1" name="$2"
    if [[ -f "$path" ]]; then
        ok "$name: $path"
    else
        fail "$name: not found at $path"
        MISSING=$((MISSING + 1))
    fi
}

check_model "$VLLM_MODEL"  "vLLM model (HF)"
check_model "$LLAMA_MODEL" "llama.cpp model (GGUF)"
check_model "$EMBED_MODEL" "Embedding model (GGUF)"

if [[ $MISSING -gt 0 ]]; then
    echo ""
    echo -e "  ${RED}${BOLD}${MISSING} model(s) missing.${NC}"
    echo -e "  Download them first:"
    echo -e "    ${CYAN}uv run python scripts/download-models.py${NC}"
    echo -e "  Or specific models:"
    echo -e "    ${CYAN}uv run python scripts/download-models.py --only qwen0.5b qwen0.5b-gguf qwen3-embedding${NC}"
    echo ""
    exit 1
fi

echo ""

# ── Check Redis ────────────────────────────────────────────────
if redis-cli ping &>/dev/null 2>&1; then
    ok "Redis running"
else
    log "Starting Redis..."
    redis-server --daemonize yes --maxmemory 8gb --maxmemory-policy allkeys-lru --logfile /tmp/redis.log 2>/dev/null || true
    sleep 1
    redis-cli ping &>/dev/null 2>&1 && ok "Redis started" || fail "Redis failed"
fi

echo ""

# ── Kill existing session ──────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# ── Start servers ──────────────────────────────────────────────
log "Starting servers..."

tmux new-session -d -s "$SESSION" -n "vllm" \
    "cd $PROJECT_ROOT && uv run vllm serve $VLLM_MODEL \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --enforce-eager 2>&1 | tee /tmp/vllm.log; sleep infinity"
ok "vLLM window created (port 8000)"

tmux new-window -t "$SESSION" -n "llama" \
    "cd $PROJECT_ROOT && bash scripts/run/run-llamacpp.sh \
        -p 8001 -n 4 -c 4096 -ng all \
        $LLAMA_MODEL 2>&1 | tee /tmp/llama.log; sleep infinity"
ok "llama.cpp window created (port 8001)"

tmux new-window -t "$SESSION" -n "embed" \
    "cd $PROJECT_ROOT && bash scripts/run/run-embedding-server.sh \
        -p 8003 -ng all -c 4096 -np last -ccu 4 \
        $EMBED_MODEL 2>&1 | tee /tmp/embed.log; sleep infinity"
ok "Embedding window created (port 8003)"

tmux new-window -t "$SESSION" -n "proxy" \
    "cd $PROJECT_ROOT && uv run python scripts/gen-litellm-config.py && sleep 3 && uv run litellm \
        --config litellm_config.yaml \
        --port 4000 --host 0.0.0.0 2>&1 | tee /tmp/proxy.log; sleep infinity"
ok "LiteLLM proxy window created (port 4000)"

echo ""
echo -e "  ${CYAN}tmux session: ${BOLD}$SESSION${NC}"
echo ""
echo -e "  ${DIM}Windows:${NC}"
echo -e "  ${CYAN}0: vllm${NC}     → http://localhost:8000"
echo -e "  ${CYAN}1: llama${NC}    → http://localhost:8001"
echo -e "  ${CYAN}2: embed${NC}    → http://localhost:8003"
echo -e "  ${CYAN}3: proxy${NC}    → http://localhost:4000"
echo ""
echo -e "  ${DIM}Attach:    tmux attach -t $SESSION${NC}"
echo -e "  ${DIM}Switch:    Ctrl-b + 0/1/2/3${NC}"
echo -e "  ${DIM}Detach:    Ctrl-b + d${NC}"
echo -e "  ${DIM}Logs:      /tmp/vllm.log, /tmp/llama.log, /tmp/embed.log, /tmp/proxy.log${NC}"
echo ""
echo -e "  ${DIM}Wait ~15s for all services, then run:${NC}"
echo -e "  ${CYAN}./scripts/test-all.sh${NC}"
echo ""
