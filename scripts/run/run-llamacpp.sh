#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists (does not override existing env vars)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BINARY="${PROJECT_ROOT}/third_party/llama-cpp-turboquant/build/bin/llama-server"

DEFAULT_MODEL="${LLAMA_MODEL:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODEL_PATH]

Start llama-cpp-turboquant server with TurboQuant KV cache support.

POSITIONAL:
  MODEL_PATH              Path to GGUF model (default: $DEFAULT_MODEL)

OPTIONS:
  -p, --port PORT         Server port (default: 8003)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -n, --ccu N             Concurrent slots / CCU (default: 4)
  -c, --context N         Total context size (default: 4096)
  -ng, --gpu-layers N     GPU layers offload: number or 'all' (default: all)
  -ts, --tensor-split S   Comma-separated GPU split ratios (e.g. "1,1,1,1,1,1" for 6 GPUs)
  -b, --batch N           Batch size for prefill (default: 2048)
  -ub, --ubatch N         Micro-batch size (default: 512)
  -t, --threads N         CPU threads (default: nproc)
  -fa, --flash-attn VAL   Flash Attention: on/off/auto (default: on)
  -ctk, --cache-key TYPE  KV cache key type (default: q8_0)
  -ctv, --cache-val TYPE  KV cache value type (default: turbo4)
  --cache-prompt          Enable prompt caching (default)
  --no-cache-prompt       Disable prompt caching
    -h, --help              Show this help

TURBOQUANT KV CACHE TYPES:
  -ctv turbo4             TurboQuant 4-bit value cache (~50% VRAM savings)
  -ctv turbo3             TurboQuant 3-bit value cache (~60% VRAM savings)
  -ctk q8_0              Standard 8-bit key cache (default)
  -ctk f16               Full precision key cache

EXAMPLES:
  $(basename "$0")                                    # Defaults (turbo4)
  $(basename "$0") -n 8 -c 32768 -ctv turbo4         # 8 CCU, 32K, turbo4
  $(basename "$0") --ccu 1 --context 65536 -ctv turbo3  # 1 user, 64K, turbo3
  $(basename "$0") -p 8002 -n 16 model.gguf          # Custom port + model

ENV OVERRIDES (lower priority than flags):
  LLAMA_PORT, LLAMA_HOST, LLAMA_N_PARALLEL, LLAMA_CTX_SIZE,
  LLAMA_N_GPU_LAYERS, LLAMA_TENSOR_SPLIT, LLAMA_BATCH, LLAMA_UBATCH, LLAMA_THREADS,
  LLAMA_FLASH_ATTN, LLAMA_CACHE_PROMPT, LLAMA_CACHE_KEY, LLAMA_CACHE_VAL
EOF
    exit 0
}

MODEL=""
PORT="${LLAMA_PORT:-8001}"
HOST="${LLAMA_HOST:-0.0.0.0}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-all}"
N_PARALLEL="${LLAMA_N_PARALLEL:-8}"
CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-all}"
N_BATCH="${LLAMA_BATCH:-4096}"
N_UBATCH="${LLAMA_UBATCH:-1024}"
N_THREADS="${LLAMA_THREADS:-60}"
TENSOR_SPLIT="${LLAMA_TENSOR_SPLIT:-}"
FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
CACHE_KEY="${LLAMA_CACHE_KEY:-q8_0}"
CACHE_VAL="${LLAMA_CACHE_VAL:-turbo4}"
CACHE_PROMPT="${LLAMA_CACHE_PROMPT:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)          PORT="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -n|--ccu)           N_PARALLEL="$2"; shift 2 ;;
        -c|--context)       CTX_SIZE="$2"; shift 2 ;;
        -ng|--gpu-layers)   N_GPU_LAYERS="$2"; shift 2 ;;
        -ts|--tensor-split) TENSOR_SPLIT="$2"; shift 2 ;;
        -b|--batch)         N_BATCH="$2"; shift 2 ;;
        -ub|--ubatch)       N_UBATCH="$2"; shift 2 ;;
        -t|--threads)       N_THREADS="$2"; shift 2 ;;
        -fa|--flash-attn)   FLASH_ATTN="$2"; shift 2 ;;
        -ctk|--cache-key)   CACHE_KEY="$2"; shift 2 ;;
        -ctv|--cache-val)   CACHE_VAL="$2"; shift 2 ;;
        --cache-prompt)     CACHE_PROMPT=1; shift ;;
        --no-cache-prompt)  CACHE_PROMPT=0; shift ;;
        -h|--help)          usage ;;
        -*)                 echo "Unknown option: $1" >&2; usage ;;
        *)                  MODEL="$1"; shift ;;
    esac
done

MODEL="${MODEL:-$DEFAULT_MODEL}"
SLOT_CTX=$(( CTX_SIZE / N_PARALLEL ))

GPU_NAME="N/A"
GPU_COUNT=0
GPU_VRAM="N/A"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
fi

CPU_NAME=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "N/A")
CPU_CORES=$(nproc 2>/dev/null || echo "?")

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: turboquant binary not found at: $BINARY" >&2
    echo "Run: ./scripts/build/build-llamacpp-turbo.sh" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model file not found at: $MODEL" >&2
    echo "Usage: $0 [OPTIONS] [MODEL_PATH]" >&2
    exit 1
fi

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

row() { printf "  ${CYAN}%-14s${NC} %s\n" "$1" "$2"; }
sep() { echo -e "  ${DIM}$(printf '%.0s─' {1..48})${NC}"; }

echo ""
echo -e "  ${YELLOW}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${YELLOW}${BOLD}  llama.cpp-turbo${NC}${DIM} — TurboQuant KV Cache Engine${NC}"
echo -e "  ${YELLOW}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Model"      "$MODEL"
row "Endpoint"   "http://${HOST}:${PORT}"
sep
row "CCU Slots"  "${N_PARALLEL} concurrent"
row "Context"    "${CTX_SIZE} total → ${SLOT_CTX}/slot"
row "GPU Layers" "$N_GPU_LAYERS"
row "Batch"      "${N_BATCH} prefill / ${N_UBATCH} micro"
row "Threads"    "${N_THREADS} / ${CPU_CORES} cores"
row "Flash Attn" "$FLASH_ATTN"
row "KV Cache"   "K=${CACHE_KEY} V=${CACHE_VAL} (TurboQuant)"
row "Cache"      "$( [[ "$CACHE_PROMPT" == "1" ]] && echo "enabled" || echo "disabled" )"
sep
row "GPU"        "${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
row "CPU"        "${CPU_NAME}"
echo -e "  ${YELLOW}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

# Auto-detect GPU count for tensor-split if not set and offloading all layers
RESOLVED_TENSOR_SPLIT="$TENSOR_SPLIT"
if [[ -z "$RESOLVED_TENSOR_SPLIT" && ("$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" == "999") ]]; then
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo "1")
        if [[ "$gpu_count" -gt 1 ]]; then
            RESOLVED_TENSOR_SPLIT=$(printf '%0.s1,' $(seq 1 "$gpu_count"))
            RESOLVED_TENSOR_SPLIT="${RESOLVED_TENSOR_SPLIT%,}"
            echo -e "  ${YELLOW}Auto-detected $gpu_count GPUs → tensor-split=$RESOLVED_TENSOR_SPLIT${NC}"
        fi
    fi
fi

exec "$BINARY" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np "$N_PARALLEL" \
    -c "$CTX_SIZE" \
    -b "$N_BATCH" \
    -ub "$N_UBATCH" \
    -t "$N_THREADS" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_KEY" \
    -ctv "$CACHE_VAL" \
    ${RESOLVED_TENSOR_SPLIT:+--tensor-split "$RESOLVED_TENSOR_SPLIT"} \
    $( [[ "$CACHE_PROMPT" == "1" ]] && echo "--cache-prompt" || echo "--no-cache-prompt" )
