#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

BINARY="${PROJECT_ROOT}/third_party/llama.cpp/build/bin/llama-server"

DEFAULT_MODEL="/home/metflow/AI-Infra/models/Qwen3.6-35B-A3B-UDT-Q5_K_XL_MTP.gguf"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODEL_PATH]

Start llama.cpp server with configurable parameters (benchmark_plan.md B.2).

POSITIONAL:
  MODEL_PATH              Path to GGUF model (default: $DEFAULT_MODEL)

OPTIONS:
  -p, --port PORT         Server port (default: 8001)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -n, --ccu N             Concurrent slots / CCU (default: 4)
  -c, --context N         Total context size (default: 4096)
  -ng, --gpu-layers N     GPU layers offload: number or 'all' (default: all)
  -b, --batch N           Batch size for prefill (default: 2048)
  -ub, --ubatch N         Micro-batch size (default: 512)
  -t, --threads N         CPU threads (default: nproc)
  -fa, --flash-attn VAL   Flash Attention: on/off/auto (default: on)
  -ctk, --cache-key TYPE  KV cache key type (default: f16)
  -ctv, --cache-val TYPE  KV cache value type (default: f16)
  --cache-prompt          Enable prompt caching (default)
  --no-cache-prompt       Disable prompt caching
  -h, --help              Show this help

EXAMPLES:
  $(basename "$0")                                    # Defaults (f16 KV cache)
  $(basename "$0") -n 8 -c 32768                     # 8 CCU, 32K context
  $(basename "$0") --ccu 1 --context 65536            # 1 user, 64K context
  $(basename "$0") -p 8002 -n 16 -c 16384 model.gguf # Custom port + model
  $(basename "$0") -ctk q8_0 -ctv q4_0               # Quantized KV cache

KV CACHE TYPES (upstream llama.cpp, saves VRAM vs f16):
  -ctk f16  -ctv f16      No savings (default, highest quality)
  -ctk q8_0 -ctv q8_0     ~35% VRAM savings
  -ctk q8_0 -ctv q4_0     ~50% VRAM savings

NOTE: For TurboQuant KV cache (turbo3/turbo4), use run-llamacpp-turbo.sh

ENV OVERRIDES (lower priority than flags):
  LLAMA_PORT, LLAMA_HOST, LLAMA_N_PARALLEL, LLAMA_CTX_SIZE,
  LLAMA_N_GPU_LAYERS, LLAMA_BATCH, LLAMA_UBATCH, LLAMA_THREADS,
  LLAMA_FLASH_ATTN, LLAMA_CACHE_PROMPT, LLAMA_CACHE_KEY, LLAMA_CACHE_VAL
EOF
    exit 0
}

MODEL=""
PORT="${LLAMA_PORT:-8001}"
HOST="${LLAMA_HOST:-0.0.0.0}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-all}"
N_PARALLEL="${LLAMA_N_PARALLEL:-4}"
CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
N_BATCH="${LLAMA_BATCH:-2048}"
N_UBATCH="${LLAMA_UBATCH:-512}"
N_THREADS="${LLAMA_THREADS:-$(nproc)}"
FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
CACHE_KEY="${LLAMA_CACHE_KEY:-f16}"
CACHE_VAL="${LLAMA_CACHE_VAL:-f16}"
CACHE_PROMPT="${LLAMA_CACHE_PROMPT:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)          PORT="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -n|--ccu)           N_PARALLEL="$2"; shift 2 ;;
        -c|--context)       CTX_SIZE="$2"; shift 2 ;;
        -ng|--gpu-layers)   N_GPU_LAYERS="$2"; shift 2 ;;
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
    echo "ERROR: llama-server binary not found at: $BINARY" >&2
    echo "Run: ./scripts/build/build-llamacpp.sh" >&2
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
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  llama.cpp${NC}${DIM} — GPU Inference Engine${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
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
row "KV Cache"   "K=${CACHE_KEY} V=${CACHE_VAL} (upstream)"
row "Cache"      "$( [[ "$CACHE_PROMPT" == "1" ]] && echo "enabled" || echo "disabled" )"
sep
row "GPU"        "${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
row "CPU"        "${CPU_NAME}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

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
    $( [[ "$CACHE_PROMPT" == "1" ]] && echo "--cache-prompt" || echo "--no-cache-prompt" )
