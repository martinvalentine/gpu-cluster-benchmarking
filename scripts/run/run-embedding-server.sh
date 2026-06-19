#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

BINARY="${LLAMA_EMBED_BINARY:-${PROJECT_ROOT}/third_party/llama-cpp-turboquant/build/bin/llama-server}"
DEFAULT_MODEL="${LLAMA_EMBED_MODEL:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODEL_PATH]

Start llama-cpp-turboquant as embedding server for LiteLLM semantic cache.
Uses llama-server --embedding (OpenAI-compatible /v1/embeddings endpoint).

POSITIONAL:
  MODEL_PATH              Path to GGUF embedding model (default: $DEFAULT_MODEL)

OPTIONS:
  -p, --port PORT         Server port (default: 8003)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -ng, --gpu-layers N     GPU layers (default: 999 = all)
  -c, --context N         Context size (default: 32768)
  -np, --pooling TYPE     Pooling: last, mean, cls (default: last)
  -t, --threads N         CPU threads (default: nproc)
  -ccu, --concurrent N    Concurrent slots (default: 20)
  -v, --verbose           Enable verbose logging
  -h, --help              Show this help

ENV OVERRIDES:
  LLAMA_EMBED_BINARY, LLAMA_EMBED_MODEL, LLAMA_EMBED_PORT,
  LLAMA_EMBED_HOST, LLAMA_EMBED_GPU_LAYERS, LLAMA_EMBED_CONTEXT,
  LLAMA_EMBED_POOLING, LLAMA_EMBED_THREADS, LLAMA_EMBED_CONCURRENT

EXAMPLES:
  $(basename "$0")                                              # Defaults
  $(basename "$0") -p 12101 --gpu-layers 999                   # All layers on GPU
  $(basename "$0") /path/to/embedding-model.gguf               # Custom model
  $(basename "$0") --pooling mean --context 8192                # Mean pooling

AFTER STARTING, update .env:
  EMBEDDING_API_BASE=http://localhost:12101/v1
  EMBEDDING_MODEL=embedding-model
  EMBEDDING_API_KEY=EMPTY

Then start LiteLLM:
  litellm --config litellm_config.yaml --port 4000
EOF
    exit 0
}

MODEL=""
PORT="${LLAMA_EMBED_PORT:-8003}"
HOST="${LLAMA_EMBED_HOST:-0.0.0.0}"
N_GPU_LAYERS="${LLAMA_EMBED_GPU_LAYERS:-999}"
CTX_SIZE="${LLAMA_EMBED_CONTEXT:-32768}"
POOLING="${LLAMA_EMBED_POOLING:-last}"
N_THREADS="${LLAMA_EMBED_THREADS:-$(nproc)}"
N_PARALLEL="${LLAMA_EMBED_CONCURRENT:-20}"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)          PORT="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -ng|--gpu-layers)   N_GPU_LAYERS="$2"; shift 2 ;;
        -c|--context)       CTX_SIZE="$2"; shift 2 ;;
        -np|--pooling)      POOLING="$2"; shift 2 ;;
        -t|--threads)       N_THREADS="$2"; shift 2 ;;
        -ccu|--concurrent)  N_PARALLEL="$2"; shift 2 ;;
        -v|--verbose)       VERBOSE="--verbose"; shift ;;
        -h|--help)          usage ;;
        -*)                 echo "Unknown option: $1" >&2; usage ;;
        *)                  MODEL="$1"; shift ;;
    esac
done

MODEL="${MODEL:-$DEFAULT_MODEL}"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: llama-server binary not found at: $BINARY" >&2
    echo "Run: ./scripts/build/build-llamacpp-turbo.sh" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model file not found at: $MODEL" >&2
    echo "Usage: $0 [OPTIONS] [MODEL_PATH]" >&2
    exit 1
fi

GPU_NAME="N/A"
GPU_COUNT=0
GPU_VRAM="N/A"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
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
echo -e "  ${YELLOW}${BOLD}  Embedding Server${NC}${DIM} — llama-cpp-turboquant${NC}"
echo -e "  ${YELLOW}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Model"      "$MODEL"
row "Endpoint"   "http://${HOST}:${PORT}/v1/embeddings"
row "Health"     "http://${HOST}:${PORT}/health"
sep
row "GPU Layers" "$N_GPU_LAYERS"
row "Context"    "$CTX_SIZE"
row "Pooling"    "$POOLING"
row "Concurrent" "$N_PARALLEL slots"
row "Threads"    "$N_THREADS"
sep
row "GPU"        "${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
echo -e "  ${YELLOW}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${DIM}After starting, update .env:${NC}"
echo -e "  ${CYAN}EMBEDDING_API_BASE=http://localhost:${PORT}/v1${NC}"
echo -e "  ${CYAN}EMBEDDING_MODEL=embedding-model${NC}"
echo -e "  ${CYAN}EMBEDDING_API_KEY=EMPTY${NC}"
echo ""

exec "$BINARY" \
    -m "$MODEL" \
    --embedding \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$N_GPU_LAYERS" \
    -c "$CTX_SIZE" \
    --pooling "$POOLING" \
    -np "$N_PARALLEL" \
    -t "$N_THREADS" \
    $VERBOSE
