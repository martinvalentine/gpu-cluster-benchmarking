#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load .env if exists (does not override existing env vars)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

DEFAULT_MODEL="${VLLM_MODEL:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODEL_PATH]

Start SGLang server with RadixAttention (benchmark_plan.md B.3).

POSITIONAL:
  MODEL_PATH              HF model path or local dir (default: $DEFAULT_MODEL)

OPTIONS:
  -p, --port PORT         Server port (default: 8002)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -tp, --tp N             Tensor parallel size (default: 6)
  -mfs, --mem-frac F      Static memory fraction (default: 0.87)
  -mtt, --max-total-tok N Max total tokens for KV cache (default: 1048576)
  -cps, --chunked-ps N    Chunked prefill size (default: 8192)
  -attn, --attn-backend B Attention backend: flashinfer, triton (default: flashinfer)
  -q, --quant TYPE        Quantization: awq, gptq, none (default: awq)
  -mrr, --max-running N   Max running requests (default: 256)
  --enable-torch-compile  Enable torch.compile (default)
  --no-torch-compile      Disable torch.compile
  --disable-radix-cache   Disable RadixAttention cache
  --trust-remote-code     Trust remote code (default)
  -h, --help              Show this help

ENV OVERRIDES:
  SGLANG_PORT, SGLANG_HOST, SGLANG_TP, SGLANG_MEM_FRAC,
  SGLANG_MAX_TOTAL_TOKENS, SGLANG_QUANT

EXAMPLES:
  $(basename "$0")                                              # Defaults (qwen2.5-0.6b, TP=1)
  $(basename "$0") -tp 1 models/hf/llama3.1-8b                 # Single GPU, 8B
  $(basename "$0") -p 8002 -tp 6 -q awq                        # AWQ quantization
  $(basename "$0") --disable-radix-cache                        # Baseline (no cache)
EOF
    exit 0
}

MODEL=""
PORT="${SGLANG_PORT:-8002}"
HOST="${SGLANG_HOST:-0.0.0.0}"
TP="${SGLANG_TP:-1}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.87}"
MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-1048576}"
CHUNKED_PS="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
ATTN_BACKEND="${SGLANG_ATTN_BACKEND:-flashinfer}"
QUANT="${SGLANG_QUANT:-none}"
MAX_RUNNING="${SGLANG_MAX_RUNNING:-256}"
TORCH_COMPILE="--enable-torch-compile"
RADIX_CACHE=""
TRUST_REMOTE="--trust-remote-code"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)              PORT="$2"; shift 2 ;;
        -H|--host)              HOST="$2"; shift 2 ;;
        -tp|--tp)               TP="$2"; shift 2 ;;
        -mfs|--mem-frac)        MEM_FRAC="$2"; shift 2 ;;
        -mtt|--max-total-tok)   MAX_TOTAL_TOKENS="$2"; shift 2 ;;
        -cps|--chunked-ps)      CHUNKED_PS="$2"; shift 2 ;;
        -attn|--attn-backend)   ATTN_BACKEND="$2"; shift 2 ;;
        -q|--quant)             QUANT="$2"; shift 2 ;;
        -mrr|--max-running)     MAX_RUNNING="$2"; shift 2 ;;
        --enable-torch-compile) TORCH_COMPILE="--enable-torch-compile"; shift ;;
        --no-torch-compile)     TORCH_COMPILE=""; shift ;;
        --disable-radix-cache)  RADIX_CACHE="--disable-radix-cache"; shift ;;
        --trust-remote-code)    TRUST_REMOTE="--trust-remote-code"; shift ;;
        -h|--help)              usage ;;
        -*)                     echo "Unknown option: $1" >&2; usage ;;
        *)                      MODEL="$1"; shift ;;
    esac
done

MODEL="${MODEL:-$DEFAULT_MODEL}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model specified. Set VLLM_MODEL env var or pass model path." >&2
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
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

row() { printf "  ${CYAN}%-16s${NC} %s\n" "$1" "$2"; }
sep() { echo -e "  ${DIM}$(printf '%.0s─' {1..48})${NC}"; }

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  SGLang${NC}${DIM} — RadixAttention Serving${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Model"          "$MODEL"
row "Endpoint"       "http://${HOST}:${PORT}"
sep
row "TP Size"        "${TP} GPUs"
row "Mem Fraction"   "${MEM_FRAC}"
row "Max Total Tok"  "${MAX_TOTAL_TOKENS}"
row "Chunked PS"     "${CHUNKED_PS}"
row "Attn Backend"   "${ATTN_BACKEND}"
row "Quantization"   "${QUANT:-none}"
row "Max Running"    "${MAX_RUNNING}"
row "Torch Compile"  "$( [[ -n "$TORCH_COMPILE" ]] && echo "ON" || echo "OFF" )"
row "Radix Cache"    "$( [[ -z "$RADIX_CACHE" ]] && echo "ON (default)" || echo "OFF" )"
sep
row "GPU"            "${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

exec python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRAC" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --chunked-prefill-size "$CHUNKED_PS" \
    --attention-backend "$ATTN_BACKEND" \
    ${QUANT:+--quantization "$QUANT"} \
    --max-running-requests "$MAX_RUNNING" \
    $TORCH_COMPILE \
    $RADIX_CACHE \
    $TRUST_REMOTE
