#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

DEFAULT_MODEL="/workspace/models/hf/qwen2.5-32b-awq"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODEL_PATH]

Start vLLM API server with prefix caching & chunked prefill (benchmark_plan.md B.1).

POSITIONAL:
  MODEL_PATH              HF model path or local dir (default: $DEFAULT_MODEL)

OPTIONS:
  -p, --port PORT         Server port (default: 8000)
  -H, --host HOST         Bind host (default: 0.0.0.0)
  -tp, --tp N             Tensor parallel size (default: 6)
  -gmu, --gpu-mem-util F  GPU memory utilization (default: 0.87)
  -mml, --max-model-len N Max model context length (default: 4096)
  -mns, --max-num-seqs N  Max concurrent sequences (default: 256)
  -q, --quant TYPE        Quantization: awq, gptq, none (default: awq)
  -pc, --prefix-cache     Enable prefix caching (default)
  -no-pc, --no-prefix-cache  Disable prefix caching
  -cp, --chunked-prefill  Enable chunked prefill (default)
  -no-cp, --no-chunked-prefill  Disable chunked prefill
  -mbt, --max-batched-tokens N  Max batched tokens for chunked prefill (default: 8192)
  -sw, --swap N           Swap space in GB (default: 4)
  -mp, --metrics-port P   Prometheus metrics port (default: 9090)
  --trust-remote-code     Trust remote code (default)
  -h, --help              Show this help

ENV OVERRIDES:
  VLLM_PORT, VLLM_HOST, VLLM_TP, VLLM_GPU_MEM_UTIL,
  VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_SEQS, VLLM_QUANT

EXAMPLES:
  $(basename "$0")                                              # Defaults (32B AWQ, TP=6)
  $(basename "$0") -tp 1 -gmu 0.90 /workspace/models/hf/llama3.1-8b   # Single GPU, 8B
  $(basename "$0") -p 8000 -tp 6 -q awq                        # AWQ quantization
  $(basename "$0") --no-prefix-cache --no-chunked-prefill       # Baseline (no cache)
EOF
    exit 0
}

MODEL=""
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
TP="${VLLM_TP:-6}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.87}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
QUANT="${VLLM_QUANT:-awq}"
PREFIX_CACHE="--enable-prefix-caching"
CHUNKED_PREFILL="--enable-chunked-prefill"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-8192}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-4}"
METRICS_PORT="${VLLM_METRICS_PORT:-9090}"
TRUST_REMOTE="--trust-remote-code"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)              PORT="$2"; shift 2 ;;
        -H|--host)              HOST="$2"; shift 2 ;;
        -tp|--tp)               TP="$2"; shift 2 ;;
        -gmu|--gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
        -mml|--max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        -mns|--max-num-seqs)    MAX_NUM_SEQS="$2"; shift 2 ;;
        -q|--quant)             QUANT="$2"; shift 2 ;;
        -pc|--prefix-cache)     PREFIX_CACHE="--enable-prefix-caching"; shift ;;
        -no-pc|--no-prefix-cache) PREFIX_CACHE=""; shift ;;
        -cp|--chunked-prefill)  CHUNKED_PREFILL="--enable-chunked-prefill"; shift ;;
        -no-cp|--no-chunked-prefill) CHUNKED_PREFILL=""; shift ;;
        -mbt|--max-batched-tokens) MAX_BATCHED_TOKENS="$2"; shift 2 ;;
        -sw|--swap)             SWAP_SPACE="$2"; shift 2 ;;
        -mp|--metrics-port)     METRICS_PORT="$2"; shift 2 ;;
        --trust-remote-code)    TRUST_REMOTE="--trust-remote-code"; shift ;;
        -h|--help)              usage ;;
        -*)                     echo "Unknown option: $1" >&2; usage ;;
        *)                      MODEL="$1"; shift ;;
    esac
done

MODEL="${MODEL:-$DEFAULT_MODEL}"

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
echo -e "  ${GREEN}${BOLD}  vLLM${NC}${DIM} — High-throughput LLM Serving${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
row "Model"          "$MODEL"
row "Endpoint"       "http://${HOST}:${PORT}"
row "Metrics"        "http://${HOST}:${METRICS_PORT}/metrics"
sep
row "TP Size"        "${TP} GPUs"
row "GPU Mem"        "${GPU_MEM_UTIL}"
row "Max Seqs"       "${MAX_NUM_SEQS}"
row "Max Context"    "${MAX_MODEL_LEN}"
row "Quantization"   "${QUANT:-none}"
row "Prefix Cache"   "$( [[ -n "$PREFIX_CACHE" ]] && echo "ON" || echo "OFF" )"
row "Chunked Prefill" "$( [[ -n "$CHUNKED_PREFILL" ]] && echo "ON (${MAX_BATCHED_TOKENS} tok)" || echo "OFF" )"
row "Swap"           "${SWAP_SPACE} GB"
sep
row "GPU"            "${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    ${QUANT:+--quantization "$QUANT"} \
    $PREFIX_CACHE \
    $CHUNKED_PREFILL \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --enable-metrics \
    --metrics-port "$METRICS_PORT" \
    --swap-space "$SWAP_SPACE" \
    $TRUST_REMOTE
