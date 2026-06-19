#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") -b BACKEND [OPTIONS]

Unified benchmark dispatcher — routes to per-backend scripts.

OPTIONS:
  -b, --backend BACKEND   Backend: vllm, sglang, llamacpp (required)
  -h, --help              Show this help

All other flags are forwarded to the per-backend script.
EOF
    exit 0
}

BACKEND=""

# Extract -b/--backend before forwarding
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--backend)   BACKEND="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$BACKEND" ]]; then
    echo "ERROR: --backend is required (vllm, sglang, llamacpp)" >&2
    usage
fi

case "$BACKEND" in
    vllm)
        exec bash "$SCRIPT_DIR/vllm_bench.sh" "${ARGS[@]}"
        ;;
    sglang)
        exec bash "$SCRIPT_DIR/sglang_bench.sh" "${ARGS[@]}"
        ;;
    llamacpp)
        exec bash "$SCRIPT_DIR/llamacpp_bench.sh" "${ARGS[@]}"
        ;;
    *)
        echo "ERROR: Unknown backend '$BACKEND'. Supported: vllm, sglang, llamacpp" >&2
        exit 1
        ;;
esac
