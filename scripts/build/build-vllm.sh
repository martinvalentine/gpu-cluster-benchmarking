#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VLLM_DIR="$PROJECT_ROOT/third_party/vllm"

MAX_JOBS="${MAX_JOBS:-4}"
VERBOSE=0

for arg in "$@"; do
  case "$arg" in
    -v|--verbose) VERBOSE=1 ;;
  esac
done

UV_FLAGS=()
[[ "$VERBOSE" -eq 1 ]] && UV_FLAGS=(-v)

echo "=== Building vLLM from source ==="
echo "  Source:  $VLLM_DIR"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  Branch:  $(git -C "$VLLM_DIR" branch --show-current 2>/dev/null || echo '?')"
echo "  Commit:  $(git -C "$VLLM_DIR" log --oneline -1 2>/dev/null || echo '?')"

cd "$VLLM_DIR"
uv pip install "${UV_FLAGS[@]}" setuptools setuptools_rust setuptools_scm torch
MAX_JOBS="$MAX_JOBS" uv pip install "${UV_FLAGS[@]}" -e . --no-build-isolation

echo ""
echo "=== Build complete ==="
python3 -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"
