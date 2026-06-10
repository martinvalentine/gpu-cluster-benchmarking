#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SGLANG_DIR="$PROJECT_ROOT/third_party/sglang/python"

MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Building SGLang from source ==="
echo "  Source:  $SGLANG_DIR"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  Branch:  $(git -C "$(dirname "$SGLANG_DIR")" branch --show-current 2>/dev/null || echo '?')"
echo "  Commit:  $(git -C "$(dirname "$SGLANG_DIR")" log --oneline -1 2>/dev/null || echo '?')"

cd "$SGLANG_DIR"
uv pip install -e . --no-build-isolation

echo ""
echo "=== Build complete ==="
python3 -c "import sglang; print(f'  SGLang version: {sglang.__version__}')"
