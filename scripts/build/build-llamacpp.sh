#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LLAMA_DIR="$PROJECT_ROOT/third_party/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

CUDA_ARCH="${CUDA_ARCHITECTURES:-86}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NPROC="${NPROC:-$(nproc)}"

echo "=== Building llama.cpp ==="
echo "  Source:  $LLAMA_DIR"
echo "  Build:   $BUILD_DIR"
echo "  CUDA arch: $CUDA_ARCH"
echo "  Jobs:    $NPROC"

cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA_GRAPHS=ON

cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$NPROC"

echo ""
echo "=== Build complete ==="
echo "  Binaries: $BUILD_DIR/bin/"
ls -la "$BUILD_DIR/bin/" 2>/dev/null || true
