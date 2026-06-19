#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER="harmony-bench"
INSIDE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- BENCH_ARGS...]

Delegate benchmarks to a running Docker container.

OPTIONS:
  --container NAME    Container name (default: harmony-bench)
  --inside            Skip docker exec — run directly (for use inside container)
  -h, --help          Show this help

BENCH_ARGS are forwarded to the benchmark script inside the container.

EXAMPLES:
  $(basename "$0") -- -b llamacpp
  $(basename "$0") -- -b vllm --ccu-mode mul --ccu-max 256
  $(basename "$0") -- -b sglang --full
  $(basename "$0") --container my-bench -- -b llamacpp
EOF
    exit 0
}

BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)    CONTAINER="$2"; shift 2 ;;
        --inside)       INSIDE=1; shift ;;
        -h|--help)      usage ;;
        --)             shift; BENCH_ARGS=("$@"); break ;;
        *)              BENCH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#BENCH_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No benchmark arguments provided. Use -- to separate." >&2
    usage
fi

if [[ "$INSIDE" -eq 1 ]]; then
    exec bash "$SCRIPT_DIR/benchmark/bench.sh" "${BENCH_ARGS[@]}"
fi

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    echo "ERROR: Container '$CONTAINER' is not running." >&2
    echo "" >&2
    echo "Start it with:" >&2
    echo "  docker run -d --gpus all --ipc=host --network host \\" >&2
    echo "    -v \$(pwd)/models:/workspace/models \\" >&2
    echo "    -v \$(pwd)/scripts:/workspace/scripts \\" >&2
    echo "    -v \$(pwd)/configs:/workspace/configs \\" >&2
    echo "    -v \$(pwd)/results:/workspace/results \\" >&2
    echo "    -v \$(pwd)/datasets:/workspace/datasets \\" >&2
    echo "    --name $CONTAINER registry.meetclawlab.com/harmony-bench-vllm-sglang-llama:cu129-v1.4 sleep infinity" >&2
    exit 1
fi

ESCAPED_ARGS=$(printf '%q ' "${BENCH_ARGS[@]}")

exec docker exec -it "$CONTAINER" \
    bash -c "cd /workspace && ./scripts/benchmark/bench.sh $ESCAPED_ARGS"
