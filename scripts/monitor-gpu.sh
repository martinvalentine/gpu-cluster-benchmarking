#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${GPU_MONITOR_DIR:-${PROJECT_ROOT}/results}"
INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
DURATION="${GPU_MONITOR_DURATION:-0}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Monitor GPU utilization and memory during benchmarks (benchmark_plan.md G.2).
Writes CSV logs and optionally displays real-time stats.

OPTIONS:
  -o, --output DIR        Output directory (default: ./results)
  -i, --interval SEC      Sampling interval in seconds (default: 5)
  -d, --duration SEC      Duration in seconds (0=unlimited, default: 0)
  -w, --watch             Display live stats with watch (requires tmux or separate terminal)
  -h, --help              Show this help

ENV OVERRIDES:
  GPU_MONITOR_DIR, GPU_MONITOR_INTERVAL, GPU_MONITOR_DURATION

EXAMPLES:
  $(basename "$0")                                    # Log every 5s indefinitely
  $(basename "$0") -i 2 -d 300                        # Every 2s for 5 minutes
  $(basename "$0") -w                                 # Live watch mode
  $(basename "$0") -o /workspace/results/gpu          # Custom output dir

OUTPUT FILES:
  gpu_monitor_<timestamp>.csv   CSV with timestamp, index, name, memory_used, memory_total, util, temp
EOF
    exit 0
}

WATCH_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)    OUTPUT_DIR="$2"; shift 2 ;;
        -i|--interval)  INTERVAL="$2"; shift 2 ;;
        -d|--duration)  DURATION="$2"; shift 2 ;;
        -w|--watch)     WATCH_MODE=1; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown: $1" >&2; usage ;;
    esac
done

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. No NVIDIA GPU available." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${OUTPUT_DIR}/gpu_monitor_${TIMESTAMP}.csv"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  GPU Monitor${NC}${DIM} — Real-time Metrics${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Output${NC}     $CSV_FILE"
echo -e "  ${CYAN}Interval${NC}   ${INTERVAL}s"
echo -e "  ${CYAN}Duration${NC}   $( [[ "$DURATION" -eq 0 ]] && echo "unlimited (Ctrl+C to stop)" || echo "${DURATION}s" )"
echo ""

# Write CSV header
echo "timestamp,gpu_index,gpu_name,memory_used_mb,memory_total_mb,memory_pct,utilization_pct,temperature_c" > "$CSV_FILE"

if [[ "$WATCH_MODE" -eq 1 ]]; then
    echo "Starting watch mode (Ctrl+C to stop)..."
    echo ""
    watch -n "$INTERVAL" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader'
    exit 0
fi

echo "Logging to $CSV_FILE (Ctrl+C to stop)..."
echo ""

START_TIME=$(date +%s)

while true; do
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name mem_used mem_total util temp; do
        # Trim whitespace
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        util=$(echo "$util" | xargs)
        temp=$(echo "$temp" | xargs)

        mem_pct=0
        if [[ "$mem_total" -gt 0 ]]; then
            mem_pct=$(( mem_used * 100 / mem_total ))
        fi

        echo "${NOW},${idx},${name},${mem_used},${mem_total},${mem_pct},${util},${temp}" >> "$CSV_FILE"
    done

    # Print summary line
    MEM_SUMMARY=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{used+=$1; total+=$2} END {printf "%d/%d MB (%d%%)", used, total, (total>0?used*100/total:0)}')
    UTIL_SUMMARY=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk '{sum+=$1; n++} END {printf "%.0f%%", sum/n}')

    echo -e "  ${DIM}$(date +%H:%M:%S)${NC}  Mem: ${CYAN}${MEM_SUMMARY}${NC}  Util: ${CYAN}${UTIL_SUMMARY}${NC}"

    # Check duration limit
    if [[ "$DURATION" -gt 0 ]]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [[ "$ELAPSED" -ge "$DURATION" ]]; then
            echo ""
            echo "Duration limit reached (${DURATION}s). Stopping."
            break
        fi
    fi

    sleep "$INTERVAL"
done

echo ""
echo "Results saved to: $CSV_FILE"
echo "Total samples: $(( $(wc -l < "$CSV_FILE") - 1 ))"
