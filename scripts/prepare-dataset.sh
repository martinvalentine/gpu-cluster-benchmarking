#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download and prepare benchmark dataset (Vietnamese vi-alpaca → ShareGPT format).

Default: downloads to /workspace/datasets/sharegpt.json
Falls back to English ShareGPT if Vietnamese dataset fails.

OPTIONS:
  -d, --dir DIR         Destination directory (default: /workspace/datasets)
  -o, --output FILE     Output filename (default: sharegpt.json)
  --skip-existing       Skip if output file already exists
  --dry-run             Show what would happen without downloading
  -h, --help            Show this help

EXAMPLES:
  $(basename "$0")                                        # Default: /workspace/datasets/
  $(basename "$0") -d /data/benchmarks                   # Custom directory
  $(basename "$0") -o my_dataset.json                     # Custom filename
  $(basename "$0") --skip-existing                        # Skip if exists
EOF
    exit 0
}

DEST_DIR="/workspace/datasets"
OUTPUT="sharegpt.json"
SKIP_EXISTING=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)           DEST_DIR="$2"; shift 2 ;;
        -o|--output)        OUTPUT="$2"; shift 2 ;;
        --skip-existing)    SKIP_EXISTING=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown: $1" >&2; usage ;;
    esac
done

OUTPUT_PATH="${DEST_DIR}/${OUTPUT}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Benchmark Dataset Downloader${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Destination${NC}  $OUTPUT_PATH"
echo -e "  ${CYAN}Strategy${NC}     Vietnamese vi-alpaca → English ShareGPT fallback"
echo ""

# ── Check existing ──────────────────────────────────────────────────────
if [[ "$SKIP_EXISTING" -eq 1 && -f "$OUTPUT_PATH" ]]; then
    SIZE=$(du -sh "$OUTPUT_PATH" 2>/dev/null | cut -f1)
    ok "Dataset already exists: $OUTPUT_PATH ($SIZE)"
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo -e "  ${YELLOW}[DRY-RUN]${NC} Would download to: $OUTPUT_PATH"
    echo -e "  ${YELLOW}[DRY-RUN]${NC} Vietnamese vi-alpaca → English ShareGPT fallback"
    exit 0
fi

mkdir -p "$DEST_DIR"

# ── Try Vietnamese vi-alpaca first ─────────────────────────────────────
log "Attempting Vietnamese vi-alpaca dataset..."

python3 -c "
import os, json, sys

try:
    os.system('pip install datasets pyarrow -q')
    from datasets import load_dataset
    print('Loading vi-alpaca from Hugging Face...')
    ds = load_dataset('bkai-foundation-models/vi-alpaca', split='train')
    sharegpt = []
    for item in ds:
        prompt = item['instruction']
        if item.get('input'):
            prompt += '\n' + item['input']
        sharegpt.append({
            'conversations': [
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': item['output']}
            ]
        })
    with open('${OUTPUT_PATH}', 'w', encoding='utf-8') as f:
        json.dump(sharegpt, f, ensure_ascii=False, indent=2)
    print(f'Success: {len(sharegpt)} samples written to ${OUTPUT_PATH}')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1

if [[ $? -eq 0 ]] && [[ -f "$OUTPUT_PATH" ]]; then
    SIZE=$(du -sh "$OUTPUT_PATH" | cut -f1)
    COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_PATH'))))" 2>/dev/null || echo "?")
    ok "Vietnamese vi-alpaca downloaded: $COUNT samples ($SIZE)"
    exit 0
fi

warn "Vietnamese dataset failed, falling back to English ShareGPT..."

# ── Fallback: English ShareGPT ──────────────────────────────────────────
log "Downloading English ShareGPT..."

wget -q -O "$OUTPUT_PATH" \
    'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json' \
    2>&1 || {
        fail "Failed to download English ShareGPT"
        exit 1
    }

if [[ -f "$OUTPUT_PATH" ]]; then
    SIZE=$(du -sh "$OUTPUT_PATH" | cut -f1)
    COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_PATH'))))" 2>/dev/null || echo "?")
    ok "English ShareGPT downloaded: $COUNT samples ($SIZE)"
else
    fail "Dataset download failed"
    exit 1
fi
