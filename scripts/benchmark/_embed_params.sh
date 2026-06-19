#!/usr/bin/env bash
# scripts/benchmark/_embed_params.sh
#
# Shared embed-params helper for all *_bench.sh scripts.
# Sources by *_bench.sh scripts. Expects $PARAMS_SNAPSHOT to be set.
#
# Functions:
#   embed_params_in_sweep <sweep_json_path>
#     - Embeds params from PARAMS_SNAPSHOT into the sweep JSON at the given path.
#     - Silent if sweep file or PARAMS_SNAPSHOT don't exist (graceful degradation).
#     - Uses jq to merge: original_json * {params: params_json}
#     - All assumptions documented inline so any dev can verify the logic.

embed_params_in_sweep() {
    local sweep_file="$1"

    # Graceful degradation: missing files are not errors (bench script may run without params)
    [[ -f "$sweep_file" ]] || return 0
    [[ -f "$PARAMS_SNAPSHOT" ]] || return 0

    # jq is a hard requirement for the merge. If absent, warn once.
    command -v jq &>/dev/null || {
        echo "  [params] WARN: jq not found; skipping params embed for $sweep_file" >&2
        return 0
    }

    # Merge: original * {params: snapshot} — overwrites original with merged
    local tmp; tmp=$(mktemp)
    if jq -s '.[0] * {params: .[1]}' "$sweep_file" "$PARAMS_SNAPSHOT" > "$tmp" 2>/dev/null; then
        mv "$tmp" "$sweep_file"
        echo "  [params] embedded in $(basename "$sweep_file")"
    else
        rm -f "$tmp"
        echo "  [params] WARN: jq merge failed for $sweep_file" >&2
    fi
}
