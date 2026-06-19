#!/usr/bin/env bash
#
# find-best-np.sh
#
# Sweeps llama-server's -np (--parallel slot count) to find the point of
# diminishing returns for a given model/GPU/prompt-size combo.
#
# Why this exists:
#   -np sets a *ceiling* on concurrent KV-cache slots, hard-capped at 256 by
#   llama.cpp itself (n_seq_max <= 256), regardless of available VRAM.
#   The *optimal* value is almost always lower than the ceiling, because
#   past some point CPU-side scheduling overhead and compute saturation
#   eat into aggregate throughput. This can only be found empirically.
#
# Method:
#   For each candidate -np value, launch the server with that many slots,
#   fire exactly that many concurrent requests at it, measure aggregate
#   throughput (tok/s) and p50/p99 latency, then tear the server down.
#   Stop early once throughput gain between consecutive rungs falls below
#   a threshold (the "knee" of the curve) for N consecutive rungs.
#
# Usage:
#   ./find-best-np.sh [options]
#
# Requires: docker, curl, python3 (no jq dependency)

set -uo pipefail  # NOTE: not -e -- we need to handle docker/curl failures per-rung, not abort the sweep

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
IMAGE="registry.meetclawlab.com/harmony-bench-vllm-sglang-llama:cu129-v1.4"
MODEL_PATH="/workspace/models/gguf/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODELS_DIR="$(pwd)/models"
PORT=8001
CONTAINER_NAME="harmony-bench-llama-npsweep"
CONTEXT_LEN=16384
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="turbo4"
FLASH_ATTN="on"              # --fa value: on, off, or auto
PROMPT_TOKENS=512           # approximate prompt length used for synthetic load
GEN_TOKENS=128               # tokens to generate per request during the probe
NP_CANDIDATES=(8 16 32 48 64 96 128 192 256)
NP_HARD_CAP=256               # llama.cpp's n_seq_max limit -- do not raise this
KNEE_THRESHOLD_PCT=5          # stop if throughput gain < this % for KNEE_PATIENCE rungs in a row
KNEE_PATIENCE=2
FULL_SWEEP=false               # if true, run all np candidates regardless of knee detection
AUTO_CTX=false                 # if true, calculate -c per rung as np * (prompt + gen)
MODEL_CTX=32768                # model's training context, used as ceiling for auto-ctx
SERVER_STARTUP_TIMEOUT=120    # seconds to wait for /health before declaring boot failure
REQUEST_TIMEOUT=120           # seconds per individual request before considered failed
RESULTS_DIR="$(pwd)/np-sweep-results-$(date +%Y%m%d-%H%M%S)"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [options]

Sweep llama-server -np slot counts to find the scheduling throughput knee.
Each rung starts a server with that -np, fires np concurrent requests,
measures aggregate tok/s, then stops and advances to the next -np.

Important: per-slot context = --ctx / --np. Your --prompt-tokens must fit
inside per-slot context, or requests will 400 with "exceeds context size".
Use --auto-ctx to calculate -c per rung automatically.

  --image IMAGE              Docker image (default: $IMAGE)
  --model PATH                GGUF model path inside container (default: $MODEL_PATH)
  --models-dir DIR            Host directory mounted to /workspace/models (default: ./models)
  --port PORT                 Server port (default: $PORT)
  --ctx N                     Total context (-c). Per-slot = ctx / np (default: $CONTEXT_LEN)
  --ctk TYPE                  --cache-type-k (default: $CACHE_TYPE_K)
  --ctv TYPE                  --cache-type-v (default: $CACHE_TYPE_V)
  --fa on|off|auto            Flash Attention (default: $FLASH_ATTN)
  --np-list "8 16 32 ..."     Space-separated -np values to test (max: $NP_HARD_CAP, higher skipped)
  --knee-threshold PCT        Min throughput gain % between rungs (default: $KNEE_THRESHOLD_PCT%)
  --knee-patience N           Consecutive below-threshold rungs before stopping (default: $KNEE_PATIENCE)
  --prompt-tokens N            Synthetic prompt length in tokens (default: $PROMPT_TOKENS)
  --gen-tokens N              Tokens to generate per request (default: $GEN_TOKENS)
  --full-sweep                Run all np candidates, disable knee-detection early exit
  --auto-ctx                  Auto-calc -c per rung: np * (prompt_tokens + gen_tokens)
                              (overrides --ctx, capped at np * model_ctx)
  --model-ctx N               Model's training context, ceiling for --auto-ctx (default: $MODEL_CTX)
  --request-timeout N          Max seconds per curl request (default: $REQUEST_TIMEOUT)
  -h, --help                  Show this help

Examples:
  # Default sweep (np=8..256, ctx=16384)
  $0
  # Fast throughput-knee sweep (small prompts, dense np ladder)
  $0 --np-list "16 32 48 64 96 128" --prompt-tokens 64 --gen-tokens 64
  # Custom context and prompt length
  $0 --np-list "16 32 64" --ctx 8192 --prompt-tokens 128 --gen-tokens 256
  # Large prompt/gen sweep (increase request-timeout)
  $0 --np-list "16 32 48 64" --auto-ctx --request-timeout 600 \
     --prompt-tokens 2000 --gen-tokens 1000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx) CONTEXT_LEN="$2"; shift 2 ;;
    --ctk) CACHE_TYPE_K="$2"; shift 2 ;;
    --ctv) CACHE_TYPE_V="$2"; shift 2 ;;
    --fa) FLASH_ATTN="$2"; shift 2 ;;
    --np-list) read -r -a NP_CANDIDATES <<< "$2"; shift 2 ;;
    --knee-threshold) KNEE_THRESHOLD_PCT="$2"; shift 2 ;;
    --knee-patience) KNEE_PATIENCE="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --full-sweep) FULL_SWEEP=true; shift ;;
    --auto-ctx) AUTO_CTX=true; shift ;;
    --model-ctx) MODEL_CTX="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
echo "np,requests_sent,requests_ok,requests_failed,wall_time_s,aggregate_tok_s,p50_latency_s,p99_latency_s,throughput_gain_pct" > "$SUMMARY_CSV"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Filter -np candidates against llama.cpp's hard cap (n_seq_max <= 256)
# Values > NP_HARD_CAP are skipped with a warning, not aborted.
# ---------------------------------------------------------------------------
filtered_candidates=()
for np in "${NP_CANDIDATES[@]}"; do
  if (( np > NP_HARD_CAP )); then
    log "WARNING: skipping $np (exceeds llama.cpp's hard cap of $NP_HARD_CAP)"
  else
    filtered_candidates+=("$np")
  fi
done
NP_CANDIDATES=("${filtered_candidates[@]}")
if [[ ${#NP_CANDIDATES[@]} -eq 0 ]]; then
  log "ERROR: all --np-list values exceed the cap of $NP_HARD_CAP. Nothing to do."
  exit 1
fi

# ---------------------------------------------------------------------------
# Cleanup handler -- always remove the container, even on Ctrl+C
# ---------------------------------------------------------------------------
cleanup_container() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT INT TERM

# ---------------------------------------------------------------------------
# Start server with a given -np, wait for health, return 0/1
# ---------------------------------------------------------------------------
start_server() {
  local np="$1"
  local logfile="$2"

  cleanup_container

  local ctx="${rung_ctx:-$CONTEXT_LEN}"

  docker run -d --rm --gpus all --ipc=host --network host \
    -v "${MODELS_DIR}:/workspace/models" \
    --name "$CONTAINER_NAME" \
    "$IMAGE" \
    llama-server \
      -m "$MODEL_PATH" \
      --port "$PORT" --host 0.0.0.0 \
      -ngl all -c "$ctx" -np "$np" -fa "$FLASH_ATTN" \
      -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V" --cache-prompt \
      > /dev/null 2>&1

  docker logs -f "$CONTAINER_NAME" >> "$logfile" 2>&1 &
  local log_tail_pid=$!

  local waited=0
  while (( waited < SERVER_STARTUP_TIMEOUT )); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "200"; then
      kill "$log_tail_pid" 2>/dev/null || true
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      kill "$log_tail_pid" 2>/dev/null || true
      # Final log flush
      docker logs "$CONTAINER_NAME" >> "$logfile" 2>&1 || true
      log "  Container exited early (likely startup failure -- check $logfile)"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done

  kill "$log_tail_pid" 2>/dev/null || true
  log "  Server did not become healthy within ${SERVER_STARTUP_TIMEOUT}s"
  return 1
}

stop_server() {
  cleanup_container
  sleep 1
}

# ---------------------------------------------------------------------------
# Fire `np` concurrent requests, each generating GEN_TOKENS tokens off a
# synthetic prompt of ~PROMPT_TOKENS tokens, via the /completion endpoint.
# Returns aggregate stats by writing per-request timing files, then
# aggregating with python3.
# ---------------------------------------------------------------------------
run_load() {
  local np="$1"
  local req_dir="$RESULTS_DIR/np_${np}_requests"
  mkdir -p "$req_dir"

  # Build a synthetic prompt of roughly PROMPT_TOKENS tokens.
  # Repeating a short phrase is a crude but adequate way to hit a target
  # token count for load-testing purposes (exact tokenization doesn't matter --
  # what matters is consistency across rungs of the sweep).
  local prompt
  prompt=$(python3 -c "print(('Explain how distributed systems handle consensus. ' * $((PROMPT_TOKENS / 8 + 1))))")

  local start_ts
  start_ts=$(date +%s.%N)

  local pids=()
  for i in $(seq 1 "$np"); do
    (
      local req_start req_end
      req_start=$(date +%s.%N)
      http_code=$(curl -s -o "$req_dir/resp_${i}.json" -w "%{http_code}" \
        --max-time "$REQUEST_TIMEOUT" \
        -X POST "http://127.0.0.1:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'prompt': '''${prompt}''',
    'n_predict': ${GEN_TOKENS},
    'cache_prompt': True,
    'stream': False
}))
")" 2>/dev/null)
      req_end=$(date +%s.%N)
      echo "${http_code} ${req_start} ${req_end}" > "$req_dir/timing_${i}.txt"
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  local end_ts
  end_ts=$(date +%s.%N)

  python3 - "$req_dir" "$np" "$GEN_TOKENS" "$start_ts" "$end_ts" <<'PYEOF'
import json, sys, glob, os

req_dir, np, gen_tokens, start_ts, end_ts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

ok = 0
failed = 0
latencies = []

for i in range(1, np + 1):
    timing_path = os.path.join(req_dir, f"timing_{i}.txt")
    resp_path = os.path.join(req_dir, f"resp_{i}.json")
    if not os.path.exists(timing_path):
        failed += 1
        continue
    with open(timing_path) as f:
        parts = f.read().split()
    if len(parts) != 3:
        failed += 1
        continue
    http_code, req_start, req_end = parts[0], float(parts[1]), float(parts[2])
    if http_code != "200":
        failed += 1
        continue
    # Sanity check the response actually contains generated content
    try:
        with open(resp_path) as f:
            body = json.load(f)
        if "content" not in body and "choices" not in body:
            failed += 1
            continue
    except Exception:
        failed += 1
        continue
    ok += 1
    latencies.append(req_end - req_start)

wall_time = end_ts - start_ts
latencies.sort()

def pct(p):
    if not latencies:
        return 0.0
    idx = min(len(latencies) - 1, int(len(latencies) * p))
    return latencies[idx]

agg_tokens = ok * gen_tokens
agg_tok_s = agg_tokens / wall_time if wall_time > 0 else 0.0
p50 = pct(0.50)
p99 = pct(0.99)

result = {
    "requests_sent": np,
    "requests_ok": ok,
    "requests_failed": failed,
    "wall_time_s": round(wall_time, 3),
    "aggregate_tok_s": round(agg_tok_s, 2),
    "p50_latency_s": round(p50, 3),
    "p99_latency_s": round(p99, 3),
}
print(json.dumps(result))
PYEOF
}

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
log "Starting -np sweep"
log "  Image:        $IMAGE"
log "  Model:        $MODEL_PATH"
if $AUTO_CTX; then
  log "  Context (-c): auto (per-rung: np*(prompt+gen), max: np*$MODEL_CTX)"
else
  log "  Context (-c): $CONTEXT_LEN"
fi
log "  Candidates:   ${NP_CANDIDATES[*]}"
log "  Results dir:  $RESULTS_DIR"
echo

prev_tok_s=0
below_threshold_streak=0
best_np=0
best_tok_s=0

for np in "${NP_CANDIDATES[@]}"; do
  log "=== -np $np ==="
  server_log="$RESULTS_DIR/server_np_${np}.log"

  if $AUTO_CTX; then
    rung_ctx=$(( np * (PROMPT_TOKENS + GEN_TOKENS) ))
    max_useful=$(( np * MODEL_CTX ))
    if (( rung_ctx > max_useful )); then
      rung_ctx=$max_useful
    fi
    log "  auto-ctx: -c $rung_ctx (per-slot: $(( rung_ctx / np )))"
  else
    rung_ctx=$CONTEXT_LEN
  fi

  if ! start_server "$np" "$server_log"; then
    log "  SKIP: server failed to start at -np $np (see $server_log)"
    echo "$np,0,0,0,0,0,0,0,FAILED_TO_START" >> "$SUMMARY_CSV"
    stop_server
    # If even a low -np fails to start, something else is wrong (bad ctv, OOM, etc) --
    # no point continuing the sweep upward.
    log "  Aborting sweep: a failure at this -np will persist for all higher values."
    break
  fi

  log "  Server healthy. Firing $np concurrent requests (~${PROMPT_TOKENS} prompt tokens, ${GEN_TOKENS} gen tokens each)..."
  result_json=$(run_load "$np")
  stop_server

  ok=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['requests_ok'])" "$result_json")
  failed=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['requests_failed'])" "$result_json")
  wall_time=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['wall_time_s'])" "$result_json")
  tok_s=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['aggregate_tok_s'])" "$result_json")
  p50=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['p50_latency_s'])" "$result_json")
  p99=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['p99_latency_s'])" "$result_json")

  gain_pct="n/a"
  if (( $(echo "$prev_tok_s > 0" | bc -l) )); then
    gain_pct=$(echo "scale=2; ($tok_s - $prev_tok_s) / $prev_tok_s * 100" | bc -l)
  fi

  log "  ok=$ok failed=$failed wall=${wall_time}s aggregate=${tok_s} tok/s p50=${p50}s p99=${p99}s gain=${gain_pct}%"
  echo "$np,$np,$ok,$failed,$wall_time,$tok_s,$p50,$p99,$gain_pct" >> "$SUMMARY_CSV"

  if (( $(echo "$tok_s > $best_tok_s" | bc -l) )); then
    best_tok_s="$tok_s"
    best_np="$np"
  fi

  if [[ "$failed" -gt 0 ]]; then
    log "  WARNING: $failed/$np requests failed at this concurrency -- treat results at and above this -np with caution."
  fi

  if [[ "$gain_pct" != "n/a" ]]; then
    if (( $(echo "$gain_pct < $KNEE_THRESHOLD_PCT" | bc -l) )); then
      below_threshold_streak=$((below_threshold_streak + 1))
      log "  Throughput gain below ${KNEE_THRESHOLD_PCT}% threshold (streak: ${below_threshold_streak}/${KNEE_PATIENCE})"
      if (( below_threshold_streak >= KNEE_PATIENCE )); then
        log "  Knee detected -- diminishing returns confirmed for ${KNEE_PATIENCE} consecutive rungs."
        if ! $FULL_SWEEP; then
          log "  Stopping sweep early (use --full-sweep to run all candidates)."
          break
        fi
      fi
    else
      below_threshold_streak=0
    fi
  fi

  prev_tok_s="$tok_s"
  echo
done

echo
log "=== Sweep complete ==="
log "Best aggregate throughput: ${best_tok_s} tok/s at -np ${best_np}"
log "Full results: $SUMMARY_CSV"
log ""
log "NOTE: 'best_np' above is the highest-throughput rung tested, not necessarily"
log "      the most *efficient* one. Check $SUMMARY_CSV for the gain_pct column --"
log "      the optimal -np for production is usually the knee (first rung where"
log "      gain_pct drops below ${KNEE_THRESHOLD_PCT}%), not the peak, since rungs"
log "      past the knee buy little throughput at the cost of worse p99 latency"
log "      and wasted VRAM reservation."