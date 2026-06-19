#!/usr/bin/env bash
# scripts/run/write_params.sh
#
# Shared helper for all run-*.sh scripts. Writes a structured params.json
# file containing server config, hardware, and system info for the
# currently-running server.
#
# Usage: source this file from a run-*.sh script, then call:
#   write_params <backend> [out_path]
#
# All server config is read from environment variables. The python heredoc
# inside write_params does type coercion and writes the JSON.
#
# Required env vars (set by the run-*.sh script):
#   MODEL, MODEL_PATH, PORT, TP, MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_MEM_UTIL,
#   N_PARALLEL, CTX_SIZE, N_BATCH, N_UBATCH, N_THREADS, FLASH_ATTN, CACHE_KEY,
#   CACHE_VAL, CACHE_PROMPT, PREFIX_CACHE, CHUNKED_PREFILL, MAX_BATCHED_TOKENS,
#   SWAP_SPACE, QUANT, DTYPE, BLOCK_SIZE, ATTN_BACKEND, RADIX_CACHE,
#   TORCH_COMPILE, MAX_TOTAL_TOKENS, CHUNKED_PS, TRUST_REMOTE
#
# Optional env vars (with sensible defaults):
#   HARMONY_IMAGE / DOCKER_IMAGE — docker image tag
#   SERVER_VERSION — server version string

write_params() {
    local backend="$1"
    local out="${2:-${PROJECT_ROOT}/results/${backend}/_active_params.json}"

    mkdir -p "$(dirname "$out")"

    # Collect hardware
    local hostname; hostname=$(hostname 2>/dev/null || echo "unknown")
    local gpu_name; gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    local gpu_count; gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    local gpu_vram_mib; gpu_vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 0)
    local driver; driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    local cuda; cuda=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    local cpu; cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "N/A")
    local cores; cores=$(nproc 2>/dev/null || echo 0)
    local mem_gb; mem_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)

    # Collect system
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local commit; commit=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "N/A")
    local branch; branch=$(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo "N/A")
    local img; img="${HARMONY_IMAGE:-${DOCKER_IMAGE:-N/A}}"
    local ver; ver="${SERVER_VERSION:-N/A}"

    # Build JSON via python3 — keeps type coercion correct (int("4") not str("4"))
    python3 - "$backend" "$out" \
        MODEL="$MODEL" MODEL_PATH="$MODEL_PATH" PORT="$PORT" \
        TP="$TP" MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        GPU_MEM_UTIL="$GPU_MEM_UTIL" N_PARALLEL="$N_PARALLEL" CTX_SIZE="$CTX_SIZE" \
        N_BATCH="$N_BATCH" N_UBATCH="$N_UBATCH" N_THREADS="$N_THREADS" \
        FLASH_ATTN="$FLASH_ATTN" CACHE_KEY="$CACHE_KEY" CACHE_VAL="$CACHE_VAL" \
        CACHE_PROMPT="$CACHE_PROMPT" PREFIX_CACHE="$PREFIX_CACHE" \
        CHUNKED_PREFILL="$CHUNKED_PREFILL" MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS" \
        SWAP_SPACE="$SWAP_SPACE" QUANT="$QUANT" DTYPE="$DTYPE" BLOCK_SIZE="$BLOCK_SIZE" \
        ATTN_BACKEND="$ATTN_BACKEND" RADIX_CACHE="$RADIX_CACHE" TORCH_COMPILE="$TORCH_COMPILE" \
        MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" CHUNKED_PS="$CHUNKED_PS" TRUST_REMOTE="$TRUST_REMOTE" \
        HOSTNAME_VAL="$hostname" GPU_NAME_VAL="$gpu_name" GPU_COUNT_VAL="$gpu_count" \
        GPU_VRAM_MIB_VAL="$gpu_vram_mib" DRIVER_VAL="$driver" CUDA_VAL="$cuda" \
        CPU_NAME_VAL="$cpu" CPU_CORES_VAL="$cores" MEM_GB_VAL="$mem_gb" \
        TS_VAL="$ts" COMMIT_VAL="$commit" BRANCH_VAL="$branch" IMG_VAL="$img" VER_VAL="$ver" \
        <<'PYEOF'
import json, os, sys


def _int(v, default=None):
    if v is None or v == "" or v == "N/A":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    if v is None or v == "" or v == "N/A":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v, default=False):
    """Treat strings like '--enable-prefix-caching' or '1' as True; '' as default."""
    if v is None or v == "":
        return default
    s = str(v).lower()
    if s in ("1", "true", "on", "yes", "y"):
        return True
    if s in ("0", "false", "off", "no", "n"):
        return False
    # CLI-flag-style strings (e.g. "--enable-prefix-caching") → True
    if s.startswith("--enable") or s.startswith("--use"):
        return True
    if s.startswith("--disable") or s.startswith("--no-"):
        return False
    return default


def _str(v, default="N/A"):
    if v is None or v == "":
        return default
    return str(v)


_, out_path = sys.argv[1], sys.argv[2]

data = {
    "server": {
        "model":              _str(os.environ.get("MODEL")),
        "model_path":         _str(os.environ.get("MODEL_PATH")),
        "endpoint":           f"http://0.0.0.0:{_str(os.environ.get('PORT'), '8000')}",
        "port":               _int(os.environ.get("PORT"), 0),
        "tp_size":            _int(os.environ.get("TP"), 1),
        "max_model_len":      _int(os.environ.get("MAX_MODEL_LEN"), 0),
        "max_num_seqs":       _int(os.environ.get("MAX_NUM_SEQS"), 0),
        "gpu_mem_util":       _float(os.environ.get("GPU_MEM_UTIL"), 0.0),
        "n_parallel":         _int(os.environ.get("N_PARALLEL"), 0),
        "ctx_size":           _int(os.environ.get("CTX_SIZE"), 0),
        "batch":              _int(os.environ.get("N_BATCH"), 0),
        "ubatch":             _int(os.environ.get("N_UBATCH"), 0),
        "threads":            _int(os.environ.get("N_THREADS"), 0),
        "flash_attn":         _str(os.environ.get("FLASH_ATTN"), "auto"),
        "cache_key":          _str(os.environ.get("CACHE_KEY"), "f16"),
        "cache_val":          _str(os.environ.get("CACHE_VAL"), "f16"),
        "cache_prompt":       _bool(os.environ.get("CACHE_PROMPT"), True),
        "chunked_prefill":    _bool(os.environ.get("CHUNKED_PREFILL"), True),
        "prefix_caching":     _bool(os.environ.get("PREFIX_CACHE"), True),
        "max_batched_tokens": _int(os.environ.get("MAX_BATCHED_TOKENS"), 0),
        "swap_space_gb":      _int(os.environ.get("SWAP_SPACE"), 0),
        "quantization":       _str(os.environ.get("QUANT"), "none"),
        "dtype":              _str(os.environ.get("DTYPE"), "auto"),
        "block_size":         _int(os.environ.get("BLOCK_SIZE"), 16),
        "attention_backend":  _str(os.environ.get("ATTN_BACKEND"), "flashinfer"),
        "radix_cache":        _bool(os.environ.get("RADIX_CACHE"), True),
        "torch_compile":      _bool(os.environ.get("TORCH_COMPILE"), True),
        "trust_remote_code":  _bool(os.environ.get("TRUST_REMOTE"), True),
        "max_total_tokens":   _int(os.environ.get("MAX_TOTAL_TOKENS"), 0),
        "chunked_prefill_size": _int(os.environ.get("CHUNKED_PS"), 0),
    },
    "hardware": {
        "hostname":       _str(os.environ.get("HOSTNAME_VAL")),
        "gpu_name":       _str(os.environ.get("GPU_NAME_VAL")),
        "gpu_count":      _int(os.environ.get("GPU_COUNT_VAL"), 0),
        "gpu_vram_mib":   _int(os.environ.get("GPU_VRAM_MIB_VAL"), 0),
        "driver_version": _str(os.environ.get("DRIVER_VAL")),
        "cuda_version":   _str(os.environ.get("CUDA_VAL")),
        "cpu_name":       _str(os.environ.get("CPU_NAME_VAL")),
        "cpu_cores":      _int(os.environ.get("CPU_CORES_VAL"), 0),
        "memory_gb":      _int(os.environ.get("MEM_GB_VAL"), 0),
    },
    "system": {
        "timestamp":      _str(os.environ.get("TS_VAL")),
        "git_commit":     _str(os.environ.get("COMMIT_VAL")),
        "git_branch":     _str(os.environ.get("BRANCH_VAL")),
        "docker_image":   _str(os.environ.get("IMG_VAL")),
        "server_version": _str(os.environ.get("VER_VAL")),
    },
}

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF

    echo "  [params] wrote $out"
}


# Standalone test invocation (for `bash write_params.sh` smoke test).
# Only runs when this file is executed directly, not when sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
    backend="${1:-llamacpp}"
    out="${2:-${PROJECT_ROOT}/results/${backend}/_active_params.json}"
    write_params "$backend" "$out"
    echo "Standalone test wrote: $out"
fi
