#!/bin/bash
set -Eeuo pipefail

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO:  $*"; }
log_warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARN:  $*" >&2; }

# ── Workspace (after volume mount) ─────────────────────────────
mkdir -p /workspace/{models/hf,models/gguf,datasets,results,logs}

# ── Redis ──────────────────────────────────────────────────────
if ! redis-cli ping &>/dev/null; then
    log_info "Starting Redis..."
    redis-server --daemonize yes \
        --maxmemory 8gb \
        --maxmemory-policy allkeys-lru \
        --logfile /workspace/logs/redis.log || log_warn "Redis start failed (non-critical)."
    sleep 1
    redis-cli ping &>/dev/null && log_info "Redis started." || log_warn "Redis not responding."
fi

# ── SSH ────────────────────────────────────────────────────────
if [ -n "${PUBLIC_KEY:-}" ]; then
    log_info "Injecting SSH public key..."
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd 2>/dev/null || log_warn "SSH daemon start failed (non-critical)."

# ── Clone Benchmark Repo ───────────────────────────────────────
if [ -n "${BENCHMARK_REPO:-}" ] && [ ! -d "/workspace/gpu-cluster-benchmarking" ]; then
    log_info "Cloning benchmark repo: $BENCHMARK_REPO"
    git clone "$BENCHMARK_REPO" /workspace/gpu-cluster-benchmarking
fi

# ── Summary ────────────────────────────────────────────────────
log_info "=========================================="
log_info " LLM Serving Base Image Ready"
log_info "=========================================="
log_info ""
log_info "Frameworks:"
command -v vllm >/dev/null 2>&1 && log_info "  vLLM:       $(command -v vllm)"
python3 -c "import sglang" >/dev/null 2>&1 && log_info "  SGLang:     python3 -m sglang.launch_server"
command -v llama-server >/dev/null 2>&1 && log_info "  llama.cpp:  $(command -v llama-server)"
command -v llama-bench >/dev/null 2>&1 && log_info "  turboquant: $(command -v llama-bench)"
log_info ""
log_info "GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || log_warn "nvidia-smi not available."
log_info ""
log_info "Workspace: /workspace"
log_info "=========================================="

# ── Execute ────────────────────────────────────────────────────
exec "$@"
