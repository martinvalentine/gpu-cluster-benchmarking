#!/bin/bash
set -Eeuo pipefail

trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO:  $*"; }
log_warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARN:  $*" >&2; }

WORKSPACE="${WORKSPACE:-/workspace}"

log_info "=== Bare Machine Init — GPU Cluster Benchmarking ==="
log_info "Workspace: $WORKSPACE"

# ── 1. System packages ─────────────────────────────────────────────
log_info "Installing system packages..."

apt-get update -y
apt-get install -y --no-install-recommends \
    htop nvtop iotop \
    wget curl git tmux jq \
    cmake build-essential \
    openssh-server \
    ca-certificates gnupg lsb-release \
    unzip tar gzip \
    procps less \
    vim \
    redis-server \
    libssl-dev libffi-dev zlib1g-dev \
    pkg-config

rm -rf /var/lib/apt/lists/*

log_info "System packages installed."

# ── 2. uv ──────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    log_info "uv already installed: $(uv --version)"
else
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    log_info "uv installed: $(uv --version)"
fi

if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# ── 3. SSH ─────────────────────────────────────────────────────────
log_info "Configuring SSH..."
mkdir -p /run/sshd
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "$PUBLIC_KEY" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    log_info "SSH public key injected from PUBLIC_KEY env var."
fi

# ── 4. Workspace directories ───────────────────────────────────────
log_info "Creating workspace directories at $WORKSPACE..."
mkdir -p "$WORKSPACE"/{models/hf,models/gguf,datasets,results,logs}
mkdir -p "$WORKSPACE"/hf_cache

log_info "Workspace directories created."

# ── 5. Redis ───────────────────────────────────────────────────────
log_info "Starting Redis..."
if redis-cli ping &>/dev/null; then
    log_info "Redis already running."
else
    redis-server --daemonize yes \
        --maxmemory 8gb \
        --maxmemory-policy allkeys-lru \
        --logfile "$WORKSPACE/logs/redis.log" || log_warn "Redis start failed (non-critical)."
    sleep 1
    redis-cli ping && log_info "Redis started." || log_warn "Redis not responding."
fi

# ── 6. GPU verification ───────────────────────────────────────────
log_info "Verifying GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    log_info "GPU topology:"
    nvidia-smi topo -m || true
else
    log_warn "nvidia-smi not found — GPU drivers may not be installed."
fi

# ── 7. Git submodules ──────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -f "$PROJECT_ROOT/.gitmodules" ]]; then
    log_info "Initializing git submodules..."
    git -C "$PROJECT_ROOT" submodule update --init --recursive || log_warn "Submodule init failed."
fi

# ── Summary ────────────────────────────────────────────────────────
log_info "=========================================="
log_info " Init complete!"
log_info "=========================================="
log_info ""
log_info "Installed tools:"
command -v uv         &>/dev/null && log_info "  uv:         $(uv --version)"
command -v vim        &>/dev/null && log_info "  vim:        $(vim --version | head -1)"
command -v git        &>/dev/null && log_info "  git:        $(git --version)"
command -v tmux       &>/dev/null && log_info "  tmux:       $(tmux -V)"
command -v redis-cli  &>/dev/null && log_info "  redis-cli:  $(redis-cli --version)"
command -v nvidia-smi &>/dev/null && log_info "  nvidia-smi: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
log_info ""
log_info "Workspace: $WORKSPACE"
log_info "=========================================="
