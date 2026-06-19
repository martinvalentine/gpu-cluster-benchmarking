#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="llm-servers"

echo "Killing tmux session: $SESSION"
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Killing server processes..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true
pkill -f "litellm" 2>/dev/null || true

sleep 2
echo "All servers stopped."