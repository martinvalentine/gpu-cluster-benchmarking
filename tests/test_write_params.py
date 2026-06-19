"""Unit tests for scripts/run/write_params.sh.

Tests invoke the bash helper via subprocess, set env vars, and verify the
resulting JSON has the expected shape and values.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER = REPO_ROOT / "scripts" / "run" / "write_params.sh"


def _run_write_params(env_overrides: dict, out_path: Path) -> subprocess.CompletedProcess:
    """Source the helper and call write_params with the given env."""
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "-c", f"source {HELPER} && write_params llamacpp {out_path}"],
        env=env, capture_output=True, text=True, check=False,
    )


def _base_env() -> dict:
    """A baseline of env vars that all tests can start from."""
    return {
        "MODEL": "qwen2.5-0.5b-instruct-q4_k_m",
        "MODEL_PATH": "/workspace/models/gguf/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "PORT": "8001",
        "TP": "1",
        "MAX_MODEL_LEN": "4096",
        "MAX_NUM_SEQS": "256",
        "GPU_MEM_UTIL": "0.87",
        "N_PARALLEL": "4",
        "CTX_SIZE": "8192",
        "N_BATCH": "2048",
        "N_UBATCH": "512",
        "N_THREADS": "60",
        "FLASH_ATTN": "on",
        "CACHE_KEY": "q8_0",
        "CACHE_VAL": "turbo4",
        "CACHE_PROMPT": "1",
        "PREFIX_CACHE": "--enable-prefix-caching",
        "CHUNKED_PREFILL": "--enable-chunked-prefill",
        "MAX_BATCHED_TOKENS": "8192",
        "SWAP_SPACE": "4",
        "QUANT": "none",
        "DTYPE": "auto",
        "BLOCK_SIZE": "16",
        "ATTN_BACKEND": "flashinfer",
        "RADIX_CACHE": "",
        "TORCH_COMPILE": "--enable-torch-compile",
        "MAX_TOTAL_TOKENS": "1048576",
        "CHUNKED_PS": "8192",
        "TRUST_REMOTE": "--trust-remote-code",
        "PROJECT_ROOT": str(REPO_ROOT),
    }


def test_t1_basic_call_produces_valid_json(tmp_path):
    """T1: write_params produces a valid JSON with the expected top-level keys."""
    out = tmp_path / "params.json"
    result = _run_write_params(_base_env(), out)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data.keys()) == {"server", "hardware", "system"}
    assert data["server"]["model"] == "qwen2.5-0.5b-instruct-q4_k_m"
    assert data["server"]["port"] == 8001
    assert data["server"]["tp_size"] == 1
    assert data["server"]["cache_key"] == "q8_0"
    assert data["server"]["cache_val"] == "turbo4"
    # Hardware keys exist (values depend on host, but keys should be present)
    assert "gpu_name" in data["hardware"]
    assert "gpu_count" in data["hardware"]
    # System keys exist
    assert "timestamp" in data["system"]
    assert "git_commit" in data["system"]
