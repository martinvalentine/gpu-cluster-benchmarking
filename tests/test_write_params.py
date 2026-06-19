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


def test_t2_type_coercion(tmp_path):
    """T2: numeric strings become int/float, '1' becomes True."""
    out = tmp_path / "params.json"
    env = _base_env()
    env.update({
        "MAX_MODEL_LEN": "4096",         # int
        "N_BATCH": "2048",                # int
        "GPU_MEM_UTIL": "0.87",           # float
        "CACHE_PROMPT": "1",              # bool True
        "PREFIX_CACHE": "--enable-prefix-caching",  # bool True via CLI flag
        "CHUNKED_PREFILL": "",            # bool default
        "TRUST_REMOTE": "--trust-remote-code",  # bool True
    })
    result = _run_write_params(env, out)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    data = json.loads(out.read_text())
    assert data["server"]["max_model_len"] == 4096
    assert isinstance(data["server"]["max_model_len"], int)
    assert data["server"]["batch"] == 2048
    assert isinstance(data["server"]["batch"], int)
    assert data["server"]["gpu_mem_util"] == 0.87
    assert isinstance(data["server"]["gpu_mem_util"], float)
    assert data["server"]["cache_prompt"] is True
    assert data["server"]["prefix_caching"] is True
    assert data["server"]["chunked_prefill"] is True  # default
    assert data["server"]["trust_remote_code"] is True


def test_t3_empty_vars_fall_back_to_defaults(tmp_path):
    """T3: empty env vars fall back to 'N/A' (strings) or default (ints/bools)."""
    out = tmp_path / "params.json"
    env = _base_env()
    # Wipe the model name and tp_size
    env["MODEL"] = ""
    env["MODEL_PATH"] = ""
    env["TP"] = ""
    env["PORT"] = ""
    result = _run_write_params(env, out)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    data = json.loads(out.read_text())
    assert data["server"]["model"] == "N/A"
    assert data["server"]["model_path"] == "N/A"
    assert data["server"]["port"] == 0           # default int
    assert data["server"]["tp_size"] == 1        # default int


def test_t4_missing_hardware_uses_n_a(tmp_path, monkeypatch):
    """T4: when nvidia-smi / nproc are unavailable, hardware fields use N/A or 0."""
    # Create stub scripts that fail, so the bash helper falls back to N/A / 0
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for cmd in ("nvidia-smi", "nproc"):
        stub = stub_dir / cmd
        stub.write_text("#!/bin/bash\nexit 1")
        stub.chmod(0o755)
    # Prepend stubs to PATH so they shadow real commands while bash itself stays findable
    monkeypatch.setenv("PATH", f"{stub_dir}:{os.environ['PATH']}")
    out = tmp_path / "params.json"
    env = _base_env()
    env["HOSTNAME_VAL"] = ""  # empty hostname also falls back
    result = _run_write_params(env, out)
    # nvidia-smi failing should not crash; output is still written
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["hardware"]["gpu_name"] == "N/A"
    assert data["hardware"]["gpu_count"] == 0
    assert data["hardware"]["gpu_vram_mib"] == 0
    assert data["hardware"]["cuda_version"] == "N/A"
    assert data["hardware"]["cpu_cores"] == 0
