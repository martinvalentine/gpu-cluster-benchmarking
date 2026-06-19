"""Tests for scripts/gen-litellm-config.py.

The tests verify:
- T1–T11: helper-level resolution (HF path, GGUF filename, dispatcher)
- T12–T14: generate_config integration
- T15–T21: main() error collection, summary, strict-mode
- T22: end-to-end regression against the committed litellm_config.yaml
"""
from pathlib import Path

import pytest
import yaml

# Project-relative path; distinct from the container's /workspace/models.
# Used by T22's skipif and the same path the test passes to --base-dir.
REPO_MODELS_DIR = Path("models")
T22_SKIP_REASON = f"T22 requires real GGUF files at {REPO_MODELS_DIR}/gguf/"


# --- Tests T1–T11: helper-level resolution ---
import importlib.util as _importlib_util
from pathlib import Path as _Path

_SCRIPT_PATH = _Path(__file__).resolve().parent.parent / "scripts" / "gen-litellm-config.py"
_spec = _importlib_util.spec_from_file_location("gen_litellm_config", _SCRIPT_PATH)
glc = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(glc)


# T6: zero files in any dir raises UnresolvableFilename
def test_gguf_zero_matches_raises_unresolvable(tmp_path):
    """No .gguf in any search dir → UnresolvableFilename."""
    model = {
        "name": "test-model",
        "backend": "llamacpp",
        "local_dir": "gguf/missing",
        "include": "*q4_k_m.gguf",
    }
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    with pytest.raises(glc.UnresolvableFilename) as exc_info:
        glc.resolve_gguf_filename(model, [empty_dir])
    # Verify the exception carries the searched dirs and the include pattern
    assert "test-model" in str(exc_info.value)
    assert "include pattern" in str(exc_info.value).lower() or "include" in str(exc_info.value)
    assert "*q4_k_m.gguf" in str(exc_info.value)


# T7: file in second search_dir when first empty → returns filename
def test_gguf_finds_file_in_second_search_dir(tmp_path):
    """First search dir has no files; second has the model → resolve from second."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    populated_dir = tmp_path / "populated"
    model_dir = populated_dir / "gguf" / "test-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model-q4_k_m.gguf").write_bytes(b"\x00" * 100)

    model = {
        "name": "test-model",
        "backend": "llamacpp",
        "local_dir": "gguf/test-model",
        "include": "*q4_k_m.gguf",
    }
    result = glc.resolve_gguf_filename(model, [empty_dir, populated_dir])
    assert result == "model-q4_k_m.gguf"


# T5: multiple .gguf files in a single search dir → raises AmbiguousFilename
def test_gguf_multiple_matches_raises_ambiguous(tmp_path):
    """Multiple .gguf files in the same dir → AmbiguousFilename with candidate list."""
    model_dir = tmp_path / "gguf" / "test-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model-q4_k_m.gguf").write_bytes(b"\x00" * 100)
    (model_dir / "model-q8_0.gguf").write_bytes(b"\x00" * 100)

    model = {
        "name": "test-model",
        "backend": "llamacpp",
        "local_dir": "gguf/test-model",
        "include": "*q4_k_m.gguf",
    }
    with pytest.raises(glc.AmbiguousFilename) as exc_info:
        glc.resolve_gguf_filename(model, [tmp_path])
    # Verify the exception lists the candidates
    assert "model-q4_k_m.gguf" in str(exc_info.value)
    assert "model-q8_0.gguf" in str(exc_info.value)


# T8: embedding model (special case) — returns just filename, no path
def test_gguf_embedding_model_returns_filename_only(tmp_path):
    """Embedding model: GGUF filename without any path prefix."""
    model_dir = tmp_path / "gguf" / "qwen3-embedding-0.6b"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3-Embedding-0.6B-Q8_0.gguf").write_bytes(b"\x00" * 100)

    model = {
        "name": "qwen3-embedding",
        "backend": "llamacpp",
        "phase": "embedding",
        "local_dir": "gguf/qwen3-embedding-0.6b",
        "include": "*Q8_0.gguf",
    }
    result = glc.resolve_gguf_filename(model, [tmp_path])
    assert result == "Qwen3-Embedding-0.6B-Q8_0.gguf"
    # No path prefix — just the filename
    assert "/" not in result


# T1: resolve_hf_path returns search_dirs[0]/local_dir
def test_hf_path_returns_first_search_dir(tmp_path):
    """HF path: returns search_dirs[0] joined with local_dir."""
    model = {"name": "test", "backend": "vllm", "local_dir": "hf/test"}
    result = glc.resolve_hf_path(model, [tmp_path], strict=False)
    assert result == str(tmp_path / "hf" / "test")


# T2: strict=True + missing dir raises FileNotFoundError with remediation
def test_hf_path_strict_raises_with_remediation(tmp_path):
    """Strict mode + missing dir → FileNotFoundError with remediation block."""
    missing = tmp_path / "does_not_exist"
    model = {"name": "test", "backend": "vllm", "local_dir": "hf/missing"}
    with pytest.raises(FileNotFoundError) as exc_info:
        glc.resolve_hf_path(model, [missing], strict=True)
    msg = str(exc_info.value)
    assert "Remediation" in msg
    assert "--base-dir" in msg


# T3: strict=False + missing dir returns path anyway (soft pass)
def test_hf_path_non_strict_returns_path_even_if_missing(tmp_path):
    """Non-strict mode + missing dir → returns path; caller notes the warning."""
    missing = tmp_path / "does_not_exist"
    model = {"name": "test", "backend": "vllm", "local_dir": "hf/missing"}
    result = glc.resolve_hf_path(model, [missing], strict=False)
    assert result == str(missing / "hf" / "missing")


# --- Tests T12–T14: generate_config integration ---
# (added in task 10)


# --- Tests T15–T21: main() error collection, summary, strict-mode ---
# (added in tasks 8, 9, 10)


# --- Test T22: end-to-end regression ---
# (added in task 11)
