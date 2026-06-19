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


# T9: dispatcher routes llamacpp → resolve_gguf_filename
def test_dispatcher_routes_llamacpp(tmp_path, monkeypatch):
    """Dispatcher calls resolve_gguf_filename for llamacpp models."""
    model = {
        "name": "test", "backend": "llamacpp", "local_dir": "gguf/missing",
        "include": "*q4_k_m.gguf",
    }
    called_with = []
    real_gguf = glc.resolve_gguf_filename

    def spy(model, search_dirs):
        called_with.append((model["name"], tuple(search_dirs)))
        return real_gguf(model, search_dirs)

    monkeypatch.setattr(glc, "resolve_gguf_filename", spy)
    with pytest.raises(glc.UnresolvableFilename):
        glc.resolve_model_path(model, [tmp_path], strict=False)
    assert called_with == [("test", (tmp_path,))]


# T10: dispatcher routes vllm → resolve_hf_path
def test_dispatcher_routes_vllm(tmp_path, monkeypatch):
    """Dispatcher calls resolve_hf_path for vllm models."""
    # Create the path so the spy's delegation to the real helper doesn't raise.
    (tmp_path / "hf" / "test").mkdir(parents=True)
    model = {"name": "test", "backend": "vllm", "local_dir": "hf/test"}
    called_with = []
    real_hf = glc.resolve_hf_path

    def spy(model, search_dirs, strict):
        called_with.append((model["backend"], tuple(search_dirs), strict))
        return real_hf(model, search_dirs, strict)

    monkeypatch.setattr(glc, "resolve_hf_path", spy)
    glc.resolve_model_path(model, [tmp_path], strict=True)
    assert called_with == [("vllm", (tmp_path,), True)]


# T11: dispatcher routes sglang → resolve_hf_path
def test_dispatcher_routes_sglang(tmp_path, monkeypatch):
    """Dispatcher calls resolve_hf_path for sglang models (same as vllm)."""
    model = {"name": "test", "backend": "sglang", "local_dir": "hf/test"}
    called_with = []
    real_hf = glc.resolve_hf_path

    def spy(model, search_dirs, strict):
        called_with.append((model["backend"], tuple(search_dirs), strict))
        return real_hf(model, search_dirs, strict)

    monkeypatch.setattr(glc, "resolve_hf_path", spy)
    glc.resolve_model_path(model, [tmp_path], strict=False)
    assert called_with == [("sglang", (tmp_path,), False)]


# --- Tests T12–T14: generate_config integration ---


# T12: disabled models are skipped
def test_generate_config_skips_disabled_models(tmp_path):
    config = {
        "base_dir": str(tmp_path), "ports": {},
        "models": [
            {"name": "enabled", "backend": "vllm", "enabled": True,
             "endpoint": True, "local_dir": "hf/a", "vllm_tp": 1},
            {"name": "disabled", "backend": "vllm", "enabled": False,
             "endpoint": True, "local_dir": "hf/b", "vllm_tp": 1},
        ],
    }
    result, _ = glc.generate_config(config, Path("."), search_dirs=[], strict=False)
    names = [m["model_name"] for m in result["model_list"]]
    assert "enabled-vllm" in names
    assert "disabled-vllm" not in names


# T13: models with endpoint=False are skipped
def test_generate_config_skips_non_endpoint_models(tmp_path):
    config = {
        "base_dir": str(tmp_path), "ports": {},
        "models": [
            {"name": "endpoint", "backend": "vllm", "enabled": True,
             "endpoint": True, "local_dir": "hf/a", "vllm_tp": 1},
            {"name": "no-endpoint", "backend": "vllm", "enabled": True,
             "endpoint": False, "local_dir": "hf/b", "vllm_tp": 1},
        ],
    }
    result, _ = glc.generate_config(config, Path("."), search_dirs=[], strict=False)
    names = [m["model_name"] for m in result["model_list"]]
    assert "endpoint-vllm" in names
    assert "no-endpoint-vllm" not in names


# T14: generate_config with valid config + models on disk produces full proxy config
def test_generate_config_full_round_trip(tmp_path):
    """End-to-end: build a small config with real files, check the output structure."""
    # Create model dirs
    vllm_dir = tmp_path / "models" / "hf" / "test-vllm"
    vllm_dir.mkdir(parents=True)
    llamacpp_dir = tmp_path / "models" / "gguf" / "test-gguf"
    llamacpp_dir.mkdir(parents=True)
    (llamacpp_dir / "model-q4_k_m.gguf").write_bytes(b"\x00")

    config = {
        "base_dir": str(tmp_path / "models"), "ports": {"vllm": 8000, "llamacpp": 8001},
        "models": [
            {"name": "test-vllm", "backend": "vllm", "enabled": True,
             "endpoint": True, "local_dir": "hf/test-vllm", "proxy_name": "test-vllm-vllm",
             "vllm_tp": 1},
            {"name": "test-gguf", "backend": "llamacpp", "enabled": True,
             "endpoint": True, "local_dir": "gguf/test-gguf", "proxy_name": "test-gguf-llamacpp",
             "include": "*q4_k_m.gguf"},
        ],
    }
    result, _ = glc.generate_config(config, Path("."), search_dirs=[], strict=False)
    assert len(result["model_list"]) == 2
    by_name = {m["model_name"]: m for m in result["model_list"]}
    assert by_name["test-vllm-vllm"]["litellm_params"]["api_base"] == "http://localhost:8000/v1"
    assert by_name["test-gguf-llamacpp"]["litellm_params"]["model"] == "openai/model-q4_k_m.gguf"


# --- Tests T15–T21: main() error collection, summary, strict-mode ---


# T16: --base-dir with nonexistent path → exit 1 with hard error
def test_main_base_dir_nonexistent_exits_1(tmp_path, capsys):
    """--base-dir with a path that doesn't exist triggers a hard error."""
    nonexistent = tmp_path / "nonexistent"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"models": [
        {"name": "test", "backend": "vllm", "enabled": True, "endpoint": True,
         "local_dir": "hf/test", "vllm_tp": 1}
    ], "base_dir": str(tmp_path), "ports": {}}))
    with pytest.raises(SystemExit) as exc_info:
        glc.main(["--config", str(config_path), "--base-dir", str(nonexistent),
                  "--preview"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing written to stdout
    assert "Remediation" in captured.err
    assert "test" in captured.err  # model name appears in the error


# T19: search_dirs=[] + strict=True → raises UnresolvableFilename (nothing to search)
def test_resolve_with_empty_search_dirs_strict(tmp_path):
    """Combinatorial: empty search_dirs + strict=True → UnresolvableFilename."""
    model = {
        "name": "test", "backend": "llamacpp", "local_dir": "gguf/missing",
        "include": "*q4_k_m.gguf",
    }
    with pytest.raises(glc.UnresolvableFilename):
        glc.resolve_model_path(model, [], strict=True)


# T15: main() with --preview and a hard error exits 1 with nothing on stdout
def test_main_preview_hard_error_exits_1_no_stdout(tmp_path, capsys):
    """--preview + hard error → exit 1, stdout empty (YAML not printed)."""
    config_path = tmp_path / "config.yaml"
    # A llamacpp model with no files anywhere → UnresolvableFilename
    config_path.write_text(yaml.dump({
        "base_dir": str(tmp_path / "models"),
        "ports": {},
        "models": [
            {"name": "broken", "backend": "llamacpp", "enabled": True,
             "endpoint": True, "local_dir": "gguf/missing", "include": "*q4_k_m.gguf",
             "phase": "p0"},
        ],
    }))
    with pytest.raises(SystemExit) as exc_info:
        glc.main(["--config", str(config_path), "--preview"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is empty (no YAML printed on hard error)


# T17: main() with soft warning exits 0, output written, warning on stderr
def test_main_soft_warning_exits_0_with_output(tmp_path, capsys):
    """HF path missing in non-strict mode → exit 0, output written, WARN in stderr."""
    # Use the real config but override base_dir to a path that doesn't have HF dirs
    config_path = tmp_path / "config.yaml"
    hf_root = tmp_path / "hf_root"  # exists, but subdirs don't
    hf_root.mkdir()
    config_path.write_text(yaml.dump({
        "base_dir": str(hf_root),
        "ports": {"vllm": 8000},
        "models": [
            {"name": "test-vllm", "backend": "vllm", "enabled": True,
             "endpoint": True, "local_dir": "hf/missing", "proxy_name": "test-vllm-vllm",
             "vllm_tp": 1},
        ],
    }))
    glc.main(["--config", str(config_path), "--output", str(tmp_path / "out.yaml")])
    captured = capsys.readouterr()
    assert (tmp_path / "out.yaml").exists()  # output was written
    assert "WARN" in captured.err
    assert "test-vllm-vllm" in captured.err


# T18: main() with --preview keeps stdout clean (pipeable)
def test_main_preview_keeps_stdout_clean_for_piping(tmp_path, capsys):
    """--preview → stdout is pure YAML, summary goes to stderr."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "base_dir": str(tmp_path / "models"),
        "ports": {},
        "models": [],  # no models → no errors, just empty output
    }))
    glc.main(["--config", str(config_path), "--preview"])
    captured = capsys.readouterr()
    # stdout: pure YAML, starts with model_list: (or similar YAML structure)
    assert "model_list" in captured.out or captured.out.strip() == ""  # valid YAML or empty
    # stderr: contains the summary
    assert "Generated LiteLLM config" in captured.err


# T21: Two models with different failure types in one run → both errors printed
def test_main_collects_all_hard_errors_no_early_exit(tmp_path, capsys):
    """Both errors reach stderr before exit 1; output is not written."""
    config_path = tmp_path / "config.yaml"
    # Model A: llamacpp, no files anywhere → UnresolvableFilename
    # Model B: llamacpp, multiple files in dir → AmbiguousFilename
    config_path.write_text(yaml.dump({
        "base_dir": str(tmp_path / "models"),
        "ports": {},
        "models": [
            {"name": "modelA", "backend": "llamacpp", "enabled": True,
             "endpoint": True, "local_dir": "gguf/modelA", "include": "*q4_k_m.gguf",
             "phase": "p0"},
            {"name": "modelB", "backend": "llamacpp", "enabled": True,
             "endpoint": True, "local_dir": "gguf/modelB", "include": "*q4_k_m.gguf",
             "phase": "p0"},
        ],
    }))
    # Set up modelB dir with 2 .gguf files (triggers AmbiguousFilename)
    modelB_dir = tmp_path / "models" / "gguf" / "modelB"
    modelB_dir.mkdir(parents=True)
    (modelB_dir / "model-q4_k_m.gguf").write_bytes(b"\x00")
    (modelB_dir / "model-q8_0.gguf").write_bytes(b"\x00")
    # modelA dir doesn't exist → UnresolvableFilename

    with pytest.raises(SystemExit) as exc_info:
        glc.main(["--config", str(config_path), "--preview"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing on stdout
    # Both errors should be in stderr
    assert "modelA" in captured.err
    assert "modelB" in captured.err
    assert "Aborted" in captured.err
    assert "2 hard error" in captured.err


# T20: Summary formatter marks HF-missing models as WARN with path message
def test_summary_marks_hf_missing_as_warn(capsys):
    """Summary lists HF-missing models with WARN tag and a specific path message."""
    import sys as _sys
    litellm_config = {
        "model_list": [
            {"model_name": "ok-model",
             "litellm_params": {"model": "openai/ok.gguf", "api_base": "http://x", "api_key": "EMPTY"}},
            {"model_name": "warn-model",
             "litellm_params": {"model": "openai/missing/path", "api_base": "http://y", "api_key": "EMPTY"}},
        ],
    }
    config = {}  # not used by print_summary

    class FakeArgs:
        base_dirs = None
        config = Path("config.yaml")
        output = Path("out.yaml")
        preview = False

    soft_warnings = [
        ("warn-model", "/missing/path not found in /missing"),
    ]
    glc.print_summary(
        litellm_config, config, Path("."), FakeArgs(),
        file=_sys.stderr, soft_warnings=soft_warnings,
    )
    captured = capsys.readouterr()
    assert "OK    ok-model" in captured.err
    assert "WARN  warn-model" in captured.err
    assert "/missing/path not found" in captured.err


# (added in tasks 8, 9, 10)


# --- Test T22: end-to-end regression ---
# (added in task 11)
