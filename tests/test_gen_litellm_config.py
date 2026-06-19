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
# (added in tasks 5 and 7)


# --- Tests T12–T14: generate_config integration ---
# (added in task 10)


# --- Tests T15–T21: main() error collection, summary, strict-mode ---
# (added in tasks 8, 9, 10)


# --- Test T22: end-to-end regression ---
# (added in task 11)
