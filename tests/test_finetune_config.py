"""
Fast unit tests for the fine-tuned-model selection logic.

Tests Config.active_chat_model() — the switch that lets production serve a
fine-tuned model via env vars. No API calls; env vars are monkeypatched to
exercise the selection logic only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config


def test_defaults_to_base_model_when_flag_off(monkeypatch):
    # Ensure a clean env: neither var set -> base model, exactly as before.
    monkeypatch.delenv("USE_FINETUNED_MODEL", raising=False)
    monkeypatch.delenv("FINETUNED_MODEL_ID", raising=False)
    assert config.active_chat_model() == config.CHAT_MODEL


def test_uses_finetuned_id_when_enabled_and_set(monkeypatch):
    fake_id = "ft:gpt-4o-mini:acme:it-support:abc123"
    monkeypatch.setenv("USE_FINETUNED_MODEL", "true")
    monkeypatch.setenv("FINETUNED_MODEL_ID", fake_id)
    assert config.active_chat_model() == fake_id


def test_falls_back_to_base_when_id_empty(monkeypatch):
    # Flag on but no ID -> must NOT return an empty model name; fall back safely.
    monkeypatch.setenv("USE_FINETUNED_MODEL", "true")
    monkeypatch.setenv("FINETUNED_MODEL_ID", "")
    result = config.active_chat_model()
    assert result == config.CHAT_MODEL
    assert result  # non-empty, will never be passed as an empty model name


def test_flag_off_ignores_configured_id(monkeypatch):
    # ID present but flag off -> base model wins (explicit opt-in required).
    monkeypatch.setenv("USE_FINETUNED_MODEL", "false")
    monkeypatch.setenv("FINETUNED_MODEL_ID", "ft:gpt-4o-mini:acme:xyz")
    assert config.active_chat_model() == config.CHAT_MODEL
