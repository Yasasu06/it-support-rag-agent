"""
Fast unit tests for the centralized failure-handling utilities.

No API keys, no network - the decorators are exercised with local functions
that deliberately raise, so we test the retry/fallback/validation logic itself.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from error_handling import (
    with_retry,
    safe_fallback,
    validate_user_input,
    ServiceUnavailableError,
)


# --- with_retry: succeeds after transient failures --------------------------
def test_with_retry_succeeds_after_two_failures():
    calls = {"n": 0}

    @with_retry(max_attempts=3, delay_seconds=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 3  # failed twice, succeeded on the 3rd


# --- with_retry: raises ServiceUnavailableError when exhausted --------------
def test_with_retry_raises_service_unavailable_when_always_failing():
    calls = {"n": 0}

    @with_retry(max_attempts=2, delay_seconds=0)
    def always_fails():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ServiceUnavailableError):
        always_fails()
    assert calls["n"] == 2  # tried exactly max_attempts times, no hang


# --- safe_fallback: returns the fallback instead of propagating -------------
def test_safe_fallback_returns_fallback_on_exception():
    @safe_fallback(fallback_value="FALLBACK", log_message="boom")
    def explodes():
        raise RuntimeError("kaboom")

    assert explodes() == "FALLBACK"


def test_safe_fallback_passes_through_on_success():
    @safe_fallback(fallback_value="FALLBACK")
    def ok():
        return "real-value"

    assert ok() == "real-value"


# --- validate_user_input ----------------------------------------------------
def test_validate_rejects_empty_string():
    ok, msg = validate_user_input("")
    assert ok is False
    assert msg  # a non-empty error message


def test_validate_rejects_whitespace_only():
    ok, msg = validate_user_input("    ")
    assert ok is False
    assert msg


def test_validate_rejects_oversized_input():
    ok, msg = validate_user_input("x" * 3000, max_length=2000)
    assert ok is False
    assert "too long" in msg.lower()


def test_validate_accepts_normal_question():
    ok, msg = validate_user_input("My printer is not working")
    assert ok is True
    assert msg is None
