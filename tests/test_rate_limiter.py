"""
Fast unit tests for the per-session rate limiter.

st.session_state and time.time() are both mocked so the sliding-window logic is
tested deterministically without a Streamlit runtime or real clock.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import rate_limiter
from rate_limiter import check_rate_limit, get_rate_limit_status


class _FakeSt:
    """Stand-in for the streamlit module exposing a plain-dict session_state."""
    def __init__(self):
        self.session_state = {}


@pytest.fixture
def fake_st(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(rate_limiter, "st", fake)
    return fake


@pytest.fixture
def frozen_time(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(rate_limiter.time, "time", lambda: clock["now"])
    return clock


def test_first_n_requests_within_limit_all_allowed(fake_st, frozen_time):
    for i in range(15):
        allowed, msg = check_rate_limit(max_requests=15, window_seconds=60)
        assert allowed is True, f"request {i + 1} should be allowed"
        assert msg is None


def test_request_over_limit_is_blocked_with_message(fake_st, frozen_time):
    for _ in range(15):
        check_rate_limit(max_requests=15, window_seconds=60)
    allowed, msg = check_rate_limit(max_requests=15, window_seconds=60)
    assert allowed is False
    assert msg and "wait" in msg.lower()


def test_allowed_again_after_window_passes(fake_st, frozen_time):
    for _ in range(15):
        check_rate_limit(max_requests=15, window_seconds=60)
    assert check_rate_limit(max_requests=15, window_seconds=60)[0] is False

    # Advance the clock past the window so the old timestamps expire.
    frozen_time["now"] += 61
    allowed, msg = check_rate_limit(max_requests=15, window_seconds=60)
    assert allowed is True
    assert msg is None


def test_two_sessions_do_not_affect_each_other(fake_st, frozen_time):
    # Fill user A to the limit.
    for _ in range(15):
        check_rate_limit(max_requests=15, window_seconds=60, session_key="user_a")
    assert check_rate_limit(
        max_requests=15, window_seconds=60, session_key="user_a"
    )[0] is False

    # User B has a fresh, independent window.
    allowed, msg = check_rate_limit(
        max_requests=15, window_seconds=60, session_key="user_b"
    )
    assert allowed is True
    assert msg is None


def test_status_reports_usage_without_consuming_a_slot(fake_st, frozen_time):
    check_rate_limit(max_requests=15, window_seconds=60)
    check_rate_limit(max_requests=15, window_seconds=60)
    status = get_rate_limit_status(max_requests=15, window_seconds=60)
    assert status["used"] == 2
    assert status["limit"] == 15
    # Reading status again must not increase usage.
    assert get_rate_limit_status(max_requests=15, window_seconds=60)["used"] == 2
