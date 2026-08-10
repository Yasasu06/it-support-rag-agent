"""
Per-session rate limiting for the IT Support RAG Agent.

Uses a sliding window over timestamps stored in Streamlit session state, so the
limit is isolated per user session — one user's activity never affects another's.
Defaults are generous (15 requests / 60s) so a recruiter testing the app never
hits it, while obvious spam/abuse is blocked before it can incur API cost.
"""

import time
import streamlit as st
from typing import Optional


def check_rate_limit(
    max_requests: int = 15,
    window_seconds: int = 60,
    session_key: str = "rate_limit_history"
) -> tuple[bool, Optional[str]]:
    """
    Per-session sliding window rate limiter. Tracks
    request timestamps in Streamlit session state
    (isolated per user session, does not affect other
    users). Returns (is_allowed, error_message).

    Default: 15 requests per 60 seconds - generous
    enough that a real user testing the app normally
    never hits it, but blocks obvious spam/abuse.
    """
    now = time.time()

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    st.session_state[session_key] = [
        t for t in st.session_state[session_key]
        if now - t < window_seconds
    ]

    if len(st.session_state[session_key]) >= max_requests:
        oldest = st.session_state[session_key][0]
        wait_time = int(window_seconds - (now - oldest))
        return False, (
            f"You're sending questions quite quickly. "
            f"Please wait about {wait_time} seconds "
            f"before asking another question. This "
            f"helps keep the demo available for everyone."
        )

    st.session_state[session_key].append(now)
    return True, None


def get_rate_limit_status(
    max_requests: int = 15,
    window_seconds: int = 60,
    session_key: str = "rate_limit_history"
) -> dict:
    """Returns current usage for display purposes,
    without consuming a request slot."""
    now = time.time()
    history = st.session_state.get(session_key, [])
    recent = [t for t in history if now - t < window_seconds]
    return {
        "used": len(recent),
        "limit": max_requests,
        "window_seconds": window_seconds
    }
