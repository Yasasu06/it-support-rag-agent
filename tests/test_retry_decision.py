"""
Fast unit tests for retry_decision() in agent_pipeline.py.

This is the pure routing function for the adaptive-retry conditional edge —
no API calls, no LLM, instant. Formalizes the manual checks done when the
retry loop was first built.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_pipeline import retry_decision


def test_failed_verification_with_retries_left_retries():
    state = {"grounded": False, "escalation": True, "retry_count": 0}
    assert retry_decision(state) == "retry"


def test_retry_budget_exhausted_ends():
    state = {"grounded": False, "escalation": True, "retry_count": 2}
    assert retry_decision(state) == "end"


def test_grounded_answer_ends():
    state = {"grounded": True, "escalation": False, "retry_count": 0}
    assert retry_decision(state) == "end"


def test_escalation_without_grounding_failure_retries():
    # escalation True (e.g. low confidence) with grounded True still retries
    # while budget remains.
    state = {"grounded": True, "escalation": True, "retry_count": 1}
    assert retry_decision(state) == "retry"
