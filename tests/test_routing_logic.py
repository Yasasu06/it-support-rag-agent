"""
Fast unit tests for the analytical-vs-RAG routing logic in app.py.

is_analytical_query() is pure keyword logic (no API, no network). Importing app
pulls in Streamlit and the pipeline modules, but all OpenAI clients are built
lazily, so this runs without an API key — which is what CI relies on.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import is_analytical_query


def test_how_many_is_analytical():
    assert is_analytical_query("how many VPN tickets") is True


def test_live_servicenow_status_is_analytical():
    assert is_analytical_query("What is the current live ServiceNow status?") is True


def test_average_resolution_time_is_analytical():
    assert is_analytical_query("average resolution time for hardware") is True


def test_plain_support_question_is_not_analytical():
    assert is_analytical_query("my printer is broken") is False


def test_routing_is_case_insensitive():
    assert is_analytical_query("HOW MANY password resets happened") is True


def test_account_is_not_misrouted_as_count():
    # Regression test: "count" is a substring of "account". A naive `kw in
    # q_lower` check misrouted any question mentioning "account" to the
    # analytical tool-agent path, which always renders with sources=[] -
    # silently dropping the "Source tickets" expander even though the
    # tool-agent produced a real, ticket-citing answer.
    assert is_analytical_query("I cannot access SAP after my account was locked") is False
    assert is_analytical_query("Account locked after too many wrong password attempts") is False


def test_sla_is_not_misrouted_inside_unrelated_words():
    # Same substring-collision class: "sla" is a substring of "Slack" and
    # "translate".
    assert is_analytical_query("My Slack notifications are not working") is False
    assert is_analytical_query("Can you translate this error message") is False


def test_sla_still_routes_when_it_is_a_real_word():
    assert is_analytical_query("Can you show me SLA compliance stats") is True
