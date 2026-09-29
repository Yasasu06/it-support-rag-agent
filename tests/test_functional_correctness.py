"""Checks that chat status and retrieval behavior match the executed path."""

from contextlib import nullcontext
import inspect
import json
from types import SimpleNamespace

import pytest

import agent_pipeline
import app
import live_eval
import security


def test_tier_two_notice_does_not_claim_a_ticket_was_queued():
    assert "Tier 2 Review Recommended" in app.ESCALATION_HTML
    assert "No ServiceNow ticket was created automatically" in app.ESCALATION_HTML
    assert "has been queued" not in inspect.getsource(app)


def test_rag_status_separates_sources_grounding_and_review():
    labels = app.response_status_labels({
        "response_kind": "rag",
        "sources": [{"ticket_id": "TKT-151"}],
        "grounded": True,
        "escalation": False,
        "confidence_score": 0.25,
    })
    assert labels == [
        ("Retrieval", "1 source ticket retrieved"),
        ("Grounding", "Check passed"),
        ("Review", "Tier 1 response"),
    ]
    assert app.response_status_labels({
        "response_kind": "rag", "sources": [], "grounded": None,
        "escalation": True,
    }) == [
        ("Retrieval", "0 source tickets retrieved"),
        ("Grounding", "Not run"),
        ("Review", "Tier 2 recommended"),
    ]


def test_analytical_status_does_not_claim_retrieval_or_grounding():
    assert app.response_status_labels({
        "response_kind": "analytical", "tool_status": "completed",
        "confidence_score": None,
    }) == [("Tool status", "Analytical tool completed")]
    assert app.response_status_labels({
        "response_kind": "analytical", "tool_status": "unavailable",
    }) == [("Tool status", "Analytical tool unavailable")]


def test_session_retry_rate_counts_only_support_answers():
    messages = [
        {"role": "user", "content": "VPN help"},
        {"role": "assistant", "response_kind": "rag", "graph_completed": True, "retry_count": 1},
        {"role": "assistant", "response_kind": "rag", "graph_completed": True, "retry_count": 0},
        {"role": "assistant", "response_kind": "rag", "graph_completed": False, "retry_count": 0},
        {"role": "assistant", "response_kind": "rag", "retry_count": 0},
        {"role": "assistant", "response_kind": "analytical", "retry_count": 0},
    ]
    assert app.session_retry_summary(messages) == (1, 2, 50)
    assert app.session_retry_summary(messages[3:]) == (0, 0, 0)


def test_degraded_pipeline_response_does_not_mark_graph_complete(monkeypatch):
    def fail_pipeline(*_args, **_kwargs):
        raise RuntimeError("pipeline unavailable")

    monkeypatch.setattr(agent_pipeline, "_run_agent_pipeline_impl", fail_pipeline)
    result = agent_pipeline.run_agent_pipeline("VPN help")
    assert result["graph_completed"] is False
    assert result["escalation"] is True
    assert app.session_retry_summary([{
        "role": "assistant", "response_kind": "rag", **result,
    }]) == (0, 0, 0)


@pytest.mark.parametrize("groundings, expected", [
    ([True, None, False], 0.5),
    ([True, False, True], 0.667),
    ([None, None], None),
])
def test_grounded_rate_uses_only_checked_runs(
    monkeypatch, tmp_path, groundings, expected,
):
    path = tmp_path / "live_eval.jsonl"
    path.write_text("\n".join(json.dumps({
        "grounded": grounded,
        "relevance_score": 0.6,
        "latency_seconds": 1.0,
        "passed_quality_bar": False,
    }) for grounded in groundings) + "\n")
    monkeypatch.setattr(live_eval, "LIVE_EVAL_LOG", str(path))

    summary = live_eval.get_live_eval_summary()
    assert summary["total_scored"] == len(groundings)
    assert summary["grounded_rate"] == expected
    assert app.format_grounded_rate(expected) == (
        "N/A" if expected is None else f"{expected:.0%}"
    )


def test_status_renderer_uses_no_confidence_percentage(monkeypatch):
    rendered = []
    monkeypatch.setattr(app.st, "markdown", lambda text, **_kwargs: rendered.append(text))
    app.render_response_status({
        "response_kind": "rag", "sources": [], "grounded": False,
        "escalation": True, "confidence_score": 0.9,
    })
    assert "Retrieval" in rendered[0]
    assert "Review needed" in rendered[0]
    assert "0.9" not in rendered[0] and "%" not in rendered[0]
    assert "* 400" not in inspect.getsource(app)


def test_tool_agent_records_no_retrieval_score(monkeypatch):
    class FakeAgent:
        def invoke(self, *_args, **_kwargs):
            return {"messages": [SimpleNamespace(content="There are five VPN tickets.")]}

    recorded = {}
    monkeypatch.setattr(agent_pipeline, "_get_tool_agent", lambda: FakeAgent())
    monkeypatch.setattr(agent_pipeline, "mask_pii", lambda text: (text, []))
    monkeypatch.setattr(agent_pipeline, "is_safe_query", lambda _text: (True, []))
    monkeypatch.setattr(agent_pipeline, "log_query", lambda **kwargs: recorded.update(kwargs))

    result = agent_pipeline.run_tool_agent("how many VPN tickets")
    assert result["tool_status"] == "completed"
    assert result["confidence_score"] is None
    assert result["tier"] is None
    assert recorded["confidence_score"] is None


def test_audit_summary_skips_missing_retrieval_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "LOG_FILE", str(tmp_path / "audit.jsonl"))
    security.log_query("tool", "tool", "answer", None, False, None, [], [])
    security.log_query("rag", "rag", "answer", 0.25, False, "Tier 1", [], [])
    entries = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert entries[0]["confidence_score"] is None
    assert security.get_audit_summary()["avg_confidence"] == 0.25


@pytest.mark.parametrize("category, expected_filter", [
    ("VPN", {"category": "VPN"}),
    (None, None),
])
def test_retrieval_uses_category_metadata_filter(monkeypatch, category, expected_filter):
    captured = {}

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            captured.update({"query": query, **kwargs})
            return []

    monkeypatch.setattr(agent_pipeline, "_get_vectorstore", lambda: FakeStore())
    retrieval = getattr(
        agent_pipeline.retrieval_agent, "__wrapped__", agent_pipeline.retrieval_agent
    )
    retrieval({
        "question": "VPN connection", "category": category,
        "retry_count": 0, "retry_history": [],
    })
    assert captured["query"] == "VPN connection"
    if expected_filter is None:
        assert "filter" not in captured
    else:
        assert captured["filter"] == expected_filter


def _initial_graph_state(category="VPN"):
    return {
        "question": "VPN connection fails", "category": category,
        "context": None, "answer": None, "escalation": None, "tier": None,
        "confidence_score": None, "grounded": None,
        "verification_notes": None, "escalation_reason": None,
        "retry_count": 0, "retry_history": [], "retrieved_sources": [],
    }


@pytest.mark.parametrize("retry_enabled, expected_attempts", [(True, 3), (False, 1)])
def test_empty_retrieval_skips_answer_model_and_exhausts_applicable_retries(
    monkeypatch, retry_enabled, expected_attempts,
):
    calls = []

    class EmptyStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            calls.append((query, kwargs))
            return []

    def unexpected_answer_chain():
        pytest.fail("answer model must not run without retrieved evidence")

    monkeypatch.setattr(agent_pipeline, "_get_vectorstore", lambda: EmptyStore())
    monkeypatch.setattr(agent_pipeline, "_get_answer_chain", unexpected_answer_chain)
    monkeypatch.setattr(agent_pipeline, "broaden_query", lambda *_args: "broader VPN query")
    monkeypatch.setattr(agent_pipeline.config, "ENABLE_ADAPTIVE_RETRY", retry_enabled)
    monkeypatch.setattr(agent_pipeline.config, "MAX_RETRIEVAL_RETRIES", 2)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    result = agent_pipeline.app_graph.invoke(_initial_graph_state())

    assert len(calls) == expected_attempts
    assert all(kwargs["filter"] == {"category": "VPN"} for _, kwargs in calls)
    assert result["retry_count"] == expected_attempts - 1
    assert result["context"] == ""
    assert result["retrieved_sources"] == []
    assert result["confidence_score"] == 0.0
    assert result["answer"] == agent_pipeline.NO_EVIDENCE_ANSWER
    assert "ServiceNow" not in result["answer"]
    assert result["escalation"] is True
    assert result["tier"] == "Tier 2"


def test_retry_with_new_evidence_resumes_normal_answer_flow(monkeypatch):
    calls = []
    answer_inputs = []
    document = SimpleNamespace(
        page_content=(
            "Ticket ID: TKT-151 | Category: VPN | "
            "Issue: VPN connection fails after reset | "
            "Resolution: Rebuild the VPN profile | Resolved in: 20 minutes"
        ),
        metadata={"ticket_id": "TKT-151", "category": "VPN"},
    )

    class Store:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            calls.append((query, kwargs))
            return [] if len(calls) == 1 else [(document, 0.3)]

    class AnswerChain:
        def invoke(self, payload):
            answer_inputs.append(payload)
            return "Rebuild the VPN profile (Ticket ID: TKT-151)."

    class VerificationModel:
        def invoke(self, _prompt):
            return SimpleNamespace(content="GROUNDED: yes\nREASON: Supported by the ticket.")

    monkeypatch.setattr(agent_pipeline, "_get_vectorstore", lambda: Store())
    monkeypatch.setattr(agent_pipeline, "_get_answer_chain", lambda: AnswerChain())
    monkeypatch.setattr(agent_pipeline, "_get_chat_llm", lambda: VerificationModel())
    monkeypatch.setattr(agent_pipeline, "broaden_query", lambda *_args: "broader VPN query")
    monkeypatch.setattr(agent_pipeline.config, "ENABLE_ADAPTIVE_RETRY", True)
    monkeypatch.setattr(agent_pipeline.config, "MAX_RETRIEVAL_RETRIES", 2)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    result = agent_pipeline.app_graph.invoke(_initial_graph_state())

    assert len(calls) == 2
    assert all(kwargs["filter"] == {"category": "VPN"} for _, kwargs in calls)
    assert calls[1][0] == "broader VPN query"
    assert len(answer_inputs) == 1
    assert "TKT-151" in answer_inputs[0]["context"]
    assert result["retry_count"] == 1
    assert result["answer"] == "Rebuild the VPN profile (Ticket ID: TKT-151)."
    assert result["retrieved_sources"][0]["ticket_id"] == "TKT-151"
    assert result["escalation"] is False
    assert result["tier"] == "Tier 1"


def test_pipeline_passes_category_as_graph_state(monkeypatch):
    captured = {}

    class FakeGraph:
        def invoke(self, state, **_kwargs):
            captured.update(state)
            return {
                "context": "", "answer": "Check the VPN profile.",
                "confidence_score": 0.25, "escalation": False, "tier": "Tier 1",
                "grounded": True, "retrieved_sources": [],
            }

    monkeypatch.setattr(agent_pipeline, "app_graph", FakeGraph())
    monkeypatch.setattr(agent_pipeline, "mask_pii", lambda text: (text, []))
    monkeypatch.setattr(agent_pipeline, "is_safe_query", lambda _text: (True, []))
    monkeypatch.setattr(agent_pipeline, "log_query", lambda **_kwargs: None)
    monkeypatch.setattr(agent_pipeline.config, "ENABLE_LIVE_EVAL", False)

    result = agent_pipeline._run_agent_pipeline_impl("VPN connection", category="VPN")
    assert captured["category"] == "VPN"
    assert result["graph_completed"] is True


@pytest.mark.parametrize("selection, expected", [
    ("VPN", "VPN"),
    ("All Categories", None),
])
def test_chat_passes_selected_category_without_prompt_suffix(
    monkeypatch, selection, expected,
):
    captured = {}
    session = SimpleNamespace(messages=[])
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app.st, "chat_message", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(app.st, "spinner", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(app.st, "markdown", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_user_message", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_response_status", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_retry_badge", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "reformulate_query", lambda text: captured.setdefault("query", text))

    def fake_pipeline(question, category=None):
        captured.update({"pipeline_question": question, "category": category})
        return {
            "answer": "Check the VPN profile.", "escalation": False,
            "tier": "Tier 1", "pii_detected": [], "sources": [],
            "confidence_score": 0.25, "grounded": True,
            "graph_completed": True,
        }

    monkeypatch.setattr(app, "run_agent_pipeline", fake_pipeline)
    app.process_question("VPN connection fails", selection)

    assert captured["category"] == expected
    assert captured["query"] == "VPN connection fails"
    assert captured["pipeline_question"] == "VPN connection fails"
    assert session.messages[0]["content"] == "VPN connection fails"
    assert session.messages[1]["response_kind"] == "rag"
    assert session.messages[1]["graph_completed"] is True
    assert "write_stream" not in inspect.getsource(app.process_question)


def test_chat_reformulates_only_four_recent_messages(monkeypatch):
    captured = {}
    session = SimpleNamespace(messages=[
        {"role": "user", "content": f"Earlier message {index}"}
        for index in range(5)
    ])
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app.st, "chat_message", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(app.st, "spinner", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(app.st, "markdown", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_user_message", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_response_status", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "render_retry_badge", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "reformulate_query", lambda text: captured.setdefault("query", text))
    monkeypatch.setattr(app, "run_agent_pipeline", lambda *_a, **_k: {
        "answer": "Check the VPN profile.", "escalation": False,
        "tier": "Tier 1", "pii_detected": [], "sources": [],
        "confidence_score": 0.25, "grounded": None,
    })

    app.process_question("What about that?", "All Categories")
    assert "Earlier message 0" not in captured["query"]
    for index in range(1, 5):
        assert f"Earlier message {index}" in captured["query"]
