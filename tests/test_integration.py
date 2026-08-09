"""
Slow end-to-end integration tests — real pipeline, real OpenAI calls.

These are marked `integration` (so `-m "not integration"` skips them) AND
guarded by `requires_api_key` (so they auto-skip when OPENAI_API_KEY is absent,
e.g. in CI without secrets configured). They verify genuine behavior — no mocks.
"""

import os

import pytest

requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY - skipped in CI without secrets configured",
)


@pytest.mark.integration
@requires_api_key
def test_rag_pipeline_returns_grounded_answer():
    from rag import get_answer

    answer = get_answer("printer not working")
    assert answer
    assert len(answer) > 20


@pytest.mark.integration
@requires_api_key
def test_multi_agent_pipeline_end_to_end():
    from agent_pipeline import run_agent_pipeline

    result = run_agent_pipeline("printer not working")
    assert result["answer"]
    assert result.get("grounded") is not None


@pytest.mark.integration
@requires_api_key
def test_out_of_scope_question_refuses():
    from rag import get_answer

    answer = get_answer("what is the weather today")
    assert "don't have information" in answer.lower()
