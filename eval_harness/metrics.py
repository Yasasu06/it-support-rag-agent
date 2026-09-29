"""
Independent scoring dimensions for the evaluation harness.

Each function scores ONE quality in isolation so a run reveals *where* the
system is weak, not just an overall pass/fail. Two dimensions use an LLM judge
(groundedness, relevance); three are pure logic (citation-ID validity, refusal
correctness, latency) and cost nothing.
"""

import re
from typing import Optional
from security import sanitize_external_text
from error_handling import safe_error_type


def score_groundedness(
    answer: str,
    context: str,
    llm_getter
) -> dict:
    """
    Asks an LLM to rate how well the answer is
    supported by the context. Returns a 0-1 judge
    score and its response.
    This reuses the same pattern as the existing
    Verification Agent but as a standalone, reusable
    scoring function.
    """
    try:
        answer = sanitize_external_text(answer)
        context = sanitize_external_text(context)
        llm = llm_getter()
        prompt = (
            "Score how well-grounded this ANSWER is in "
            "the CONTEXT, from 0 to 10. 10 means every "
            "claim is directly supported. 0 means the "
            "answer is entirely fabricated.\n\n"
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Respond with ONLY a number 0-10."
        )
        response = llm.invoke(prompt)
        score = float(
            ''.join(c for c in response.content
                   if c.isdigit() or c == '.')
        )
        return {
            "dimension": "groundedness",
            "score": max(0.0, min(score / 10.0, 1.0)),
            "raw_response": sanitize_external_text(response.content.strip())
        }
    except Exception as e:
        return {
            "dimension": "groundedness",
            "score": 0.0,
            "error": safe_error_type(e)
        }


def score_relevance(
    question: str,
    answer: str,
    llm_getter
) -> dict:
    """
    Asks an LLM to rate whether the answer addresses the
    question, separately from grounding.
    """
    try:
        question = sanitize_external_text(question)
        answer = sanitize_external_text(answer)
        llm = llm_getter()
        prompt = (
            "Score how well this ANSWER addresses the "
            "specific QUESTION asked, from 0 to 10. "
            "10 means it directly and completely "
            "answers what was asked. 0 means it's "
            "completely off-topic.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
            "Respond with ONLY a number 0-10."
        )
        response = llm.invoke(prompt)
        score = float(
            ''.join(c for c in response.content
                   if c.isdigit() or c == '.')
        )
        return {
            "dimension": "relevance",
            "score": max(0.0, min(score / 10.0, 1.0)),
            "raw_response": sanitize_external_text(response.content.strip())
        }
    except Exception as e:
        return {
            "dimension": "relevance",
            "score": 0.0,
            "error": safe_error_type(e)
        }


def score_citation_id_validity(
    answer: str,
    source_ticket_ids: list
) -> dict:
    """
    Pure logic check (no LLM call): extract recognizable Ticket IDs from the
    answer and compare them with retrieved source IDs. A valid ID match does
    not establish that the cited source supports the surrounding claim.
    """
    cited_ids = re.findall(
        r'\b(?:TKT|KGL|GH|HF|SNOW)-[\w\-]+\b',
        answer
    )

    if not cited_ids:
        return {
            "dimension": "citation_id_validity",
            "score": None,
            "note": "No ticket IDs cited in answer"
        }

    valid_citations = sum(
        1 for cid in cited_ids
        if cid in source_ticket_ids
    )
    validity = valid_citations / len(cited_ids)

    return {
        "dimension": "citation_id_validity",
        "score": validity,
        "cited_ids": cited_ids,
        "source_ids": source_ticket_ids,
        "invalid_citations": [
            cid for cid in cited_ids
            if cid not in source_ticket_ids
        ]
    }


def score_refusal_correctness(
    question: str,
    answer: str,
    expected_should_answer: bool
) -> dict:
    """
    Pure logic check (no LLM call): verifies refusal
    behavior matches expectation. Reuses the same
    detection pattern as the existing eval.py.
    """
    refused = (
        "don't have information" in answer.lower() or
        "enterprise knowledge base" in answer.lower()
    )
    answered = not refused

    correct = (answered == expected_should_answer)

    return {
        "dimension": "refusal_correctness",
        "score": 1.0 if correct else 0.0,
        "refused": refused,
        "expected_should_answer": expected_should_answer,
        "correct": correct
    }


def score_latency(latency_seconds: float) -> dict:
    """
    Pure logic - scores response time against a
    reasonable threshold for a chat interface.
    """
    if latency_seconds <= 3.0:
        score = 1.0
    elif latency_seconds <= 8.0:
        score = 0.7
    elif latency_seconds <= 15.0:
        score = 0.4
    else:
        score = 0.1

    return {
        "dimension": "latency",
        "score": score,
        "latency_seconds": latency_seconds
    }
