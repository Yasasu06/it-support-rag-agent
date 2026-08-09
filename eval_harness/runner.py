"""
Multi-dimensional evaluation harness runner.

Runs every case in the existing 20-question test set through the full
multi-agent pipeline and scores each on 5 independent dimensions. Structured,
repeatable, and multi-dimensional — the difference between a real eval harness
and a single pass/fail script.
"""

import json
import time
import os
import sys
from datetime import datetime

sys.path.insert(0,
    os.path.dirname(os.path.dirname(__file__)))

from eval_harness.metrics import (
    score_groundedness,
    score_relevance,
    score_citation_accuracy,
    score_refusal_correctness,
    score_latency
)

# Reuse the existing 20-question test set - do not
# duplicate it, import it
from eval import TEST_CASES


def run_full_harness() -> dict:
    """
    Runs every test case through the full multi-agent
    pipeline and scores it on all 5 independent
    dimensions. This is the "harness" - structured,
    repeatable, multi-dimensional evaluation.
    """
    from agent_pipeline import run_agent_pipeline, _get_chat_llm

    results = []

    for case in TEST_CASES:
        question = case["question"]
        expected_should_answer = case["should_answer"]

        start = time.time()
        pipeline_result = run_agent_pipeline(question)
        latency = time.time() - start

        answer = pipeline_result.get("answer", "")
        # NOTE: run_agent_pipeline exposes retrieved tickets under the key
        # "sources" in its return dict (it renames the internal
        # "retrieved_sources" state field). Confirmed against
        # agent_pipeline.py — this is the correct key.
        sources = pipeline_result.get(
            "sources", []
        )
        source_ids = [
            s.get("ticket_id", "") for s in sources
        ] if sources else []
        context = " ".join([
            f"{s.get('issue', '')} {s.get('resolution', '')}"
            for s in sources
        ]) if sources else ""

        scores = {
            "question": question,
            "answer": answer[:200],
            "groundedness": score_groundedness(
                answer, context, _get_chat_llm
            ) if context else {
                "dimension": "groundedness",
                "score": None,
                "note": "no context retrieved"
            },
            "relevance": score_relevance(
                question, answer, _get_chat_llm
            ),
            "citation_accuracy": score_citation_accuracy(
                answer, source_ids
            ),
            "refusal_correctness":
                score_refusal_correctness(
                    question, answer,
                    expected_should_answer
                ),
            "latency": score_latency(latency)
        }

        results.append(scores)
        print(f"Scored: {question[:50]}...")

    summary = _compute_summary(results)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_cases": len(results),
        "summary": summary,
        "detailed_results": results
    }

    output_path = os.path.join(
        os.path.dirname(__file__),
        "harness_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def _compute_summary(results: list) -> dict:
    """Aggregates per-dimension averages across all
    test cases, handling None scores (like 'no context
    retrieved') gracefully."""
    dimensions = [
        "groundedness", "relevance",
        "citation_accuracy", "refusal_correctness",
        "latency"
    ]

    summary = {}
    for dim in dimensions:
        scores = [
            r[dim]["score"] for r in results
            if r[dim].get("score") is not None
        ]
        if scores:
            summary[dim] = {
                "average": round(
                    sum(scores) / len(scores), 3
                ),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "count_scored": len(scores)
            }
        else:
            summary[dim] = {
                "average": None,
                "note": "no valid scores in this run"
            }

    return summary


if __name__ == "__main__":
    print("Running full evaluation harness...")
    print("This will take a few minutes - each question requires multiple LLM calls for scoring.")
    result = run_full_harness()
    print("\n" + "=" * 60)
    print("HARNESS SUMMARY")
    print("=" * 60)
    for dim, stats in result["summary"].items():
        print(f"{dim}: {stats}")
