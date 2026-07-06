"""
Real-time continuous evaluation for the IT Support RAG Agent.

Scores every production query automatically as it flows through the pipeline.
Writes one JSONL entry per query to live_eval_log.jsonl.
Exposes get_live_eval_summary() for the System tab in app.py.
"""

import json
import os
from datetime import datetime
from typing import Optional

LIVE_EVAL_LOG = os.path.join(
    os.path.dirname(__file__),
    "live_eval_log.jsonl"
)


def score_response_relevance(
    question: str,
    answer: str,
    llm_getter,
) -> float:
    """
    Uses GPT-4o-mini to score how relevant the answer is to the question
    on a 0-1 scale. Fail-safe: returns 0.0 on any error so a scoring
    failure never silently counts as a pass.
    """
    try:
        llm = llm_getter()
        prompt = (
            "Rate how well this ANSWER addresses "
            "the QUESTION on a scale of 0 to 10.\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            "Respond with ONLY a single number "
            "from 0 to 10, nothing else."
        )
        response = llm.invoke(prompt)
        score_text = response.content.strip()
        score_num = float(
            "".join(c for c in score_text if c.isdigit() or c == ".")
        )
        return max(0.0, min(score_num / 10.0, 1.0))
    except Exception:
        return 0.0


def log_live_eval(
    question: str,
    answer: str,
    grounded: Optional[bool],
    verification_notes: str,
    confidence_score: float,
    relevance_score: float,
    escalation: bool,
    tier: str,
    latency_seconds: float,
) -> None:
    """Appends one scored evaluation entry per production query to live_eval_log.jsonl."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question_length": len(question),
        "answer_length": len(answer),
        "grounded": grounded,
        "verification_notes": verification_notes,
        "confidence_score": round(confidence_score, 3) if confidence_score else None,
        "relevance_score": round(relevance_score, 3),
        "escalation": escalation,
        "tier": tier,
        "latency_seconds": round(latency_seconds, 2),
        "passed_quality_bar": (
            grounded is True and relevance_score >= 0.6
        ),
    }

    with open(LIVE_EVAL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_live_eval_summary() -> dict:
    """
    Reads live_eval_log.jsonl and returns aggregate metrics for the System tab.
    Returns safe zero-defaults if the log doesn't exist or is empty.
    """
    default = {
        "total_scored": 0,
        "avg_relevance": 0.0,
        "grounded_rate": 0.0,
        "quality_pass_rate": 0.0,
        "avg_latency": 0.0,
    }

    if not os.path.exists(LIVE_EVAL_LOG):
        return default

    entries = []
    try:
        with open(LIVE_EVAL_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        return default

    if not entries:
        return default

    total = len(entries)
    avg_relevance = sum(e.get("relevance_score", 0) for e in entries) / total
    grounded_count = sum(1 for e in entries if e.get("grounded") is True)
    quality_pass_count = sum(1 for e in entries if e.get("passed_quality_bar") is True)
    avg_latency = sum(e.get("latency_seconds", 0) for e in entries) / total

    return {
        "total_scored": total,
        "avg_relevance": round(avg_relevance, 3),
        "grounded_rate": round(grounded_count / total, 3),
        "quality_pass_rate": round(quality_pass_count / total, 3),
        "avg_latency": round(avg_latency, 2),
    }
