"""
Tests for the live evaluation module (live_eval.py).

Run with:
    cd ~/Desktop/it-support-rag-agent && source venv/bin/activate
    python3 test_live_eval.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

# Use a temporary log file so tests never touch the real one.
import live_eval as _le
_tmp_log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
_tmp_log.close()
_le.LIVE_EVAL_LOG = _tmp_log.name

from live_eval import (
    score_response_relevance,
    log_live_eval,
    get_live_eval_summary,
)
from agent_pipeline import _get_chat_llm

passed = 0
failed = 0
failures = []


# ---------------------------------------------------------------------------
# TEST 1 — Good answer: relevance score should be >= 0.5
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Relevance score for a clearly on-topic answer")
print("-" * 60)
try:
    question = "printer not working"
    answer = (
        "Based on Ticket TKT-036, restart the print spooler service "
        "and check the DHCP address assignment for the printer."
    )
    score = score_response_relevance(question, answer, _get_chat_llm)
    print(f"  relevance_score: {score:.3f}")

    if not isinstance(score, float):
        raise AssertionError(f"Expected float, got {type(score)}")
    if score < 0.5:
        raise AssertionError(
            f"Expected score >= 0.5 for a clearly relevant answer, got {score:.3f}"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 1 (good answer relevance)", str(e)))


# ---------------------------------------------------------------------------
# TEST 2 — Irrelevant answer: relevance score should be < 0.5
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Relevance score for a clearly off-topic answer")
print("-" * 60)
try:
    question = "printer not working"
    answer = "The capital of France is Paris and the weather today is sunny."
    score = score_response_relevance(question, answer, _get_chat_llm)
    print(f"  relevance_score: {score:.3f}")

    if not isinstance(score, float):
        raise AssertionError(f"Expected float, got {type(score)}")
    if score >= 0.5:
        raise AssertionError(
            f"Expected score < 0.5 for an irrelevant answer, got {score:.3f}"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 2 (irrelevant answer relevance)", str(e)))


# ---------------------------------------------------------------------------
# TEST 3 — Full log + summary cycle
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: log_live_eval() + get_live_eval_summary() round-trip")
print("-" * 60)
try:
    log_live_eval(
        question="printer not working",
        answer="Restart the print spooler (TKT-036).",
        grounded=True,
        verification_notes="All claims supported by context.",
        confidence_score=0.28,
        relevance_score=0.9,
        escalation=False,
        tier="Tier 1",
        latency_seconds=3.2,
    )
    log_live_eval(
        question="VPN disconnects randomly",
        answer="Reset credentials per TKT-012.",
        grounded=True,
        verification_notes="Fully grounded.",
        confidence_score=0.25,
        relevance_score=0.8,
        escalation=False,
        tier="Tier 1",
        latency_seconds=2.8,
    )
    log_live_eval(
        question="fix my car engine",
        answer="Replace the alternator.",
        grounded=False,
        verification_notes="Answer not grounded in ticket context.",
        confidence_score=0.10,
        relevance_score=0.1,
        escalation=True,
        tier="Tier 2",
        latency_seconds=4.1,
    )

    summary = get_live_eval_summary()
    print(f"  summary: {summary}")

    if summary["total_scored"] != 3:
        raise AssertionError(
            f"Expected total_scored=3, got {summary['total_scored']}"
        )
    for key in ("avg_relevance", "grounded_rate", "quality_pass_rate", "avg_latency"):
        if summary[key] is None:
            raise AssertionError(f"summary['{key}'] is None — must be a float")

    expected_grounded_rate = round(2 / 3, 3)
    if abs(summary["grounded_rate"] - expected_grounded_rate) > 0.01:
        raise AssertionError(
            f"Expected grounded_rate≈{expected_grounded_rate}, "
            f"got {summary['grounded_rate']}"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 3 (log + summary round-trip)", str(e)))
finally:
    # Clean up temp file
    try:
        os.unlink(_tmp_log.name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
total = passed + failed
print(f"LIVE EVAL TESTS: {passed}/{total} PASSED")

if failures:
    print()
    print("Failed tests:")
    for name, reason in failures:
        print(f"  - {name}")
        print(f"    Reason: {reason}")

sys.exit(0 if failed == 0 else 1)
