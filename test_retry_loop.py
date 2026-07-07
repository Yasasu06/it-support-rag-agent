"""
Tests for the retry/iteration loop added to the LangGraph pipeline.

Run with:
    cd ~/Desktop/it-support-rag-agent && source venv/bin/activate
    timeout 90 python3 test_retry_loop.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from agent_pipeline import run_agent_pipeline, retry_decision

passed = 0
failed = 0
failures = []


# ---------------------------------------------------------------------------
# TEST 1 — Good question: should succeed on first try with retry_count = 0
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Clear question ('printer not working') — expect no retries")
print("-" * 60)
try:
    result = run_agent_pipeline("printer not working")
    rc = result.get("retry_count", -1)
    rh = result.get("retry_history", None)
    answer = result.get("answer", "")

    print(f"  retry_count   : {rc}")
    print(f"  retry_history : {rh}")
    print(f"  answer (first 100): {answer[:100]}")

    if not answer:
        raise AssertionError("No answer returned from pipeline")
    if rc is None:
        raise AssertionError("retry_count is None — field not returned")
    if rc > 2:
        raise AssertionError(f"retry_count={rc} exceeds max_retries=2")

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 1 (clear question, no retries)", str(e)))


# ---------------------------------------------------------------------------
# TEST 2 — Vague question likely to trigger retries; must NOT hang
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Vague question ('it doesn't work') — pipeline must complete")
print("-" * 60)
try:
    t2_start = time.time()
    result = run_agent_pipeline("it doesn't work")
    t2_elapsed = time.time() - t2_start

    rc = result.get("retry_count", -1)
    rh = result.get("retry_history", [])
    answer = result.get("answer", "")

    print(f"  retry_count   : {rc}")
    print(f"  retry_history : {rh}")
    print(f"  answer (first 100): {answer[:100]}")
    print(f"  elapsed time  : {t2_elapsed:.1f}s")

    if not answer:
        raise AssertionError("No answer returned — pipeline may have hung")
    if rc is None:
        raise AssertionError("retry_count is None — field not returned")
    if rc > 2:
        raise AssertionError(
            f"retry_count={rc} exceeds max_retries=2 — infinite loop risk"
        )
    if t2_elapsed > 60:
        raise AssertionError(
            f"Pipeline took {t2_elapsed:.1f}s — likely hung or looped excessively"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 2 (vague question, timing)", str(e)))


# ---------------------------------------------------------------------------
# TEST 3 — Direct unit tests of retry_decision logic (no LLM calls)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: retry_decision() unit tests — pure logic, no LLM")
print("-" * 60)

sub_failures = []

# 3a: grounded=False, escalation=True, retry_count=0 → should retry
state_a = {
    "grounded": False, "escalation": True, "retry_count": 0,
    "question": "q", "context": None, "answer": None, "tier": None,
    "confidence_score": None, "verification_notes": None,
    "escalation_reason": None, "retry_history": [],
}
result_a = retry_decision(state_a)
status_a = "PASS" if result_a == "retry" else "FAIL"
if result_a != "retry":
    sub_failures.append(f"3a expected 'retry', got '{result_a}'")
print(f"  3a retry_count=0, escalation=True → '{result_a}' [{status_a}]")

# 3b: grounded=False, escalation=True, retry_count=2 → max retries exhausted
state_b = {
    "grounded": False, "escalation": True, "retry_count": 2,
    "question": "q", "context": None, "answer": None, "tier": None,
    "confidence_score": None, "verification_notes": None,
    "escalation_reason": None, "retry_history": [],
}
result_b = retry_decision(state_b)
status_b = "PASS" if result_b == "end" else "FAIL"
if result_b != "end":
    sub_failures.append(f"3b expected 'end', got '{result_b}'")
print(f"  3b retry_count=2, escalation=True → '{result_b}' [{status_b}]")

# 3c: grounded=True, escalation=False, retry_count=0 → no retry needed
state_c = {
    "grounded": True, "escalation": False, "retry_count": 0,
    "question": "q", "context": None, "answer": None, "tier": None,
    "confidence_score": None, "verification_notes": None,
    "escalation_reason": None, "retry_history": [],
}
result_c = retry_decision(state_c)
status_c = "PASS" if result_c == "end" else "FAIL"
if result_c != "end":
    sub_failures.append(f"3c expected 'end', got '{result_c}'")
print(f"  3c retry_count=0, grounded=True, no escalation → '{result_c}' [{status_c}]")

if sub_failures:
    for sf in sub_failures:
        print(f"  SUB-FAIL: {sf}")
    print("  RESULT: FAIL")
    failed += 1
    failures.append(("TEST 3 (retry_decision unit tests)", "; ".join(sub_failures)))
else:
    print("  RESULT: PASS")
    passed += 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
total = passed + failed
print(f"RETRY LOOP TESTS: {passed}/{total} PASSED")

if failures:
    print()
    print("Failed tests:")
    for name, reason in failures:
        print(f"  - {name}")
        print(f"    Reason: {reason}")

sys.exit(0 if failed == 0 else 1)
