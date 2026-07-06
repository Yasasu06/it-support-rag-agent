"""
Tests for the Verification Agent added to the LangGraph pipeline.

Run with:
    cd ~/Desktop/it-support-rag-agent && source venv/bin/activate
    python3 test_verification_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from agent_pipeline import verification_agent, run_agent_pipeline

passed = 0
failed = 0
failures = []


# ---------------------------------------------------------------------------
# TEST 1 — Grounded answer via the full pipeline (should not crash, grounded
#           must be bool, and if grounded=True tier must not be Tier 2 purely
#           from the verification path).
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Full pipeline with real question ('printer not working')")
print("-" * 60)
try:
    result = run_agent_pipeline("printer not working")
    grounded_val = result.get("grounded")
    tier = result.get("tier")
    notes = result.get("verification_notes", "")

    print(f"  grounded         : {grounded_val}")
    print(f"  tier             : {tier}")
    print(f"  verification_notes: {notes}")
    print(f"  answer (first 120): {result.get('answer', '')[:120]}")

    if grounded_val is None:
        raise AssertionError("grounded is None — verification_agent did not set it")

    if not isinstance(grounded_val, bool):
        raise AssertionError(f"grounded must be bool, got {type(grounded_val)}")

    # If grounded=True, the escalation must NOT be driven by the verification
    # path — we confirm by checking verification_notes doesn't contain
    # the forced-escalation marker.
    if grounded_val is True and tier == "Tier 2":
        if notes and "failed groundedness" in str(result.get("escalation_reason", "")):
            raise AssertionError(
                "Tier 2 was triggered by verification path despite grounded=True"
            )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 1 (full pipeline grounded answer)", str(e)))


# ---------------------------------------------------------------------------
# TEST 2 — Fabricated answer: verification_agent must detect it as ungrounded.
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Fabricated answer — must be detected as grounded=False")
print("-" * 60)
try:
    fake_state = {
        "question": "printer not working",
        "context": (
            "Ticket ID: TKT-001 | Category: Printer "
            "| Issue: Printer offline "
            "| Resolution: Restarted print spooler service "
            "| Resolved in: 20 minutes"
        ),
        "answer": (
            "Based on Ticket TKT-001, you should replace the printer's "
            "motherboard and reinstall Windows completely."
        ),
        "escalation": None,
        "tier": None,
        "confidence_score": 0.25,
        "grounded": None,
        "verification_notes": None,
        "escalation_reason": None,
    }

    result_state = verification_agent(fake_state)
    grounded_val = result_state.get("grounded")
    notes = result_state.get("verification_notes", "")

    print(f"  grounded         : {grounded_val}")
    print(f"  verification_notes: {notes}")

    if grounded_val is not False:
        raise AssertionError(
            f"Expected grounded=False for fabricated answer, got {grounded_val!r}. "
            f"Notes: {notes}"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 2 (fabricated answer detection)", str(e)))


# ---------------------------------------------------------------------------
# TEST 3 — Missing context edge case: must return grounded=False without crash.
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: Empty context — must return grounded=False without crashing")
print("-" * 60)
try:
    edge_state = {
        "question": "printer not working",
        "context": "",
        "answer": "Restart the print spooler service.",
        "escalation": None,
        "tier": None,
        "confidence_score": 0.25,
        "grounded": None,
        "verification_notes": None,
        "escalation_reason": None,
    }

    result_state = verification_agent(edge_state)
    grounded_val = result_state.get("grounded")
    notes = result_state.get("verification_notes", "")

    print(f"  grounded         : {grounded_val}")
    print(f"  verification_notes: {notes}")

    if grounded_val is not False:
        raise AssertionError(
            f"Expected grounded=False for empty context, got {grounded_val!r}"
        )

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 3 (missing context edge case)", str(e)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
total = passed + failed
print(f"VERIFICATION AGENT TESTS: {passed}/{total} PASSED")

if failures:
    print()
    print("Failed tests:")
    for name, reason in failures:
        print(f"  - {name}")
        print(f"    Reason: {reason}")

sys.exit(0 if failed == 0 else 1)
