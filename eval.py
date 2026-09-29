"""
Evaluation script for the IT Support RAG Agent.

Runs a fixed set of 20 test questions against rag.get_answer, checks
whether the agent answered or refused, compares that against the
expected behavior for each question, and reports a behavioral pass rate.

Run with:
    python3 eval.py
"""

import json
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from rag import get_answer
from security import sanitize_external_text

REFUSAL_MARKERS = ["don't have information", "enterprise knowledge base"]

TEST_CASES = [
    {
        "question": "My printer is not printing anything",
        "expected_category": "Printer",
        "should_answer": True,
    },
    {
        "question": "I cannot access SAP after my account was locked",
        "expected_category": "ERP Access",
        "should_answer": True,
    },
    {
        "question": "Outlook keeps asking me to sign in repeatedly",
        "expected_category": "Email",
        "should_answer": True,
    },
    {
        "question": "My laptop screen went black and won't turn on",
        "expected_category": "Hardware",
        "should_answer": True,
    },
    {
        "question": "I need access to Power BI for my team",
        "expected_category": "Software Access",
        "should_answer": True,
    },
    {
        "question": "Network drive is not showing up on my computer",
        "expected_category": "Network",
        "should_answer": True,
    },
    {
        "question": "I forgot my Windows password and am locked out",
        "expected_category": "Password",
        "should_answer": True,
    },
    {
        "question": "VPN disconnects every few minutes while working from home",
        "expected_category": "VPN",
        "should_answer": True,
    },
    {
        "question": "Printer shows offline but is turned on",
        "expected_category": "Printer",
        "should_answer": True,
    },
    {
        "question": "I need SAP access for the new project I joined",
        "expected_category": "ERP Access",
        "should_answer": True,
    },
    {
        "question": "My email signature is not showing in Outlook",
        "expected_category": "Email",
        "should_answer": True,
    },
    {
        "question": "Computer is running very slowly after Windows update",
        "expected_category": "Hardware",
        "should_answer": True,
    },
    {
        "question": "Cannot install Adobe Acrobat on my work laptop",
        "expected_category": "Software Access",
        "should_answer": True,
    },
    {
        "question": "WiFi keeps dropping in the conference room",
        "expected_category": "Network",
        "should_answer": True,
    },
    {
        "question": "Account locked after too many wrong password attempts",
        "expected_category": "Password",
        "should_answer": True,
    },
    {
        "question": "AnyConnect shows login failed after password reset",
        "expected_category": "VPN",
        "should_answer": True,
    },
    {
        "question": "What is the weather in Grand Rapids today?",
        "expected_category": "Out of scope",
        "should_answer": False,
    },
    {
        "question": "Who won the Super Bowl last year?",
        "expected_category": "Out of scope",
        "should_answer": False,
    },
    {
        "question": "Can you write me a poem?",
        "expected_category": "Out of scope",
        "should_answer": False,
    },
    {
        "question": "What is the capital of France?",
        "expected_category": "Out of scope",
        "should_answer": False,
    },
]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def behavior_summary_for(match_rate: float) -> str:
    if match_rate >= 90:
        return "At least 90% matched on this test set"
    if match_rate >= 70:
        return "70–89% matched on this test set"
    return "Below 70% matched on this test set"


def run_eval() -> dict:
    results = []
    for case in TEST_CASES:
        safe_question = sanitize_external_text(case["question"])
        answer = sanitize_external_text(get_answer(safe_question))
        refused = is_refusal(answer)
        actually_answered = not refused

        if case["should_answer"] and refused:
            passed = False
        elif not case["should_answer"] and actually_answered:
            passed = False
        else:
            passed = True

        results.append(
            {
                "question": safe_question,
                "expected_category": case["expected_category"],
                "should_answer": case["should_answer"],
                "answer": answer,
                "expected_behavior": "Answer" if case["should_answer"] else "Refuse",
                "actual_behavior": "Answered" if actually_answered else "Refused",
                "passed": passed,
            }
        )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    behavior_match_rate = round((passed_count / total) * 100, 1) if total else 0.0

    in_scope = [r for r in results if r["should_answer"]]
    out_of_scope = [r for r in results if not r["should_answer"]]
    in_scope_answered_as_expected = sum(1 for r in in_scope if r["passed"])
    out_of_scope_refused_as_expected = sum(1 for r in out_of_scope if r["passed"])

    behavior_summary = behavior_summary_for(behavior_match_rate)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "passed": passed_count,
        "failed": failed_count,
        "behavior_match_rate": behavior_match_rate,
        "in_scope_answered_as_expected": in_scope_answered_as_expected,
        "in_scope_total": len(in_scope),
        "out_of_scope_refused_as_expected": out_of_scope_refused_as_expected,
        "out_of_scope_total": len(out_of_scope),
        "behavior_summary": behavior_summary,
        "results": results,
    }


def print_report(summary: dict) -> None:
    print("=" * 70)
    print("IT SUPPORT RAG AGENT — EVALUATION REPORT")
    print("=" * 70)

    for i, r in enumerate(summary["results"], start=1):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{i}] {status} — {r['question']}")
        print(f"    Expected: {r['expected_behavior']} ({r['expected_category']})")
        print(f"    Actual:   {r['actual_behavior']}")
        print(f"    Answer:   {r['answer'][:150]}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Expected answer/refusal behavior: {summary['behavior_match_rate']}%")
    print(
        f"In-scope questions answered as expected: "
        f"{summary['in_scope_answered_as_expected']}/{summary['in_scope_total']}"
    )
    print(
        f"Out-of-scope questions refused as expected: "
        f"{summary['out_of_scope_refused_as_expected']}/{summary['out_of_scope_total']}"
    )
    print(f"Benchmark summary: {summary['behavior_summary']}")
    print("=" * 70)


def main() -> None:
    summary = run_eval()
    print_report(summary)

    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()
