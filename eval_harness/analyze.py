"""
Honest re-analysis of harness groundedness/relevance averages.

Correctly-refused out-of-scope questions score 0 on groundedness and relevance
by construction — there is no context to be grounded in, and refusing is the
right behaviour, so those zeros are a metric artifact rather than a quality
signal. This script reports TWO numbers per dimension:

    1. overall average (every scored case, unchanged)
    2. average excluding CORRECT refusals

Only correct refusals are excluded. A *false* refusal (a question that should
have been answered but was refused) is a genuine failure and is intentionally
kept in both numbers. This is a more truthful way to read the metric — not a
way to inflate it.
"""

import json
import os

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "harness_results.json"
)

DIMENSIONS = ["groundedness", "relevance"]


def _is_correct_refusal(case: dict) -> bool:
    """A correct refusal: the model refused AND that was the expected behaviour
    (i.e. an out-of-scope question). False refusals return False here so they
    stay counted as the real misses they are."""
    rc = case.get("refusal_correctness", {})
    return rc.get("refused") is True and rc.get("correct") is True


def _average(values: list):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def analyze(results_path: str = RESULTS_PATH) -> dict:
    with open(results_path) as f:
        data = json.load(f)
    results = data["detailed_results"]

    correct_refusals = [r for r in results if _is_correct_refusal(r)]
    kept = [r for r in results if not _is_correct_refusal(r)]

    report = {
        "total_cases": len(results),
        "correct_refusals_excluded": len(correct_refusals),
        "correct_refusal_questions": [r["question"] for r in correct_refusals],
        "dimensions": {},
    }

    for dim in DIMENSIONS:
        overall = _average([r[dim].get("score") for r in results])
        excl = _average([r[dim].get("score") for r in kept])
        report["dimensions"][dim] = {
            "overall_average": overall,
            "average_excluding_correct_refusals": excl,
        }

    # Transparency: genuine in-scope weak spots (kept in BOTH numbers). This
    # keeps the script honest — it surfaces real quality issues rather than
    # hiding them behind the exclusion.
    weak_spots = []
    for r in kept:
        g = r["groundedness"].get("score")
        rel = r["relevance"].get("score")
        if (g is not None and g < 0.7) or (rel is not None and rel < 0.7):
            weak_spots.append(
                {"question": r["question"], "groundedness": g, "relevance": rel}
            )
    report["in_scope_weak_spots"] = weak_spots

    return report


if __name__ == "__main__":
    report = analyze()
    print("=" * 66)
    print("HARNESS RE-ANALYSIS — groundedness & relevance")
    print("=" * 66)
    print(f"Total cases: {report['total_cases']}")
    print(
        f"Correct refusals excluded: {report['correct_refusals_excluded']} "
        f"-> {report['correct_refusal_questions']}"
    )
    print("-" * 66)
    for dim, stats in report["dimensions"].items():
        print(f"{dim}:")
        print(f"  overall average (all cases)          : {stats['overall_average']}")
        print(f"  average excluding correct refusals   : {stats['average_excluding_correct_refusals']}")
    print("-" * 66)
    print("Genuine in-scope weak spots (kept in BOTH numbers, not excluded):")
    if report["in_scope_weak_spots"]:
        for w in report["in_scope_weak_spots"]:
            print(f"  g={w['groundedness']} rel={w['relevance']} :: {w['question']}")
    else:
        print("  none")
