"""
Compare two harness_results.json snapshots (e.g. before/after a code change)
and report which dimensions improved, regressed, or stayed stable.
"""

import json
import sys


def compare(old_path: str, new_path: str):
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    print(f"Comparing {old['timestamp']} vs {new['timestamp']}")
    print("-" * 60)

    for dim in old["summary"]:
        old_avg = old["summary"][dim].get("average")
        new_avg = new["summary"][dim].get("average")
        if old_avg is None or new_avg is None:
            continue
        delta = new_avg - old_avg
        direction = (
            "IMPROVED" if delta > 0.01
            else "REGRESSED" if delta < -0.01
            else "STABLE"
        )
        print(
            f"{dim}: {old_avg:.3f} -> {new_avg:.3f} "
            f"({direction}, delta={delta:+.3f})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_runs.py <old.json> <new.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
