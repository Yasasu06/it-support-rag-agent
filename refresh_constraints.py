"""Resolve Python 3.12 Linux transitive pins without installing packages.

Run deliberately after reviewing direct versions in requirements.txt, then
review the generated constraints diff and validate it in CI.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
CONSTRAINTS = ROOT / "constraints-py312.txt"


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def direct_requirements() -> tuple[list[str], set[str]]:
    lines = []
    names = set()
    for raw_line in REQUIREMENTS.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-c "):
            continue
        if "==" in line:
            name = line.split("==", 1)[0]
        elif " @ " in line:
            name = line.split(" @ ", 1)[0]
        else:
            raise ValueError(f"Expected an exact direct requirement: {line}")
        names.add(canonical_name(name))
        lines.append(line)
    return lines, names


def main() -> None:
    lines, direct_names = direct_requirements()
    with tempfile.TemporaryDirectory(prefix="rag-deps-") as temp_dir:
        requirements = Path(temp_dir) / "direct.txt"
        report = Path(temp_dir) / "report.json"
        requirements.write_text("\n".join(lines) + "\n")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--dry-run",
                "--ignore-installed", "--only-binary=:all:",
                "--python-version", "3.12", "--implementation", "cp",
                "--abi", "cp312", "--platform", "manylinux_2_28_x86_64",
                "--platform", "manylinux2014_x86_64",
                "--report", str(report), "-r", str(requirements),
            ],
            check=True,
        )
        resolved = json.loads(report.read_text())

    pins = {
        canonical_name(item["metadata"]["name"]): item["metadata"]["version"]
        for item in resolved["install"]
    }
    if not direct_names.issubset(pins):
        raise ValueError("Resolution report is missing a direct dependency")
    output = [
        "# Generated for Python 3.12 on Linux by refresh_constraints.py.",
        "# Review changes before deployment; do not edit pins by hand.",
    ]
    output.extend(
        f"{name}=={version}"
        for name, version in sorted(pins.items())
        if name not in direct_names
    )
    CONSTRAINTS.write_text("\n".join(output) + "\n")
    print(f"Wrote {len(output) - 2} transitive pins to {CONSTRAINTS.name}")


if __name__ == "__main__":
    main()
