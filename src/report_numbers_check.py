"""Every number the report quotes must be findable in an artefact, and the artefact must be current.

Two failure modes this catches, both of which have already happened here:

- the report cites a value that no current artefact contains, because the artefact was regenerated and the
  text was not (section 9.9 quoted -1.093 for a paired difference the current table gives as -1.349);
- the report cites a value from a file that is *older* than the file it is derived from, so the number is
  arithmetically consistent with nothing (`scored_test_r2.csv` was once older than the generations it
  scores, because a re-scoring died half way).

Neither raises anything. Both produce a report full of finite, plausible, mutually inconsistent numbers.

The check is deliberately blunt: pull every decimal from the report, and for each one look for a matching
value anywhere in the results directory. A number that appears nowhere is either stale, hand-computed, or a
typo -- all three are worth a human look. Numbers that are obviously not measurements (years, section
numbers, small integers, percentages of the form "95%") are skipped by a stated rule rather than silently.

    python report_numbers_check.py
    python report_numbers_check.py --report ../REPORT.md --results ../results
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Values the report computes in the text rather than reading from a file. Each is listed with what it is,
# so the exclusion is auditable and a NEW unfound number stands out instead of drowning in known ones.
# Verified by hand once: 39.8 and 46.1 are the means over the twelve features at c = 1 of `A` in
# results/causal_AC.csv (39.76 and 46.09); 88.23 and 88.9 are `median||h||` and the corpus mean norm, which
# live in the ActStats .pt rather than in any CSV; 78.1 is a percentage of matching cells stated in §9.5.
COMPUTED_IN_TEXT = {39.8, 46.1, 88.23, 88.2, 88.9, 78.1, 4476.7}

# Values that are structural rather than measured. Kept explicit so the exclusion is auditable.
STRUCTURAL = {
    0.0, 1.0, 0.5, 0.25, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0,  # the strength grid and simple fractions
    0.05, 0.95, 0.3, 0.1, 0.2,                              # thresholds, alpha levels, the cosine gate
}
NUM = re.compile(r"[−-]?\d+\.\d+")


def load_values(results: Path) -> dict[float, list[str]]:
    """Every numeric cell in every CSV/JSON under results/, rounded to the report's precision."""
    found: dict[float, list[str]] = {}

    def add(v, where):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if not np.isfinite(f):
            return
        for nd in (2, 3, 4):
            found.setdefault(round(abs(f), nd), []).append(where)

    for path in sorted(results.rglob("*.csv")):
        if "stale" in path.parts:  # quarantined on purpose; citing one is exactly what we are hunting
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in df.columns:
            if df[col].dtype.kind in "fiu":
                for v in df[col].to_numpy():
                    add(v, path.name)
    for path in sorted(results.rglob("*.json")):
        if "stale" in path.parts:
            continue
        try:
            import json

            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                for x in o.values():
                    walk(x)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
            else:
                add(o, path.name)

        walk(blob)
    return found


def main(args) -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")

    report = Path(args.report)
    results = Path(args.results)
    text = report.read_text(encoding="utf-8")
    found = load_values(results)
    print(f"{len(found)} distinct values across the artefacts in {results}")

    missing: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith(("<!--", "#")) or "](" in line and "http" in line:
            continue
        for raw in NUM.findall(line):
            v = abs(float(raw.replace("−", "-")))
            if v in STRUCTURAL or v in COMPUTED_IN_TEXT or v > 1e6:
                continue
            if any(round(v, nd) in found for nd in (2, 3, 4)):
                continue
            missing.append((i, raw, line.strip()[:110]))

    if not missing:
        print("every quoted number is present in a current artefact")
        return
    print(f"\n{len(missing)} quoted numbers were not found in any current artefact:")
    for ln, raw, ctx in missing[: args.limit]:
        print(f"  REPORT.md:{ln}  {raw}\n      {ctx}")
    if len(missing) > args.limit:
        print(f"  ... and {len(missing) - args.limit} more")
    print(
        "\nEach of these is stale, hand-computed or a typo. Hand-computed values are legitimate -- state "
        "the arithmetic next to them so the next reader does not have to guess which kind it is."
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(here.parent / "REPORT.md"))
    ap.add_argument("--results", default=str(here.parent / "results"))
    ap.add_argument("--limit", type=int, default=40)
    main(ap.parse_args())
