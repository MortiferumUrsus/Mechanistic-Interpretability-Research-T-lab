"""The round-two arm against the naive one, on both axes at once, in a form the report can cite.

Section 9.1 rests on a joint statement: at the working strengths the correction's nonlinear damage `C` is
not above the naive arm's, while its text is better by a nat or more. Both halves were quoted from separate
tables and derived by hand -- a median of per-feature ratios from `causal_AC.csv` and a paired mean from
`cells.csv` -- so neither could be checked against anything, and both drifted when the underlying artefacts
were regenerated. Writing the derived quantities to their own artefact makes the claim checkable.

`C` is compared by MEDIAN, not mean. It is a ratio whose denominator is the aligned amplitude, and for the
`mts` arm that amplitude passes near zero on one feature, which sends the mean to hundreds. The arms
compared here do not have that pathology, but the median is the honest default for the statistic.

    python dirfix_vs_naive.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RESULTS


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")

    ac = pd.read_csv(RESULTS / "causal_AC.csv")
    cells = pd.read_csv(RESULTS / "cells.csv")
    for name, d, need in (("causal_AC.csv", ac, {"dirfix", "naive"}), ("cells.csv", cells, {"dirfix", "naive"})):
        have = set(d["arm"].unique())
        if not need <= have:
            raise SystemExit(f"{name} lacks {sorted(need - have)}; found {sorted(have)}")

    rows = []
    for c in sorted(ac["c"].unique()):
        s = ac[np.isclose(ac["c"], c)]
        cd = s[s["arm"] == "dirfix"].set_index("feature")["C"]
        cn = s[s["arm"] == "naive"].set_index("feature")["C"]
        j = cd.index.intersection(cn.index)
        row = {
            "c": float(c),
            "C_dirfix_median": float(np.median(cd[j])),
            "C_naive_median": float(np.median(cn[j])),
            "C_ratio_median": float(np.median(cd[j]) / np.median(cn[j])),
            "n_features_AC": int(len(j)),
        }
        # The perplexity half is paired per feature at the same strength, so it answers "is this arm better
        # on the same features" rather than "is this arm's average lower".
        t = cells[np.isclose(cells["c"], c)]
        ld = t[t["arm"] == "dirfix"].set_index("feature")["logppl"]
        ln = t[t["arm"] == "naive"].set_index("feature")["logppl"]
        k = ld.index.intersection(ln.index)
        if len(k):
            diff = (ld[k] - ln[k]).to_numpy()
            row["logppl_paired_mean"] = float(diff.mean())
            row["logppl_paired_median"] = float(np.median(diff))
            row["n_features_ppl"] = int(len(k))
        rows.append(row)

    df = pd.DataFrame(rows)
    dest = RESULTS / "dirfix_vs_naive.csv"
    df.to_csv(dest, index=False)
    print(df.round(4).to_string(index=False))

    work = df[df["c"] >= 1.0]
    print(
        f"\nНа рабочих силах (c >= 1) отношение медиан C = "
        f"{work['C_ratio_median'].min():.2f}…{work['C_ratio_median'].max():.2f}, "
        f"парная разность log-PPL = {work['logppl_paired_mean'].min():+.2f}…{work['logppl_paired_mean'].max():+.2f} ната."
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
