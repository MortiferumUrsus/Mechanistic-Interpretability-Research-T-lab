"""Paired (arm - reference) differences per strength, not pooled over the grid.

paired_at_strength.py pools every (feature, strength) cell with c >= c_min into one number. That is the
right summary for an arm whose effect has one sign across the grid, and the wrong one for an arm whose
text collapses at the top of the grid: the shared direction injected alone (`shared_only`) lowers
generation log-PPL by several nats at c = 1..2 and raises it at c = 5, and the pooled mean of that curve
can coincide with a completely different curve by accident. This script writes the per-strength means
(over features, each feature first averaged over its prompts) so the report can quote the curve.

    python paired_by_strength.py --scored scored_expA_r3.csv --out paired_expA_r3_by_strength.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import RESULTS


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored).dropna(subset=["logppl", args.concept])
    cells = (
        df.groupby(["arm", "feature", "c"])
        .agg(logppl=("logppl", "mean"), concept=(args.concept, "mean"), rep4=("rep4", "mean"))
        .reset_index()
    )
    ref = cells[cells["arm"] == args.ref].set_index(["feature", "c"])
    rows = []
    for arm, g in cells.groupby("arm"):
        if arm == args.ref:
            continue
        g = g.set_index(["feature", "c"])
        common = g.index.intersection(ref.index)
        d = g.loc[common, ["logppl", "concept", "rep4"]] - ref.loc[common, ["logppl", "concept", "rep4"]]
        d = d.reset_index()
        for c, gc in d.groupby("c"):
            rows.append(
                {
                    "arm": arm,
                    "c": float(c),
                    "n_features": int(gc["feature"].nunique()),
                    "d_logppl": float(gc["logppl"].mean()),
                    "d_concept": float(gc["concept"].mean()),
                    "d_rep4": float(gc["rep4"].mean()),
                    "arm_logppl": float(g.loc[[i for i in common if i[1] == c], "logppl"].mean()),
                    "ref_logppl": float(ref.loc[[i for i in common if i[1] == c], "logppl"].mean()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["arm", "c"])
    out.to_csv(RESULTS / args.out, index=False)
    print(out.round(3).to_string(index=False))
    print(f"wrote results/{args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_expA_r3.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--ref", default="naive")
    ap.add_argument("--out", default="paired_expA_r3_by_strength.csv")
    main(ap.parse_args())
