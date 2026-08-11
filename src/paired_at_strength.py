"""Paired comparison at matched strength, with a bootstrap over features.

Both arms inject a perturbation of the same norm at the same `c`, so `c` pairs them by construction.
That makes the natural test a paired one on (feature, strength) cells, which keeps far more power than
compressing two whole fronts into one number at one fluency budget. Both axes are reported, because a
claim of dominance has to hold on each of them separately.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored).dropna(subset=["logppl", args.concept])
    cells = (
        df.groupby(["arm", "feature", "c"])
        .agg(logppl=("logppl", "mean"), concept=(args.concept, "mean"))
        .reset_index()
    )
    ref = cells[cells["arm"] == args.ref].drop(columns="arm")
    rng = np.random.default_rng(args.seed)
    rows = []
    for arm, g in cells.groupby("arm"):
        if arm == args.ref:
            continue
        m = g.drop(columns="arm").merge(ref, on=["feature", "c"], suffixes=("", "_ref"))
        m = m[m["c"] >= args.c_min]
        if m.empty:
            continue
        m["d_logppl"] = m["logppl"] - m["logppl_ref"]
        m["d_concept"] = m["concept"] - m["concept_ref"]
        feats = m["feature"].unique()
        draws = {"d_logppl": [], "d_concept": [], "both": []}
        for _ in range(args.n_boot):
            sel = rng.choice(feats, size=len(feats), replace=True)
            sub = pd.concat([m[m["feature"] == f] for f in sel], ignore_index=True)
            dl, dc = sub["d_logppl"].mean(), sub["d_concept"].mean()
            draws["d_logppl"].append(dl)
            draws["d_concept"].append(dc)
            draws["both"].append(float(dl < 0 and dc > 0))
        rows.append(
            {
                "arm": arm,
                "n_cells": len(m),
                "n_features": len(feats),
                "d_logppl": m["d_logppl"].mean(),
                "logppl_lo95": float(np.quantile(draws["d_logppl"], 0.025)),
                "logppl_hi95": float(np.quantile(draws["d_logppl"], 0.975)),
                "d_concept": m["d_concept"].mean(),
                "concept_lo95": float(np.quantile(draws["d_concept"], 0.025)),
                "concept_hi95": float(np.quantile(draws["d_concept"], 0.975)),
                "p_dominates": float(np.mean(draws["both"])),
            }
        )
    out = pd.DataFrame(rows).sort_values("d_concept", ascending=False)
    out.to_csv(RESULTS / args.out, index=False)
    print(f"paired at matched strength, c >= {args.c_min}, reference = {args.ref}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_test_r2.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--ref", default="naive")
    ap.add_argument("--c-min", dest="c_min", type=float, default=1.0)
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paired_at_strength.csv")
    main(ap.parse_args())
