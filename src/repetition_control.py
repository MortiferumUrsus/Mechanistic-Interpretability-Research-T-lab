"""Is the direction correction's perplexity advantage just degeneration into repetition?

The objection is specific and it is a good one. Repetitive text is easy to predict, so a model that starts
looping scores a low perplexity while producing worse text. The corrected direction does repeat more than
naive steering at high strength -- 4-gram repetition reaches 24% against 13% at c = 3 -- so a perplexity win
measured across all strengths could be that artefact rather than better text.

The test is to compare the two arms at MATCHED repetition rather than at matched strength. If the advantage
is degeneration, it must vanish once the two arms repeat equally; if it survives at zero repetition, it is
not degeneration.

    python repetition_control.py --scored scored_r3.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import RESULTS

# Bin edges on 4-gram repetition. The first bin is the one that settles the question: it holds only cells
# with no repeated 4-gram at all, so the two arms cannot differ on that axis inside it.
BINS = [-1e-9, 1e-9, 0.05, 0.15, 1.01]
LABELS = ["ровно 0", "0–5%", "5–15%", ">15%"]


def main(args) -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")

    d = pd.read_csv(RESULTS / args.scored)
    missing = {"arm", "c", "logppl", "rep4", "keyword_hit"} - set(d.columns)
    if missing:
        raise SystemExit(f"{args.scored} lacks {sorted(missing)}; run metrics.py with the dist stage")
    d = d[d["c"] > 0].copy()
    arms = sorted(d["arm"].unique())
    if not {"dirfix", "naive"} <= set(arms):
        raise SystemExit(f"need both dirfix and naive, found {arms}")

    print("--- by strength: the advantage and the repetition gap grow together, which is what raises the doubt")
    g = d.groupby(["arm", "c"]).agg(logppl=("logppl", "mean"), rep4=("rep4", "mean"),
                                    kw=("keyword_hit", "mean")).unstack(0)
    out = pd.DataFrame({
        "Δ log-ppl": g[("logppl", "dirfix")] - g[("logppl", "naive")],
        "Δ концепт": g[("kw", "dirfix")] - g[("kw", "naive")],
        "Δ повторы": g[("rep4", "dirfix")] - g[("rep4", "naive")],
    })
    print(out.round(4).to_string())

    print("\n--- at matched repetition, which is the test that separates them")
    d["bin"] = pd.cut(d["rep4"], BINS, labels=LABELS)
    t = d.groupby(["bin", "arm"], observed=True).agg(
        n=("logppl", "size"), logppl=("logppl", "mean"), kw=("keyword_hit", "mean"), rep=("rep4", "mean")
    )
    u = t.unstack(1)
    rows = []
    for b in LABELS:
        if b not in u.index:
            continue
        try:
            rows.append({
                "повторы": b,
                "n dirfix": int(u.loc[b, ("n", "dirfix")]),
                "n naive": int(u.loc[b, ("n", "naive")]),
                "Δ log-ppl": u.loc[b, ("logppl", "dirfix")] - u.loc[b, ("logppl", "naive")],
                "Δ концепт": u.loc[b, ("kw", "dirfix")] - u.loc[b, ("kw", "naive")],
                "Δ повторы": u.loc[b, ("rep", "dirfix")] - u.loc[b, ("rep", "naive")],
            })
        except KeyError:
            continue
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))

    zero = res[res["повторы"] == "ровно 0"]
    if zero.empty:
        raise SystemExit("no cells with zero repetition; the control cannot be run on this table")
    dl = float(zero["Δ log-ppl"].iloc[0])
    dk = float(zero["Δ концепт"].iloc[0])
    print(
        f"\nВ ячейках без единого повторённого 4-грамма поправка даёт {dl:+.3f} нат перплексии и "
        f"{dk:+.3f} концепта."
    )
    if dl < 0:
        print("Значит преимущество не является вырождением: оно есть там, где вырождаться нечему.")
    # The sign flip in the degenerate bin is the second half of the argument and worth printing explicitly:
    # if repetition were the source of the win, the most repetitive cells would show the largest win, not
    # the only loss.
    worst = res[res["повторы"] == ">15%"]
    if not worst.empty and float(worst["Δ log-ppl"].iloc[0]) > 0:
        print(
            f"А в самых вырожденных ячейках преимущество меняет знак ({float(worst['Δ log-ppl'].iloc[0]):+.3f}), "
            "то есть повторы поправке вредят, а не помогают."
        )

    dest = RESULTS / (args.scored.replace(".csv", "") + "_repetition_control.csv")
    res.to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_r3.csv")
    main(ap.parse_args())
