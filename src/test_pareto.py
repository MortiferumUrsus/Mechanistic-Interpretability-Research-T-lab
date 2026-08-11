"""Checks for the Pareto front and the primary endpoint. Run: python test_pareto.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pareto import _front, cell_means, concept_at_budget, endpoint_table, paired_delta


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    assert cond, name


def synthetic(gain: float, n_feat: int = 6, n_prompt: int = 20, seed: int = 0) -> pd.DataFrame:
    """Two arms on a shared trade-off curve; `gain` lifts the second arm's concept axis."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_feat):
        for arm, lift in (("naive", 0.0), ("method", gain)):
            for c in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
                for p in range(n_prompt):
                    rows.append(
                        {
                            "feature": f,
                            "arm": arm,
                            "c": c,
                            "prompt_idx": p,
                            "text": "x",
                            "logppl": 3.0 + 0.8 * c + rng.normal(0, 0.05),
                            "keyword_hit": float(
                                rng.random() < min(0.95, 0.25 * c + lift)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.1, 0.5, 0.4, 0.9])
    fx, fy = _front(x, y)
    check("front drops dominated points", list(fx) == [1.0, 2.0, 4.0], f"{list(fx)}")
    check("front is concept-increasing", bool(np.all(np.diff(fy) > 0)), f"{list(fy)}")

    check(
        "budget inside support interpolates",
        abs(concept_at_budget(x, y, 1.5) - 0.3) < 1e-9,
        f"{concept_at_budget(x, y, 1.5)}",
    )
    check("budget below the cheapest point is NaN", np.isnan(concept_at_budget(x, y, 0.5)))
    check(
        "budget above the whole front credits its maximum",
        abs(concept_at_budget(x, y, 9.0) - 0.9) < 1e-9,
        f"{concept_at_budget(x, y, 9.0)}",
    )

    df = synthetic(gain=0.0)
    tbl = endpoint_table(cell_means(df, "keyword_hit"))
    agg = tbl.groupby("arm")["concept_at_budget"].mean()
    check(
        "identical arms give no endpoint gap",
        abs(agg["method"] - agg["naive"]) < 0.06,
        f"gap={agg['method'] - agg['naive']:.4f}",
    )
    d0 = paired_delta(df, "keyword_hit", n_boot=200, seed=0)
    p0 = float(d0[d0["arm"] == "method"]["p_gt_0"].iloc[0])
    check("null case does not claim significance", 0.1 < p0 < 0.9, f"p_gt_0={p0:.3f}")

    df = synthetic(gain=0.25)
    tbl = endpoint_table(cell_means(df, "keyword_hit"))
    agg = tbl.groupby("arm")["concept_at_budget"].mean()
    check(
        "real gain is recovered",
        agg["method"] - agg["naive"] > 0.15,
        f"gap={agg['method'] - agg['naive']:.4f}",
    )
    d1 = paired_delta(df, "keyword_hit", n_boot=200, seed=0)
    p1 = float(d1[d1["arm"] == "method"]["p_gt_0"].iloc[0])
    check("real gain is detected", p1 > 0.95, f"p_gt_0={p1:.3f}")
    check(
        "reference arm has zero delta by construction",
        abs(float(d1[d1["arm"] == "naive"]["delta_mean"].iloc[0])) < 1e-12,
    )
    print("pareto machinery behaves")


if __name__ == "__main__":
    main()
