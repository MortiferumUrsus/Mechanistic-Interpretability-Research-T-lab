"""Feature-level distribution of the (arm - baseline) effect, not just its paired mean.

A single averaged "arm beats baseline" number can hide the fact that some features move the wrong
way. Anti-steerability (Tan et al., NeurIPS 2024) is exactly this: for some latents, more steering
along the "right" direction makes the target concept less likely, not more. This script reports the
per-feature table of paired differences (arm - baseline) at fixed strengths, and the fraction of
features whose sign is opposite the group's own mean sign, for both log-PPL and keyword hit rate.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def opposite_sign(delta: pd.Series, mean_delta: float) -> pd.Series:
    """True where `delta`'s sign disagrees with the group mean's sign. Zeros never count as opposite."""
    if mean_delta == 0:
        return pd.Series(False, index=delta.index)
    ref = np.sign(mean_delta)
    d_sign = np.sign(delta)
    return (d_sign != 0) & (d_sign != ref)


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored)
    cell = df.groupby(["arm", "feature", "c"])[["logppl", "keyword_hit"]].mean().reset_index()
    baseline = args.baseline
    base = cell[cell["arm"] == baseline]

    default_arms = sorted(set(cell["arm"]) - {baseline, "clean"})
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else default_arms
    cs = [float(x) for x in args.c.split(",")]

    note_rows = []
    for arm in arms:
        arm_cell = cell[cell["arm"] == arm]
        per_c = []
        for c in cs:
            b = base[np.isclose(base["c"], c)][["feature", "logppl", "keyword_hit"]].rename(
                columns={"logppl": "base_logppl", "keyword_hit": "base_keyword_hit"}
            )
            a = arm_cell[np.isclose(arm_cell["c"], c)][["feature", "logppl", "keyword_hit"]].rename(
                columns={"logppl": "arm_logppl", "keyword_hit": "arm_keyword_hit"}
            )
            m = b.merge(a, on="feature", how="inner")
            if m.empty:
                print(f"feature_distribution: no overlapping features for arm={arm} c={c}; skipping this c")
                continue
            m["c"] = c
            m["delta_logppl"] = m["arm_logppl"] - m["base_logppl"]
            m["delta_keyword_hit"] = m["arm_keyword_hit"] - m["base_keyword_hit"]
            mean_dl = float(m["delta_logppl"].mean())
            mean_dk = float(m["delta_keyword_hit"].mean())
            m["sign_opposite_logppl"] = opposite_sign(m["delta_logppl"], mean_dl)
            m["sign_opposite_keyword_hit"] = opposite_sign(m["delta_keyword_hit"], mean_dk)
            per_c.append(m)

            note_rows.append(
                {
                    "arm": arm,
                    "c": c,
                    "n_features": len(m),
                    "mean_delta_logppl": mean_dl,
                    "frac_opposite_logppl": float(m["sign_opposite_logppl"].mean()),
                    "min_delta_logppl": float(m["delta_logppl"].min()),
                    "median_delta_logppl": float(m["delta_logppl"].median()),
                    "max_delta_logppl": float(m["delta_logppl"].max()),
                    "mean_delta_keyword_hit": mean_dk,
                    "frac_opposite_keyword_hit": float(m["sign_opposite_keyword_hit"].mean()),
                    "min_delta_keyword_hit": float(m["delta_keyword_hit"].min()),
                    "median_delta_keyword_hit": float(m["delta_keyword_hit"].median()),
                    "max_delta_keyword_hit": float(m["delta_keyword_hit"].max()),
                }
            )
        out_cols = [
            "feature", "c", "base_logppl", "arm_logppl", "delta_logppl", "sign_opposite_logppl",
            "base_keyword_hit", "arm_keyword_hit", "delta_keyword_hit", "sign_opposite_keyword_hit",
        ]
        out_df = pd.concat(per_c, ignore_index=True)[out_cols] if per_c else pd.DataFrame(columns=out_cols)
        out_path = RESULTS / f"{args.out_prefix}_{arm}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"wrote {len(out_df)} rows to {out_path}")

    note_df = pd.DataFrame(note_rows)
    lines = [
        f"# Feature distribution of the (arm - baseline) effect ({args.scored}, baseline = {baseline})",
        "",
        "Anti-steerability check: the fraction of features whose paired delta has the sign opposite "
        "the group mean, at fixed c, plus the min/median/max of the delta across features.",
        "",
    ]
    if not note_df.empty:
        show = note_df.round(4)
        lines.append(_df_to_md(show))
    else:
        lines.append("(no data)")
    lines.append("")
    note_path = RESULTS / f"{args.out_prefix}_NOTE.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default="scored_expA_r3.csv")
    ap.add_argument("--baseline", default="naive")
    ap.add_argument("--c", default="1.0,1.5,2.0")
    ap.add_argument("--arms", default="", help="comma-separated; default = every arm except baseline and clean")
    ap.add_argument("--out-prefix", dest="out_prefix", default="featdist")
    main(ap.parse_args())
