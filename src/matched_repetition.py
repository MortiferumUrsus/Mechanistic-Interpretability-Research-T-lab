"""Compare arms at matched repetition (or matched concept hit rate), not at matched strength c.

Different arms reach the same repetition rate at different `c`, so a comparison "at c=2" secretly
compares arms at different amounts of degenerate looping. This script instead asks: at the strength
where an arm reproduces the baseline's repetition rate for this (feature, c_n) cell, how does its
log-PPL and keyword hit rate compare? The match point `c_a` is found by linear interpolation of the
arm's own rep4(c) curve (built from the same c-grid the scored CSV was generated on); if the
baseline's rep4_n falls outside the arm's achievable range, the cell is skipped and counted.

A second mode (--match keyword_hit) matches on keyword hit rate instead, and compares rep4 and
log-PPL -- the mirror question: at matched concept strength, is the arm less repetitive and less
perplexing?

Aggregates are a paired mean of (arm - baseline) over all matched (feature, c_n) cells, bootstrapped
over features (resampling features with replacement, 2000 replications, 95% percentile interval), in
line with paired_at_strength.py's clustering. Reported once pooled over the whole grid and once per
c_n.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS

METRIC_COLS = ("logppl", "keyword_hit", "rep4")
MATCH_MODES = ("rep4", "keyword_hit")
# for each match mode, which delta columns are the ones worth reporting (the matched-on metric's
# own delta is ~0 by construction and is not informative)
COMPARE_COLS = {
    "rep4": ["d_logppl", "d_keyword_hit"],
    "keyword_hit": ["d_logppl", "d_rep4"],
}
N_BOOT = 2000


def curve_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, feature, c), the prompt-averaged metrics -- the curves everything else interpolates."""
    return (
        df.groupby(["arm", "feature", "c"])[list(METRIC_COLS)]
        .mean()
        .reset_index()
        .sort_values(["arm", "feature", "c"])
    )


def find_match(c_grid: np.ndarray, y_grid: np.ndarray, target: float):
    """First grid segment (ascending c) whose y-range brackets `target`; linear interpolation.

    Returns (c_a, seg_idx, t) or None if no segment brackets the target (curve doesn't reach it).
    """
    for i in range(len(c_grid) - 1):
        y0, y1 = y_grid[i], y_grid[i + 1]
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        if lo <= target <= hi:
            t = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
            c_a = c_grid[i] + t * (c_grid[i + 1] - c_grid[i])
            return float(c_a), i, float(t)
    return None


def build_cells(curves: pd.DataFrame, baseline: str, arms: list[str], match_col: str) -> pd.DataFrame:
    base = curves[curves["arm"] == baseline]
    rows = []
    for arm in arms:
        arm_curves = curves[curves["arm"] == arm]
        for feature, base_g in base.groupby("feature"):
            base_g = base_g.sort_values("c")
            arm_g = arm_curves[arm_curves["feature"] == feature].sort_values("c")
            if len(arm_g) < 2:
                for _, brow in base_g.iterrows():
                    rows.append(_missing_row(match_col, arm, feature, brow))
                continue
            c_grid = arm_g["c"].to_numpy(dtype=float)
            y_grid = arm_g[match_col].to_numpy(dtype=float)
            grids = {c: arm_g[c].to_numpy(dtype=float) for c in METRIC_COLS}
            for _, brow in base_g.iterrows():
                target = float(brow[match_col])
                hit = find_match(c_grid, y_grid, target)
                if hit is None:
                    rows.append(_missing_row(match_col, arm, feature, brow))
                    continue
                c_a, i, t = hit
                arm_vals = {c: float(grids[c][i] + t * (grids[c][i + 1] - grids[c][i])) for c in METRIC_COLS}
                rec = {
                    "match_mode": match_col,
                    "arm": arm,
                    "feature": feature,
                    "c_n": float(brow["c"]),
                    "match_value": target,
                    "matched": True,
                    "c_a": c_a,
                    "base_logppl": float(brow["logppl"]),
                    "base_keyword_hit": float(brow["keyword_hit"]),
                    "base_rep4": float(brow["rep4"]),
                    "arm_logppl": arm_vals["logppl"],
                    "arm_keyword_hit": arm_vals["keyword_hit"],
                    "arm_rep4": arm_vals["rep4"],
                    "d_logppl": arm_vals["logppl"] - float(brow["logppl"]),
                    "d_keyword_hit": arm_vals["keyword_hit"] - float(brow["keyword_hit"]),
                    "d_rep4": arm_vals["rep4"] - float(brow["rep4"]),
                }
                rows.append(rec)
    return pd.DataFrame(rows)


def _missing_row(match_col: str, arm: str, feature, brow) -> dict:
    return {
        "match_mode": match_col,
        "arm": arm,
        "feature": feature,
        "c_n": float(brow["c"]),
        "match_value": float(brow[match_col]),
        "matched": False,
        "c_a": np.nan,
        "base_logppl": float(brow["logppl"]),
        "base_keyword_hit": float(brow["keyword_hit"]),
        "base_rep4": float(brow["rep4"]),
        "arm_logppl": np.nan,
        "arm_keyword_hit": np.nan,
        "arm_rep4": np.nan,
        "d_logppl": np.nan,
        "d_keyword_hit": np.nan,
        "d_rep4": np.nan,
    }


def bootstrap_mean(sub: pd.DataFrame, col: str, rng: np.random.Generator, n_boot: int = N_BOOT):
    feats = sub["feature"].unique()
    if len(feats) == 0:
        return float("nan"), float("nan"), float("nan")
    per_feat = {f: sub.loc[sub["feature"] == f, col].to_numpy(dtype=float) for f in feats}
    point = float(sub[col].mean())
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sel = rng.choice(feats, size=len(feats), replace=True)
        vals = np.concatenate([per_feat[f] for f in sel])
        draws[b] = vals.mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize(cells: pd.DataFrame, arms: list[str], match_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    compare_cols = COMPARE_COLS[match_col]
    rows = []
    for arm in arms:
        sub_all = cells[cells["arm"] == arm]
        groups = [("all", sub_all)] + [(f"c_n={c}", g) for c, g in sub_all.groupby("c_n")]
        for label, g in groups:
            n_total = len(g)
            matched = g[g["matched"]]
            n_skipped = n_total - len(matched)
            row = {
                "match_mode": match_col,
                "arm": arm,
                "group": label,
                "n_pairs": n_total,
                "n_matched": len(matched),
                "n_skipped": int(n_skipped),
                "skip_frac": float(n_skipped / n_total) if n_total else float("nan"),
                "n_features": int(matched["feature"].nunique()),
            }
            for col in compare_cols:
                point, lo, hi = bootstrap_mean(matched, col, rng)
                row[f"mean_{col}"] = point
                row[f"{col}_lo95"] = lo
                row[f"{col}_hi95"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored)
    curves = curve_table(df)
    baseline = args.baseline
    default_arms = sorted(set(curves["arm"]) - {baseline, "clean"})
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else default_arms
    missing = [a for a in arms if a not in set(curves["arm"])]
    if missing:
        print(f"warning: arms not present in {args.scored}: {missing} (their cells will all be reported as unmatched)")

    all_cells = []
    all_summary = []
    for match_col in MATCH_MODES:
        cells = build_cells(curves, baseline, arms, match_col)
        all_cells.append(cells)
        summary = summarize(cells, arms, match_col, args.seed)
        all_summary.append(summary)

    cells_df = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    cells_path = RESULTS / f"{args.out_prefix}_cells.csv"
    summary_path = RESULTS / f"{args.out_prefix}_summary.csv"
    cells_df.to_csv(cells_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {len(cells_df)} rows to {cells_path}")
    print(f"wrote {len(summary_df)} rows to {summary_path}")

    lines = [
        f"# Matched-repetition comparison ({args.scored}, baseline = {baseline})",
        "",
        "Each arm is compared to the baseline at the strength where it reproduces the baseline's "
        "value of the matching metric on that (feature, c_n) cell (linear interpolation on the arm's "
        "own c-grid), not at the same nominal c.",
        "",
    ]
    for match_col in MATCH_MODES:
        overall = summary_df[(summary_df["match_mode"] == match_col) & (summary_df["group"] == "all")]
        overall = overall.drop(columns=["match_mode", "group"]).reset_index(drop=True)
        total_pairs = int(overall["n_pairs"].sum())
        total_skipped = int(overall["n_skipped"].sum())
        lines.append(f"## Matched on `{match_col}` (comparing {', '.join(COMPARE_COLS[match_col])})")
        lines.append("")
        lines.append(
            f"{total_skipped} / {total_pairs} (feature, c_n, arm) cells skipped overall "
            f"(baseline's {match_col} value fell outside the arm's achievable range)."
        )
        lines.append("")
        lines.append(_df_to_md(overall.round(4)) if not overall.empty else "(no arms)")
        lines.append("")
    note_path = RESULTS / f"{args.out_prefix}_NOTE.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default="scored_expA_r3.csv")
    ap.add_argument("--baseline", default="naive")
    ap.add_argument("--arms", default="", help="comma-separated; default = every arm except baseline and clean")
    ap.add_argument("--out-prefix", dest="out_prefix", default="matched")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
