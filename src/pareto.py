"""Pareto fronts, the pre-registered primary endpoint, and its paired bootstrap CI.

Primary endpoint: concept score interpolated onto a fixed fluency budget, where the budget is
the log-perplexity of naive steering at c = 1.0 for that feature. Reported per feature and
averaged, with a paired hierarchical bootstrap (features, then prompts) that recomputes the
whole front inside every replicate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS

REF_ARM = "naive"
REF_C = 1.0
# Secondary budgets, declared before TEST was opened: the primary one is tight, and the naive
# front peaks in concept somewhat above it, so the same comparison is also reported where the
# baseline is actually operating.
SECONDARY_C = (1.5, 2.0)


def _front(logppl: np.ndarray, concept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Achievable frontier: best concept reachable without exceeding a given log-perplexity.

    Any weaker strength is always available, so at budget B every point with logppl <= B is on
    the table; the frontier is therefore the running maximum of concept over sorted logppl.
    """
    order = np.argsort(logppl)
    x, y = logppl[order], concept[order]
    keep, best = [], -np.inf
    for i in range(len(x)):
        if y[i] > best:
            best = y[i]
            keep.append(i)
    idx = np.array(keep, dtype=int)
    return x[idx], y[idx]


def concept_at_budget(logppl: np.ndarray, concept: np.ndarray, budget: float) -> float:
    """Best concept reachable without exceeding a log-perplexity budget.

    Undefined only when the arm cannot reach that fluency at all, that is when its cheapest point is
    already above the budget. When the whole front sits below the budget the answer is its maximum
    concept: every point satisfies the constraint. Returning NaN in that case would penalise exactly
    the arms that are uniformly more fluent than the reference.
    """
    x, y = _front(logppl, concept)
    if len(x) == 0 or budget < x.min():
        return float("nan")
    return float(np.interp(budget, x, y))


def cell_means(df: pd.DataFrame, concept_col: str) -> pd.DataFrame:
    return (
        df.groupby(["feature", "arm", "c"])
        .agg(logppl=("logppl", "mean"), concept=(concept_col, "mean"), n=("text", "size"))
        .reset_index()
    )


def endpoint_table(cells: pd.DataFrame, ref_c: float = REF_C) -> pd.DataFrame:
    rows = []
    for feat, g in cells.groupby("feature"):
        ref = g[(g["arm"] == REF_ARM) & (np.isclose(g["c"], ref_c))]
        if ref.empty:
            continue
        budget = float(ref["logppl"].iloc[0])
        for arm, ga in g.groupby("arm"):
            rows.append(
                {
                    "feature": feat,
                    "arm": arm,
                    "ref_c": ref_c,
                    "budget_logppl": budget,
                    "concept_at_budget": concept_at_budget(
                        ga["logppl"].to_numpy(), ga["concept"].to_numpy(), budget
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap(df: pd.DataFrame, concept_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired hierarchical bootstrap: resample features, then prompts inside each feature."""
    rng = np.random.default_rng(seed)
    feats = df["feature"].unique()
    prompts_by_feat = {f: df.loc[df["feature"] == f, "prompt_idx"].unique() for f in feats}
    arms = sorted(df["arm"].unique())
    draws: dict[str, list[float]] = {a: [] for a in arms}

    for _ in range(n_boot):
        fsel = rng.choice(feats, size=len(feats), replace=True)
        parts = []
        for rep, f in enumerate(fsel):
            pool = prompts_by_feat[f]
            psel = rng.choice(pool, size=len(pool), replace=True)
            sub = df[df["feature"] == f]
            idx = pd.Index(psel, name="prompt_idx")
            picked = sub.set_index("prompt_idx").loc[idx].reset_index()
            picked["feature"] = rep  # replicate id keeps resampled features independent
            parts.append(picked)
        boot = pd.concat(parts, ignore_index=True)
        tbl = endpoint_table(cell_means(boot, concept_col))
        agg = tbl.groupby("arm")["concept_at_budget"].mean()
        for a in arms:
            draws[a].append(float(agg.get(a, np.nan)))

    out = []
    for a in arms:
        v = np.array(draws[a], dtype=float)
        v = v[~np.isnan(v)]
        out.append(
            {
                "arm": a,
                "mean": float(v.mean()) if len(v) else float("nan"),
                "lo95": float(np.quantile(v, 0.025)) if len(v) else float("nan"),
                "hi95": float(np.quantile(v, 0.975)) if len(v) else float("nan"),
                "n_valid": int(len(v)),
            }
        )
    return pd.DataFrame(out)


def paired_delta(
    df: pd.DataFrame, concept_col: str, n_boot: int, seed: int, ref_c: float = REF_C
) -> pd.DataFrame:
    """Bootstrap the per-replicate difference against the reference arm, which is what a
    paired comparison actually needs: CI on the delta, not on two independent means."""
    rng = np.random.default_rng(seed)
    feats = df["feature"].unique()
    arms = sorted(df["arm"].unique())
    deltas: dict[str, list[float]] = {a: [] for a in arms}

    for _ in range(n_boot):
        fsel = rng.choice(feats, size=len(feats), replace=True)
        parts = []
        for rep, f in enumerate(fsel):
            sub = df[df["feature"] == f]
            pool = sub["prompt_idx"].unique()
            psel = rng.choice(pool, size=len(pool), replace=True)
            picked = sub.set_index("prompt_idx").loc[pd.Index(psel, name="prompt_idx")].reset_index()
            picked["feature"] = rep
            parts.append(picked)
        boot = pd.concat(parts, ignore_index=True)
        agg = (
            endpoint_table(cell_means(boot, concept_col), ref_c)
            .groupby("arm")["concept_at_budget"]
            .mean()
        )
        base = agg.get(REF_ARM, np.nan)
        for a in arms:
            deltas[a].append(float(agg.get(a, np.nan) - base))

    rows = []
    for a in arms:
        v = np.array(deltas[a], dtype=float)
        v = v[~np.isnan(v)]
        rows.append(
            {
                "arm": a,
                "delta_mean": float(v.mean()) if len(v) else float("nan"),
                "lo95": float(np.quantile(v, 0.025)) if len(v) else float("nan"),
                "hi95": float(np.quantile(v, 0.975)) if len(v) else float("nan"),
                "p_gt_0": float((v > 0).mean()) if len(v) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def identity_check(df: pd.DataFrame) -> pd.DataFrame:
    """At zero strength every repair arm must reproduce the unsteered generation exactly.

    All arms share the same sampling seed per prompt batch, so this is a byte-level check on real
    generations rather than an appeal to the algebra.
    """
    zero = df[np.isclose(df["c"], 0.0)]
    ref = zero[zero["arm"] == "clean"].set_index(["feature", "prompt_idx"])["text"]
    if ref.empty:
        return pd.DataFrame()
    rows = []
    for arm, g in zero.groupby("arm"):
        s = g.set_index(["feature", "prompt_idx"])["text"]
        common = s.index.intersection(ref.index)
        if len(common) == 0:
            continue
        same = (s.loc[common] == ref.loc[common]).mean()
        rows.append({"arm": arm, "identical_at_zero": float(same), "n": len(common)})
    return pd.DataFrame(rows)


def main(args) -> None:
    # Output names carry a prefix, because the two rounds analyse different scored files and a fixed name
    # means the second run silently overwrites the first one's artefacts. That is exactly what happened
    # once: the round-1 endpoint table was replaced by round 2's, leaving the reported numbers without a
    # file behind them.
    def out(name: str) -> Path:
        return RESULTS / f"{args.prefix}{name}"

    df = pd.read_csv(RESULTS / args.scored)
    ident = identity_check(df)
    if not ident.empty:
        ident.to_csv(out("identity_at_zero.csv"), index=False)
        print("identity at zero strength (should be 1.0 for every arm)")
        print(ident.to_string(index=False), "\n")
    df = df.dropna(subset=["logppl", args.concept])
    cells = cell_means(df, args.concept)
    cells.to_csv(out("cells.csv"), index=False)
    tbl = endpoint_table(cells)
    tbl.to_csv(out("endpoint_per_feature.csv"), index=False)

    summary = (
        tbl.groupby("arm")["concept_at_budget"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "endpoint_mean", "median": "endpoint_median", "count": "n_features"})
        .reset_index()
    )
    # The endpoint mean above is per-arm; the paired delta below is on the features both arms define. When
    # those supports differ, the delta is NOT the difference of the two endpoint means, and a reader who
    # subtracts the columns gets a contradiction. So also report the endpoint restricted to common support.
    common = set(tbl.loc[tbl["arm"] == REF_ARM, "feature"])
    for arm, g in tbl.groupby("arm"):
        common &= set(g.dropna(subset=["concept_at_budget"])["feature"])
    on_common = (
        tbl[tbl["feature"].isin(common)]
        .groupby("arm")["concept_at_budget"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "endpoint_on_common", "count": "n_common"})
        .reset_index()
    )
    summary = summary.merge(on_common, on="arm", how="left")
    boot = bootstrap(df, args.concept, args.n_boot, args.seed).rename(
        columns={"mean": "boot_mean", "lo95": "boot_lo95", "hi95": "boot_hi95"}
    )
    delta = paired_delta(df, args.concept, args.n_boot, args.seed).rename(
        columns={"lo95": "delta_lo95", "hi95": "delta_hi95"}
    )
    summary = summary.merge(boot, on="arm").merge(delta, on="arm")
    summary.to_csv(out("endpoint_summary.csv"), index=False)
    print(f"primary endpoint, budget = naive at c={REF_C}")
    print(summary.to_string(index=False))

    sec = []
    for rc in SECONDARY_C:
        t = endpoint_table(cells, rc)
        if t.empty:
            continue
        d = paired_delta(df, args.concept, max(400, args.n_boot // 4), args.seed, rc).rename(
            columns={"lo95": "delta_lo95", "hi95": "delta_hi95"}
        )
        m = (
            t.groupby("arm")["concept_at_budget"]
            .mean()
            .rename("concept_at_budget")
            .reset_index()
            .merge(d, on="arm")
        )
        m.insert(0, "ref_c", rc)
        sec.append(m)
    if sec:
        secondary = pd.concat(sec, ignore_index=True)
        secondary.to_csv(out("endpoint_secondary.csv"), index=False)
        print("\nsecondary budgets")
        print(secondary.to_string(index=False))
    (out("endpoint_summary.json")).write_text(
        json.dumps(
            {
                "concept_metric": args.concept,
                "reference_arm": REF_ARM,
                "reference_c": REF_C,
                "n_boot": args.n_boot,
                "rows": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_test.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="", help="prefix for output artefacts, so rounds do not overwrite each other")
    main(ap.parse_args())
