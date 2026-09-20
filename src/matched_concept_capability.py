"""Real-text damage at matched concept delivery.

Comparing arms at the same steering strength compares them at different amounts of delivered concept,
and comparing them on their own generations rewards whatever makes the generating model more confident
(section 10.4 of the report). This script joins the two measurements that avoid both problems:

  * the concept curve, keyword hit rate versus strength, averaged over features and prompts from a
    scored generation table (`--scored`, e.g. scored_expF_r3.csv, 12 test_r3 features);
  * the capability curve, teacher-forced NLL and next-token top-1 accuracy on held-out real text
    versus strength, from capability_sweep.py (`--capability`, e.g. capability_sweep.csv, 4 features).

For each arm and each target hit rate, the strength at which the arm's hit rate first reaches the target
is found by linear interpolation on the rising part of the concept curve (up to its maximum), and the
NLL and top-1 at that strength are read from the capability curve by linear interpolation. Arms whose
concept curve never reaches the target get NaN. The two curves are averaged over different feature sets;
the report says so where it quotes the table.

    python matched_concept_capability.py --bootstrap 2000       # scored_expF_r3.csv + capability_sweep.csv, 12-feature concept curve
    python matched_concept_capability.py --first-n-features 4 --bootstrap 2000 --out matched_concept_capability_4feat.csv
    python matched_concept_capability.py --scored scored_expF_ent_clean.csv --capability capability_ent.csv \
        --arm-map dirfix=dir_ent --bootstrap 2000 --out matched_concept_capability_ent.csv

`--bootstrap` resamples the concept-side features with replacement and reports a percentile interval on the
NLL difference against `--ref`; the capability curve has no per-feature rows, so the interval covers the
uncertainty of the matched strength only.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS

TARGETS = (0.25, 0.30, 0.40)


def strength_at_target(c: np.ndarray, hit: np.ndarray, target: float) -> float:
    """Smallest strength at which the rising part of the hit-rate curve reaches `target`."""
    peak = int(np.argmax(hit))
    c_rise, h_rise = c[: peak + 1], hit[: peak + 1]
    if h_rise[-1] < target:
        return float("nan")
    for i in range(1, len(c_rise)):
        if h_rise[i] >= target:
            lo_c, hi_c = c_rise[i - 1], c_rise[i]
            lo_h, hi_h = h_rise[i - 1], h_rise[i]
            if hi_h == lo_h:
                return float(hi_c)
            return float(lo_c + (target - lo_h) / (hi_h - lo_h) * (hi_c - lo_c))
    return float(c_rise[0])


def concept_curves(scored: pd.DataFrame, concept: str, features) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-arm (c grid, mean hit rate) over the given features, each feature first averaged over prompts."""
    sub = scored[scored["feature"].isin(features)]
    per_feat = sub.groupby(["arm", "feature", "c"])[concept].mean().reset_index()
    curve = per_feat.groupby(["arm", "c"])[concept].mean().reset_index().sort_values(["arm", "c"])
    return {arm: (g["c"].to_numpy(float), g[concept].to_numpy(float)) for arm, g in curve.groupby("arm")}


def read_capability(cap: pd.DataFrame, arm: str, c_star: float) -> tuple[float, float]:
    cap_arm = cap[cap["arm"] == arm].sort_values("c")
    if cap_arm.empty or not np.isfinite(c_star):
        return float("nan"), float("nan")
    return (
        float(np.interp(c_star, cap_arm["c"], cap_arm["nll"])),
        float(np.interp(c_star, cap_arm["c"], cap_arm["top1_acc"])),
    )


def main(args) -> None:
    scored = pd.read_csv(RESULTS / args.scored)
    cap = pd.read_csv(RESULTS / args.capability)
    arm_map = dict(kv.split("=", 1) for kv in args.arm_map) if args.arm_map else {}
    arms = [a for a in sorted(scored["arm"].unique()) if a in set(cap["arm"])]
    skipped = sorted(set(scored["arm"].unique()) - set(arms))
    if skipped:
        print(f"no capability curve in {args.capability} for {skipped}; skipped")

    # Which features feed the concept curve: all of them (default) or, with --first-n-features, the first
    # N in the split's order, which is the subset capability_sweep.py used (its --features default is 4).
    features = sorted(scored["feature"].unique())
    if args.first_n_features:
        from common import load_features

        order = [int(r["index"]) for r in load_features(args.split)]
        features = [f for f in order if f in set(features)][: args.first_n_features]
        print(f"concept curve restricted to the first {len(features)} features of {args.split}: {features}")

    curves = concept_curves(scored, args.concept, features)
    rows = []
    for arm in arms:
        c_grid, hit = curves[arm]
        for target in TARGETS:
            c_star = strength_at_target(c_grid, hit, target)
            nll, top1 = read_capability(cap, arm, c_star)
            rows.append(
                {
                    "arm": arm_map.get(arm, arm),
                    "target_concept": target,
                    "c_at_target": c_star,
                    "nll": nll,
                    "top1_acc": top1,
                    "n_features_concept": len(features),
                    "n_features_capability": int(cap[cap["arm"] == arm]["n_features"].iloc[0]) if "n_features" in cap else -1,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / args.out, index=False)
    print(out.round(3).to_string(index=False))
    print(f"wrote results/{args.out}")

    # Feature bootstrap on the concept side only. The capability curve is a 4-feature mean with no
    # per-feature rows, so this interval reflects the uncertainty of the matched strength c*, not of the
    # NLL curve itself; it is reported as such.
    if args.bootstrap and args.ref in arms:
        rng = np.random.default_rng(args.seed)
        feats = np.array(features)
        diffs: dict[tuple[str, float], list[float]] = {}
        for _ in range(args.bootstrap):
            sample = rng.choice(feats, size=len(feats), replace=True)
            # duplicate features by concatenating their rows, so the resample weights the curve
            sub = pd.concat([scored[scored["feature"] == f] for f in sample], ignore_index=True)
            sub["feature"] = np.repeat(np.arange(len(sample)), [int((scored["feature"] == f).sum()) for f in sample])
            cv = concept_curves(sub, args.concept, list(range(len(sample))))
            base = {}
            for target in TARGETS:
                c_ref = strength_at_target(*cv[args.ref], target)
                base[target] = read_capability(cap, args.ref, c_ref)[0]
            for arm in arms:
                if arm == args.ref:
                    continue
                for target in TARGETS:
                    c_star = strength_at_target(*cv[arm], target)
                    nll = read_capability(cap, arm, c_star)[0]
                    diffs.setdefault((arm, target), []).append(nll - base[target])
        brows = []
        for (arm, target), vals in diffs.items():
            v = np.array(vals, dtype=float)
            ok = v[np.isfinite(v)]
            brows.append(
                {
                    "arm": arm_map.get(arm, arm),
                    "target_concept": target,
                    "d_nll_vs_ref_point": float(out[(out["arm"] == arm_map.get(arm, arm)) & (out["target_concept"] == target)]["nll"].iloc[0]
                                                - out[(out["arm"] == arm_map.get(args.ref, args.ref)) & (out["target_concept"] == target)]["nll"].iloc[0]),
                    "boot_mean": float(ok.mean()) if ok.size else float("nan"),
                    "lo95": float(np.percentile(ok, 2.5)) if ok.size else float("nan"),
                    "hi95": float(np.percentile(ok, 97.5)) if ok.size else float("nan"),
                    "n_valid": int(ok.size),
                    "n_boot": int(v.size),
                    "share_target_unreachable": float(1 - ok.size / v.size),
                }
            )
        bout = pd.DataFrame(brows).sort_values(["arm", "target_concept"])
        bpath = args.out.replace(".csv", "_boot.csv")
        bout.to_csv(RESULTS / bpath, index=False)
        print(bout.round(3).to_string(index=False))
        print(f"wrote results/{bpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_expF_r3.csv")
    ap.add_argument("--capability", default="capability_sweep.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--arm-map", dest="arm_map", nargs="*", default=None, help="rename arms in the output, e.g. dirfix=dir_ent")
    ap.add_argument("--out", default="matched_concept_capability.csv")
    ap.add_argument("--split", default="test_r3")
    ap.add_argument("--first-n-features", dest="first_n_features", type=int, default=0,
                    help="restrict the concept curve to the first N features of the split (the subset capability_sweep.py used)")
    ap.add_argument("--bootstrap", type=int, default=0, help="feature-bootstrap replicates for the NLL difference vs --ref")
    ap.add_argument("--ref", default="naive")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
