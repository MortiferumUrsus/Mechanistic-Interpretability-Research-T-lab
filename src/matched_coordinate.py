"""Does the direction correction still win when the baseline is matched on the concept coordinate?

The correction injects along `w = normalise(v + A B^T v)` at the same norm as naive steering, and
`cos(w, v-hat)` is 0.866 on average (0.72 to 0.96 across features). So at the same `c`, the correction puts
only `cos * s` along `v-hat` itself: on average 13 per cent less, and up to 28 per cent less for some
features. That is a real alternative explanation for the fluency half of the result -- a smaller effective
push along the direction that does the damage would improve perplexity all by itself.

This script answers it without generating anything new. For every feature it computes the exact per-feature
cosine from the trained correction, then reads naive steering off its own measured front at the *matched*
coordinate `c' = cos * c` by interpolation, and compares the correction against that. Two matchings are
therefore reported side by side:

  equal norm      -- the same perturbation budget is spent (what the report has so far)
  equal coordinate -- the same increment along v-hat is delivered (what this adds)

If the correction still dominates under the second matching, the mechanical explanation is closed. If it
does not, the fluency gain is at least partly an artefact of pushing less hard along `v-hat`, and the report
has to say so.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS


def per_feature_cos(features: list[int], ckpt_path: str) -> dict[int, float]:
    """cos(w, v-hat) for the trained low-rank correction, computed exactly rather than assumed."""
    import torch
    from sae_lens import SAE

    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob.get("state_dict", blob)
    # The trained module is `w = normalise(v + up(down(v)))` with down: 768 -> 64 and up: 64 -> 768,
    # see DirectionCorrection.forward in train_direction.py. Reproduced here rather than imported so the
    # check does not depend on the training module staying importable.
    down = state["down.weight"].float()
    up = state["up.weight"].float()
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre", device="cpu")
    out = {}
    for f in features:
        v = sae.W_dec[f].detach().float()
        vh = v / v.norm()
        w = vh + up @ (down @ vh)
        w = w / w.norm()
        out[int(f)] = float(torch.dot(w, vh))
    return out


def interp_naive(cells: pd.DataFrame, feature: int, c_target: float) -> tuple[float, float]:
    """Naive steering's (log-PPL, concept) at an arbitrary strength, read off its measured curve."""
    g = cells[(cells["feature"] == feature) & (cells["arm"] == "naive")].sort_values("c")
    if g.empty:
        return float("nan"), float("nan")
    cs = g["c"].to_numpy()
    return (
        float(np.interp(c_target, cs, g["logppl"].to_numpy())),
        float(np.interp(c_target, cs, g["concept"].to_numpy())),
    )


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored)
    df = df.dropna(subset=["logppl", args.concept])
    cells = (
        df.groupby(["feature", "arm", "c"])
        .agg(logppl=("logppl", "mean"), concept=(args.concept, "mean"))
        .reset_index()
    )
    features = sorted(cells["feature"].unique().tolist())
    cosines = per_feature_cos(features, args.ckpt)
    print("per-feature cos(w, v_hat):")
    for f in features:
        print(f"  {f:6d}  {cosines[f]:.4f}")
    print(f"  mean {np.mean(list(cosines.values())):.4f}  min {min(cosines.values()):.4f}  max {max(cosines.values()):.4f}")

    strengths = sorted(c for c in cells["c"].unique() if c >= args.c_min)
    rows = []
    for f in features:
        cos = cosines[f]
        for c in strengths:
            fix = cells[(cells["feature"] == f) & (cells["arm"] == "dirfix") & (np.isclose(cells["c"], c))]
            nai = cells[(cells["feature"] == f) & (cells["arm"] == "naive") & (np.isclose(cells["c"], c))]
            if fix.empty or nai.empty:
                continue
            m_ppl, m_con = interp_naive(cells, f, cos * c)
            rows.append(
                {
                    "feature": f,
                    "c": c,
                    "cos": cos,
                    "c_matched": cos * c,
                    "fix_logppl": float(fix["logppl"].iloc[0]),
                    "fix_concept": float(fix["concept"].iloc[0]),
                    "naive_logppl": float(nai["logppl"].iloc[0]),
                    "naive_concept": float(nai["concept"].iloc[0]),
                    "matched_logppl": m_ppl,
                    "matched_concept": m_con,
                }
            )
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RESULTS / f"{args.prefix}matched_coordinate_cells.csv", index=False)

    def paired(logppl_col: str, concept_col: str, label: str) -> dict:
        d_ppl = cmp["fix_logppl"] - cmp[logppl_col]
        d_con = cmp["fix_concept"] - cmp[concept_col]
        rng = np.random.default_rng(args.seed)
        feats = cmp["feature"].unique()
        boots_ppl, boots_con, dominates = [], [], []
        for _ in range(args.n_boot):
            sel = rng.choice(feats, size=len(feats), replace=True)
            sub = pd.concat([cmp[cmp["feature"] == f] for f in sel])
            bp = float((sub["fix_logppl"] - sub[logppl_col]).mean())
            bc = float((sub["fix_concept"] - sub[concept_col]).mean())
            boots_ppl.append(bp)
            boots_con.append(bc)
            dominates.append(bp < 0 and bc > 0)
        out = {
            "matching": label,
            "d_logppl": float(d_ppl.mean()),
            "logppl_lo95": float(np.quantile(boots_ppl, 0.025)),
            "logppl_hi95": float(np.quantile(boots_ppl, 0.975)),
            "d_concept": float(d_con.mean()),
            "concept_lo95": float(np.quantile(boots_con, 0.025)),
            "concept_hi95": float(np.quantile(boots_con, 0.975)),
            "p_dominates": float(np.mean(dominates)),
            "n_cells": int(len(cmp)),
            "n_features": int(len(feats)),
        }
        return out

    res = [
        paired("naive_logppl", "naive_concept", "equal norm (same c)"),
        paired("matched_logppl", "matched_concept", "equal concept coordinate (naive at cos*c)"),
    ]
    table = pd.DataFrame(res)
    table.to_csv(RESULTS / f"{args.prefix}matched_coordinate_summary.csv", index=False)
    (RESULTS / f"{args.prefix}matched_coordinate_summary.json").write_text(
        json.dumps({"cos_mean": float(np.mean(list(cosines.values()))), "results": res}, indent=2),
        encoding="utf-8",
    )
    print("\ndirection correction against naive, two matchings, paired bootstrap over features:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    verdict = res[1]
    print(
        "\nunder equal concept coordinate the correction "
        + ("still dominates" if verdict["p_dominates"] > 0.8 else "no longer clearly dominates")
        + f" (p = {verdict['p_dominates']:.2f})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_test_r2.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--ckpt", default=str(RESULTS.parent / "artifacts" / "direction_correction.pt"))
    ap.add_argument("--c-min", dest="c_min", type=float, default=1.0)
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    # Output names take a prefix so different rounds can never overwrite each other's tables.
    ap.add_argument("--prefix", default="", help="prepended to output filenames, e.g. r3_")
    main(ap.parse_args())
