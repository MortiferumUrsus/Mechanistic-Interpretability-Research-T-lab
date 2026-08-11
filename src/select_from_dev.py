"""Pick the winning knob setting per arm family on DEV and write configs/frozen.yaml.

Selection uses the same endpoint the report uses, computed on DEV features only. Once this file is
written, TEST is generated once and nothing here is revisited.
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd
import yaml

from common import RESULTS, ROOT
from pareto import cell_means, endpoint_table

CONFIGS = ROOT / "configs"

FAMILY = {
    "cds": re.compile(r"^cds\|cds_(?P<den>.+)_lam(?P<lam>[\d.]+)$"),
    "denoise_naive": re.compile(r"^denoise_naive\|dn_(?P<den>.+)_eta(?P<eta>[\d.]+)$"),
    "mts": re.compile(r"^mts\|mts_sh(?P<shrink>[\d.]+)$"),
    "fsr": re.compile(r"^fsr\|fsr_k(?P<fsr_k>[\d.]+)$"),
}


def parse(arm: str) -> tuple[str, dict] | None:
    for fam, rx in FAMILY.items():
        m = rx.match(arm)
        if m:
            return fam, m.groupdict()
    return None


def main(args) -> None:
    df = pd.read_csv(RESULTS / args.scored).dropna(subset=["logppl", args.concept])
    # the DEV runs tag arm names; the untagged reference is the naive arm from the base run
    df["arm"] = df["arm"].str.replace(r"^(clean|naive|norm_preserving)\|base$", r"\1", regex=True)
    cells = cell_means(df, args.concept)
    tbl = endpoint_table(cells)

    # Paired per-feature differences on common support. Averaging each arm's endpoint over whatever
    # features happen to be defined for it compares different feature sets: the endpoint is
    # undefined wherever the fluency budget falls outside an arm's front.
    ref = tbl[tbl["arm"] == "naive"][["feature", "concept_at_budget"]].rename(
        columns={"concept_at_budget": "ref"}
    )
    rows = []
    for arm, g in tbl.groupby("arm"):
        m = g.merge(ref, on="feature").dropna(subset=["concept_at_budget", "ref"])
        if m.empty:
            rows.append({"arm": arm, "n_paired": 0, "mean_paired": float("nan"),
                         "ref_paired": float("nan"), "gain_vs_naive": float("nan"),
                         "n_defined": int(g["concept_at_budget"].notna().sum())})
            continue
        rows.append(
            {
                "arm": arm,
                "n_paired": len(m),
                "mean_paired": float(m["concept_at_budget"].mean()),
                "ref_paired": float(m["ref"].mean()),
                "gain_vs_naive": float((m["concept_at_budget"] - m["ref"]).mean()),
                "n_defined": int(g["concept_at_budget"].notna().sum()),
            }
        )
    agg = pd.DataFrame(rows).sort_values("gain_vs_naive", ascending=False)
    agg.to_csv(RESULTS / "dev_candidates.csv", index=False)
    n_feat = tbl["feature"].nunique()
    print(f"DEV features: {n_feat}; gains are paired per-feature differences on common support\n")
    print(agg.to_string(index=False))

    winners: dict[str, dict] = {}
    for _, row in agg.iterrows():
        parsed = parse(str(row["arm"]))
        if parsed is None or row["n_paired"] < args.min_paired:
            continue
        fam, params = parsed
        if fam in winners:
            continue
        winners[fam] = {
            "arm": row["arm"],
            "gain": float(row["gain_vs_naive"]),
            "n_paired": int(row["n_paired"]),
            **params,
        }

    frozen = yaml.safe_load((CONFIGS / "frozen.yaml").read_text(encoding="utf-8"))
    if "cds" in winners:
        frozen["denoiser"] = winners["cds"]["den"]
        frozen["lam"] = float(winners["cds"]["lam"])
    if "denoise_naive" in winners:
        frozen["eta"] = float(winners["denoise_naive"]["eta"])
        # the literal arm keeps its own best denoiser recorded, for the report
        frozen["denoise_naive_denoiser"] = winners["denoise_naive"]["den"]
    if "mts" in winners:
        frozen["shrink"] = float(winners["mts"]["shrink"])
    if "fsr" in winners:
        frozen["fsr_k"] = float(winners["fsr"]["fsr_k"])

    header = (
        "# Inference knobs, selected on DEV features and frozen before TEST was generated.\n"
        "# Written by src/select_from_dev.py; TEST varies only the strength c.\n"
    )
    (CONFIGS / "frozen.yaml").write_text(
        header + yaml.safe_dump(frozen, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    (RESULTS / "dev_winners.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    print("\nfrozen.yaml:")
    print(yaml.safe_dump(frozen, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_dev_all.csv")
    ap.add_argument("--concept", default="keyword_hit")
    ap.add_argument("--min-paired", dest="min_paired", type=int, default=4)
    main(ap.parse_args())
