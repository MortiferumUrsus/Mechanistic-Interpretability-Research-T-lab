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
    agg = (
        tbl.groupby("arm")["concept_at_budget"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    base = float(agg[agg["arm"] == "naive"]["mean"].iloc[0])
    agg["gain_vs_naive"] = agg["mean"] - base
    agg.to_csv(RESULTS / "dev_candidates.csv", index=False)
    print(f"reference: naive endpoint on DEV = {base:.4f}\n")
    print(agg.to_string(index=False))

    winners: dict[str, dict] = {}
    for _, row in agg.iterrows():
        parsed = parse(str(row["arm"]))
        if parsed is None:
            continue
        fam, params = parsed
        if fam in winners:
            continue
        winners[fam] = {"arm": row["arm"], "gain": float(row["gain_vs_naive"]), **params}

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
    main(ap.parse_args())
