"""Report the natural steering scale of each selected latent.

A latent's own ceiling on real text is the meaningful unit of steering strength: injecting
`ceiling * ||W_dec||` reproduces the strongest activation the feature ever reaches naturally.
Expressing strength as a fraction of the global activation norm instead hides the fact that
different latents live on very different scales.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import yaml

from common import DATA, DEVICE, RESULTS, ROOT, ActStats, load_sae


def main(args) -> None:
    stats = ActStats.load()
    sae = load_sae()
    ceiling = torch.load(DATA / "sae_feature_stats.pt", map_location=DEVICE)["max_act"].to(DEVICE)
    feats = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))

    rows = []
    for split in ("test", "dev"):
        for r in feats[split]:
            f = int(r["index"])
            v = sae.W_dec[f].detach().float()
            dn = float(v.norm())
            nat = float(ceiling[f]) * dn
            rows.append(
                {
                    "split": split,
                    "index": f,
                    "dec_norm": round(dn, 4),
                    "ceiling_act": round(float(ceiling[f]), 3),
                    "natural_s": round(nat, 2),
                    "natural_c_in_hnorm": round(nat / stats.median_norm, 4),
                    "top": " ".join(r["top_tokens"][:5]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "feature_scales.csv", index=False)
    print(f"median||h|| = {stats.median_norm:.2f}")
    print(df.to_string(index=False))
    print(
        "\nnatural_c_in_hnorm summary: "
        f"min={df.natural_c_in_hnorm.min():.3f} median={df.natural_c_in_hnorm.median():.3f} "
        f"max={df.natural_c_in_hnorm.max():.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
