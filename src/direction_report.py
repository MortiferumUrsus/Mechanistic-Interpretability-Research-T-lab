"""Diagnose "confidence contamination" in steering directions.

Round two found that the trained direction correction w = normalize(v + Mv) is about half
explained by one shared direction d_bar (anatomy.py, checkpoints/shared_direction.pt), and that d_bar
sits mostly in the weak half of the unembedding spectrum and acts as a generic confidence dial
(it collapses next-token entropy and triggers repetition) rather than a concept-specific
direction. This script measures how much of that same signature -- alignment with d_bar,
alignment with the corpus mean, weak-spectrum energy, raw reach into the logits -- sits in each
of the direction families used by steering.py's arms, for every evaluation feature in
configs/features.yaml (test, dev, test_r3).

Families per feature:
    decoder             v_hat, the raw SAE decoder column
    shared              normalize(v_hat + kappa_shared * d_bar)
    diffmeans           normalize(mu_f - mu), the DiffMeansArm direction
    centred             normalize(v_hat - (v_hat . mu_hat) mu_hat), the CentredArm direction
    purified            normalize(v_hat - (v_hat . d_bar) d_bar), the PurifiedArm direction
    diffmeans_purified  normalize(u_dm - (u_dm . d_bar) d_bar), the DiffMeansPurifiedArm direction
    learned             normalize(v_hat + M v_hat), M = up.weight @ down.weight from --ckpt

    python direction_report.py [--ckpt dir_hot] [--kappa-shared 0.75] [--device cpu]
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import yaml

from common import CKPT, DATA, RESULTS, ROOT, SHARED_DIRECTION, ActStats, load_features, load_model, load_sae

N_WEAK = 384  # half of d_model=768: the "weak half" of the W_U spectrum, smallest singular values


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp_min(1e-9)


def load_M(ckpt: str, device: str) -> torch.Tensor:
    """M = up.weight @ down.weight from a train_direction.py checkpoint; same as anatomy.py."""
    blob = torch.load(CKPT / f"{ckpt}.pt", map_location=device)
    sd = blob["state_dict"]
    return sd["up.weight"].float() @ sd["down.weight"].float()


def lookup_concept_mu(blob: dict, feat_idx: int) -> torch.Tensor:
    """mu_f for a feature: eval_mu if it was mined as a held-out eval latent, else the fit-pool mu.

    Same fallback order as denoiser_cond._lookup_mu / generate.py's lookup_concept_mu.
    """
    eval_features = blob["eval_features"]
    hit = (eval_features == feat_idx).nonzero(as_tuple=True)[0]
    if hit.numel() > 0:
        return blob["eval_mu"][hit[0]]
    features = blob["features"]
    hit = (features == feat_idx).nonzero(as_tuple=True)[0]
    if hit.numel() > 0:
        return blob["mu"][hit[0]]
    raise ValueError(
        f"feature {feat_idx} has no mu_f in data/concept_stats.pt (checked eval_features and "
        "features); mine it with concept_data.py or add the feature to configs/features.yaml first"
    )


@torch.no_grad()
def main(args) -> None:
    device = args.device
    model = load_model(device=device)
    sae = load_sae(device=device)
    stats = ActStats.load(device=device)
    W_dec = sae.W_dec.detach().float()

    d_bar = torch.load(SHARED_DIRECTION, map_location=device)["d_bar"].to(device).float()
    mu_hat = unit(stats.mean)
    M = load_M(args.ckpt, device)
    concept_blob = torch.load(DATA / "concept_stats.pt", map_location=device)

    # Weak half of the unembedding spectrum: the N_WEAK left singular vectors of W_U with the
    # smallest singular values, same construction as the probe this task refers to.
    W_U = model.W_U.detach().float()  # [d_model, d_vocab]
    U, S, _ = torch.linalg.svd(W_U, full_matrices=False)
    weak_order = torch.argsort(S)[:N_WEAK]  # ascending: N_WEAK smallest-S directions
    weak = U[:, weak_order]  # [d_model, N_WEAK], orthonormal

    feats = load_features()
    rows = []
    for split in ("test", "dev", "test_r3"):
        for rec in feats.get(split, []):
            f = int(rec["index"])
            v_hat = unit(W_dec[f])
            mu_f = lookup_concept_mu(concept_blob, f).to(device=device, dtype=torch.float32)
            u_dm = unit(mu_f - stats.mean)

            families = {
                "decoder": v_hat,
                "shared": unit(v_hat + args.kappa_shared * d_bar),
                "diffmeans": u_dm,
                "centred": unit(v_hat - (v_hat @ mu_hat) * mu_hat),
                "purified": unit(v_hat - (v_hat @ d_bar) * d_bar),
                "diffmeans_purified": unit(u_dm - (u_dm @ d_bar) * d_bar),
                "learned": unit(v_hat + M @ v_hat),
            }
            for family, u in families.items():
                weak_energy = float(((u @ weak) ** 2).sum())
                rows.append(
                    {
                        "split": split,
                        "feature": f,
                        "family": family,
                        "cos_d_bar": float(u @ d_bar),
                        "cos_mu_hat": float(u @ mu_hat),
                        "cos_v_hat": float(u @ v_hat),
                        "weak_half_energy": weak_energy,
                        "logit_reach": float((u @ W_U).norm()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "direction_report.csv", index=False)

    cols = ["cos_d_bar", "cos_mu_hat", "cos_v_hat", "weak_half_energy", "logit_reach"]
    summary = df.groupby("family")[cols].mean().round(4)
    # keep a stable, story-order row order instead of groupby's alphabetical default
    order = ["decoder", "shared", "diffmeans", "centred", "purified", "diffmeans_purified", "learned"]
    summary = summary.reindex([f for f in order if f in summary.index])
    table = summary.to_string()

    (RESULTS / "direction_report.md").write_text(
        "# Direction report\n\n"
        f"Mean over {df['feature'].nunique()} features x {df['split'].nunique()} splits "
        f"({len(df)} rows), per direction family. kappa_shared={args.kappa_shared}, "
        f"ckpt={args.ckpt}.\n\n"
        "```\n" + table + "\n```\n",
        encoding="utf-8",
    )
    print(f"wrote {len(df)} rows to results/direction_report.csv\n")
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dir_hot", help="direction-correction checkpoint under checkpoints/ (no .pt)")
    ap.add_argument("--kappa-shared", dest="kappa_shared", type=float, default=0.75)
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args())
