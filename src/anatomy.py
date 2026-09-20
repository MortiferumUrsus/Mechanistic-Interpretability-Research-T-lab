"""Anatomy of the learned direction correction w(v) = normalize(v + M v), M = up.weight @ down.weight.

Round two trained a shared low-rank correction on FIT directions (see train_direction.py). This
script asks what that correction actually does to the population of directions it was trained on:
whether it pushes every feature towards one shared direction, how much of the correction cloud's
variance is explained by a handful of directions, how it sits relative to the activation
covariance, and what it does to the quadratic forms v^T Sigma^-1 v / v^T Sigma v that other arms
in this study care about.

The per-feature aggregates (cosines, variance shares, singular values of M, participation ratio,
quadratic-form means, PC5 mass) are computed on the first 2000 FIT directions -- the same sample
size the exploratory version of this analysis used, kept fixed here for reproducibility. The
shared direction `d_bar` saved to checkpoints/shared_direction.pt is a mean over ALL FIT directions
(chunked, so the full 24000-direction pool never needs to live in memory as one batch of deltas).

    python anatomy.py [--ckpt dir_hot] [--device cpu] [--tag _ent]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from common import CKPT, DATA, RESULTS, ROOT, SHARED_DIRECTION, ActStats, load_features, load_sae

N_COS_SAMPLE = 2000  # size of the FIT sample used for the aggregate/pairwise-cosine analysis
CHUNK = 2048  # chunk size for the full-FIT mean-delta pass
N_PC = 5  # top-k covariance eigen-directions for the "mass" statistic
SHRINK = 0.01  # shrinkage used for the quadratic-form covariance


def unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-9)


def load_M(ckpt: str, device: str) -> tuple[torch.Tensor, dict]:
    blob = torch.load(CKPT / f"{ckpt}.pt", map_location=device)
    sd = blob["state_dict"]
    M = sd["up.weight"].float() @ sd["down.weight"].float()
    return M, blob


def deltas(W_dec: torch.Tensor, M: torch.Tensor, idxs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """V, W, D (unit vectors) for the given SAE latent indices: v, w(v), and normalize(w - v)."""
    V = unit(W_dec[list(idxs)])
    W = unit(V + V @ M.T)
    D = unit(W - V)
    return V, W, D


def quad_forms(X: torch.Tensor, sigma64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row x^T Sigma^-1 x and x^T Sigma x, solved in float64."""
    xd = X.double()
    sol = torch.linalg.solve(sigma64, xd.T)  # [d, n]
    quad_inv = (xd * sol.T).sum(-1)
    quad_fwd = (xd @ sigma64 * xd).sum(-1)
    return quad_inv.float(), quad_fwd.float()


def pc_mass(X: torch.Tensor, evecs_top: torch.Tensor) -> torch.Tensor:
    """Per-row fraction of squared norm lying in the span of `evecs_top` (X rows are unit norm)."""
    proj = X @ evecs_top
    return (proj**2).sum(-1)


def main(args) -> None:
    device = args.device
    sae = load_sae(device=device)
    W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
    stats = ActStats.load(device=device)

    M, ckpt_blob = load_M(args.ckpt, device)

    evals, evecs = torch.linalg.eigh(stats.cov.to(device).float())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    evecs_top = evecs[:, :N_PC]

    U, S, Vt = torch.linalg.svd(M)
    frob = float(torch.linalg.matrix_norm(M, ord="fro"))
    Sd = S.double()
    participation_ratio = float((Sd.sum() ** 2) / (Sd**2).sum())

    fit = np.load(DATA / "splits.npz")["fit"]

    # --- Aggregate analysis on the first N_COS_SAMPLE FIT directions ---
    sample = fit[:N_COS_SAMPLE]
    Vf, Wf, Df = deltas(W_dec, M, sample)

    G = Df @ Df.T
    off_mask = ~torch.eye(len(Df), dtype=torch.bool, device=Df.device)
    off = G[off_mask]
    mean_pairwise_cos = float(off.mean())
    median_pairwise_cos = float(off.median())

    mean_delta = Df.mean(0)
    delta_norm_ratio = float(mean_delta.norm() / Df.norm(dim=-1).mean())

    _, Ss, _ = torch.linalg.svd(Df - Df.mean(0, keepdim=True), full_matrices=False)
    ev = (Ss**2 / (Ss**2).sum()).double()
    var_top1 = float(ev[0])
    var_top5 = float(ev[:5].sum())
    var_top20 = float(ev[:20].sum())

    sigma64 = stats.shrunk_cov(SHRINK).to(device).double()
    quad_inv_v, quad_v = quad_forms(Vf, sigma64)
    quad_inv_w, quad_w = quad_forms(Wf, sigma64)

    # The inverse form is dominated by the smallest eigenvalues, so "the correction injects along
    # low-variance directions" is only a claim if it survives the regularisation that makes the
    # inverse well defined at all. Report the whole shrinkage curve rather than one number: an
    # effect that only exists at one shrinkage is an artefact of that shrinkage.
    shrink_curve = []
    for sh in (0.001, 0.01, 0.05, 0.2):
        sig = stats.shrunk_cov(sh).to(device).double()
        qi_v, qf_v = quad_forms(Vf, sig)
        qi_w, qf_w = quad_forms(Wf, sig)
        shrink_curve.append(
            {
                "shrink": sh,
                "quad_inv_v": float(qi_v.mean()),
                "quad_inv_w": float(qi_w.mean()),
                "quad_inv_ratio": float(qi_w.mean() / qi_v.mean()),
                "quad_v": float(qf_v.mean()),
                "quad_w": float(qf_w.mean()),
                "quad_fwd_ratio": float(qf_w.mean() / qf_v.mean()),
            }
        )
        del sig
    mass_v = pc_mass(Vf, evecs_top)
    mass_w = pc_mass(Wf, evecs_top)

    summary = {
        "ckpt": args.ckpt,
        "rank": ckpt_blob.get("rank"),
        "n_cos_sample": int(len(sample)),
        "mean_pairwise_cos": mean_pairwise_cos,
        "median_pairwise_cos": median_pairwise_cos,
        "mean_delta_norm_over_mean_norm": delta_norm_ratio,
        "delta_cloud_variance_share": {"top1": var_top1, "top5": var_top5, "top20": var_top20},
        "M_singular_values_top10": [float(x) for x in S[:10]],
        "M_frobenius_norm": frob,
        "M_participation_ratio": participation_ratio,
        "quad_inv_v_mean": float(quad_inv_v.mean()),
        "quad_v_mean": float(quad_v.mean()),
        "quad_inv_w_mean": float(quad_inv_w.mean()),
        "quad_shrinkage_curve": shrink_curve,
        "quad_w_mean": float(quad_w.mean()),
        "pc5_mass_v_mean": float(mass_v.mean()),
        "pc5_mass_w_mean": float(mass_w.mean()),
        "shrink": SHRINK,
        "n_pc": N_PC,
    }
    (RESULTS / f"anatomy_summary{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # --- Shared direction d_bar: mean delta over ALL FIT directions, chunked ---
    t0 = time.time()
    d_sum = torch.zeros(W_dec.shape[1], device=device, dtype=torch.float32)
    n_fit = int(len(fit))
    for i in range(0, n_fit, CHUNK):
        _, _, D_chunk = deltas(W_dec, M, fit[i : i + CHUNK])
        d_sum += D_chunk.sum(0)
    d_bar = unit(d_sum / n_fit)
    shared_path = SHARED_DIRECTION.with_name(f"shared_direction{args.tag}.pt")
    torch.save({"d_bar": d_bar.cpu(), "source": args.ckpt, "n_fit": n_fit}, shared_path)
    print(f"d_bar over {n_fit} FIT directions in {time.time() - t0:.1f}s -> {shared_path.relative_to(ROOT)}")

    # --- Per-feature CSV over the evaluation pools ---
    feats = load_features()
    rows = []
    for split in ("test", "dev", "test_r3", "test_r4"):
        idxs = [int(rec["index"]) for rec in feats.get(split, [])]
        if not idxs:
            continue
        V, W, _ = deltas(W_dec, M, idxs)
        cos_w_v = (W * V).sum(-1)
        cos_v_dbar = V @ d_bar
        cos_w_dbar = W @ d_bar
        qinv_v, _ = quad_forms(V, sigma64)
        qinv_w, _ = quad_forms(W, sigma64)
        m_v = pc_mass(V, evecs_top)
        m_w = pc_mass(W, evecs_top)
        for k, f in enumerate(idxs):
            rows.append(
                {
                    "split": split,
                    "feature": f,
                    "cos_w_v": float(cos_w_v[k]),
                    "cos_v_dbar": float(cos_v_dbar[k]),
                    "cos_w_dbar": float(cos_w_dbar[k]),
                    "quad_inv_v": float(qinv_v[k]),
                    "quad_inv_w": float(qinv_w[k]),
                    "pc5_mass_v": float(m_v[k]),
                    "pc5_mass_w": float(m_w[k]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"anatomy_per_feature{args.tag}.csv", index=False)
    print(f"wrote {len(df)} rows to results/anatomy_per_feature{args.tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dir_hot", help="checkpoint name under checkpoints/ (no .pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for every output (anatomy_summary<tag>.json, anatomy_per_feature<tag>.csv, "
        "shared_direction<tag>.pt) so a second checkpoint's anatomy does not overwrite the first",
    )
    main(ap.parse_args())
