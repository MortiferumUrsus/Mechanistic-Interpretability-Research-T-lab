"""Counterfactual concept pairs for the conditional denoiser (Experiment G).

The idea the whole experiment rests on: a denoiser trained to pull activations toward the
*general* corpus manifold also erases the steering vector, because the steering vector is
itself off-manifold. Pulling instead toward the manifold of "activations where this concept
is naturally expressed" should not erase the concept, because that manifold contains it.

There are no natural (corrupted -> correct) pairs for that, so they are built through the SAE
itself. For a latent f and a corpus position where f fires hard:

    h_plus  = the real activation                              (concept naturally present)
    a_f     = the SAE's own encoder activation of f there (ReLU'd)
    h_minus = h_plus - a_f * sae.W_dec[f]                       (same context, concept ablated)

`h_minus + s * v_hat_f` is a crude re-injection of the concept into the same context; `h_plus`
is what a natural expression of it looks like. That is the training pair.

Two modes:
    mine   -- scan the activation dump once, per sampled FIT latent keep the top-k positions
              where it fires hardest, save positions/activations (data/concept_positions.npz).
    stats  -- from those positions, compute the closed-form ingredients for the (untrained)
              ConditionalWienerArm: per-feature mu_f and one shared conditional covariance
              (data/concept_stats.pt). Also mines mu_f for the held-out eval features
              (configs/features.yaml: test, dev, test_r3) -- needed at inference, never used
              to build a training pair.

Every latent in configs/features.yaml (test, dev, test_r3) and in splits.npz's test/dev, and
splits_r3.npz's test_r3, is excluded from the `mine` pool. splits.npz's own "fit" array
predates round three and still contains test_r3 indices, so test_r3 is excluded explicitly
here rather than trusted to "fit" alone.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from tqdm import tqdm

from common import D_MODEL, DATA, DEVICE, ROOT, load_features, load_sae, open_memmap, seed_all

CONFIGS = ROOT / "configs"


def _yaml_indices(key: str) -> set[int]:
    feats = load_features()
    return {int(rec["index"]) for rec in feats.get(key, [])}


def forbidden_feature_indices() -> set[int]:
    """Every latent that must never contribute a training pair.

    Union of configs/features.yaml's test/dev/test_r3 (the authoritative eval lists) with
    splits.npz's test/dev and splits_r3.npz's test_r3, so a mismatch between the two sources
    fails closed rather than silently leaking a held-out latent into training.
    """
    forbidden = _yaml_indices("test") | _yaml_indices("dev") | _yaml_indices("test_r3")
    splits = np.load(DATA / "splits.npz")
    forbidden |= {int(i) for i in splits["test"]} | {int(i) for i in splits["dev"]}
    r3 = np.load(DATA / "splits_r3.npz")
    forbidden |= {int(i) for i in r3["test_r3"]}
    return forbidden


def eval_feature_indices() -> list[int]:
    """Latents that need mu_f at inference time (configs/features.yaml test+dev+test_r3)."""
    return sorted(_yaml_indices("test") | _yaml_indices("dev") | _yaml_indices("test_r3"))


def _fit_pool() -> np.ndarray:
    fit = np.load(DATA / "splits.npz")["fit"]
    forbidden = forbidden_feature_indices()
    keep = np.array([f for f in fit.tolist() if f not in forbidden], dtype=np.int64)
    return keep


@torch.no_grad()
def _mine_topk(
    sae, acts_mm, feature_idx: torch.Tensor, top_k: int, chunk: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunked pass over the corpus (same pattern as src/sae_stats.py), running top-k.

    Returns (positions[n_feat, top_k] int64, acts[n_feat, top_k] float32) on CPU. A position
    of -1 with activation 0 means the latent never reached top_k firing positions at all in
    the corpus (surfaced by the `mine` diagnostic, not treated as an error).
    """
    n_feat = feature_idx.numel()
    n_tok = acts_mm.shape[0]
    best_acts = torch.zeros(n_feat, top_k, device=device)
    best_pos = torch.full((n_feat, top_k), -1, dtype=torch.long, device=device)

    for i in tqdm(range(0, n_tok, chunk), unit="chunk"):
        h = torch.from_numpy(np.ascontiguousarray(acts_mm[i : i + chunk])).to(device).float()
        z = sae.encode(h)  # [c, d_sae], post-ReLU
        z_sub = z[:, feature_idx]  # [c, n_feat]
        c = z_sub.shape[0]
        k = min(top_k, c)
        chunk_acts, chunk_idx = torch.topk(z_sub, k, dim=0)  # [k, n_feat]
        chunk_pos = chunk_idx.to(torch.long) + i
        cat_acts = torch.cat([best_acts, chunk_acts.T], dim=1)
        cat_pos = torch.cat([best_pos, chunk_pos.T], dim=1)
        top_acts, top_idx = torch.topk(cat_acts, top_k, dim=1)
        best_acts = top_acts
        best_pos = torch.gather(cat_pos, 1, top_idx)
        del h, z, z_sub

    return best_pos.cpu(), best_acts.cpu()


def mine(args) -> None:
    seed_all(args.seed)
    sae = load_sae()
    acts_mm = open_memmap()

    pool = _fit_pool()
    forbidden = forbidden_feature_indices()
    rng = np.random.default_rng(args.seed)
    n_pick = min(args.n_features, len(pool))
    chosen = np.sort(rng.choice(pool, size=n_pick, replace=False))
    leaked = sorted(set(chosen.tolist()) & forbidden)
    if leaked:
        raise SystemExit(f"FATAL: forbidden features leaked into the mining pool: {leaked}")

    feature_idx = torch.as_tensor(chosen, dtype=torch.long, device=DEVICE)
    positions, acts_topk = _mine_topk(sae, acts_mm, feature_idx, args.top_k, args.chunk, DEVICE)

    np.savez(
        DATA / "concept_positions.npz",
        features=chosen.astype(np.int32),
        positions=positions.numpy().astype(np.int64),
        acts=acts_topk.numpy().astype(np.float32),
    )

    acts_np = acts_topk.numpy()
    n_full = int((acts_np.min(axis=1) > 0).sum())
    median_act = float(np.median(acts_np))
    print(
        f"mined {n_pick} features (pool={len(pool)}, forbidden excluded={len(forbidden)}), "
        f"top_k={args.top_k}: {n_full}/{n_pick} features have all {args.top_k} positions "
        f"active (>0), median stored activation = {median_act:.4f}"
    )
    print(f"wrote {DATA / 'concept_positions.npz'}")


def stats(args) -> None:
    blob = np.load(DATA / "concept_positions.npz")
    features = blob["features"]
    positions = blob["positions"]
    top_k = positions.shape[1]
    n = len(features)
    acts_mm = open_memmap()

    mu = torch.zeros(n, D_MODEL)
    cov_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    for i in tqdm(range(n), unit="feature", desc="mu/cov"):
        pos_i = positions[i]
        valid = pos_i >= 0
        pos_i = pos_i[valid]
        if pos_i.size == 0:
            continue  # dead-ish latent, never reached a single positive activation; mu stays 0
        h_plus = torch.from_numpy(np.ascontiguousarray(acts_mm[pos_i])).float()  # [k', 768]
        mu_i = h_plus.mean(0)
        mu[i] = mu_i
        centered = h_plus - mu_i
        denom = max(pos_i.size - 1, 1)
        cov_sum += ((centered.T @ centered) / denom).double()
    cov = (cov_sum / n).float()

    sae = load_sae()
    eval_idx = eval_feature_indices()
    eval_feature_t = torch.as_tensor(eval_idx, dtype=torch.long, device=DEVICE)
    eval_positions, _ = _mine_topk(sae, acts_mm, eval_feature_t, top_k, args.chunk, DEVICE)
    eval_mu = torch.zeros(len(eval_idx), D_MODEL)
    for i in range(len(eval_idx)):
        pos_i = eval_positions[i].numpy()
        pos_i = pos_i[pos_i >= 0]
        if pos_i.size == 0:
            continue
        h_plus = torch.from_numpy(np.ascontiguousarray(acts_mm[pos_i])).float()
        eval_mu[i] = h_plus.mean(0)

    torch.save(
        {
            "features": torch.as_tensor(features.astype(np.int64), dtype=torch.long),
            "mu": mu,
            "cov": cov,
            "top_k": top_k,
            "eval_features": torch.as_tensor(np.array(eval_idx, dtype=np.int64), dtype=torch.long),
            "eval_mu": eval_mu,
        },
        DATA / "concept_stats.pt",
    )
    print(
        f"saved {DATA / 'concept_stats.pt'}: n_features={n}, top_k={top_k}, "
        f"eval_features={len(eval_idx)}, cov_trace={float(torch.diagonal(cov).sum()):.2f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    mine_p = sub.add_parser("mine")
    mine_p.add_argument("--n-features", dest="n_features", type=int, default=4000)
    mine_p.add_argument("--top-k", dest="top_k", type=int, default=128)
    mine_p.add_argument("--chunk", type=int, default=4096)
    mine_p.add_argument("--seed", type=int, default=0)

    stats_p = sub.add_parser("stats")
    stats_p.add_argument("--chunk", type=int, default=4096, help="chunk size for the eval-feature scan")

    a = ap.parse_args()
    if a.mode == "mine":
        mine(a)
    else:
        stats(a)
