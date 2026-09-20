"""Train the direction-conditional denoiser (ConditionalDenoiserArm's D) on counterfactual
concept pairs from src/concept_data.py.

Each step mixes two kinds of rows:
  - concept rows (1 - p_clean): h_minus (concept ablated via the SAE) + s*v_hat_f + noise ->
    h_plus (the real, concept-present activation). Teaches "turn a crude re-injection into a
    natural expression of the same concept in the same context."
  - clean rows (p_clean): a random corpus position, s = 0, input = target = h. Without this
    identity anchor the arm has no signal telling it to leave s = 0 alone, and would start
    adding the concept even at zero steering force. Each clean row is still given a real
    (randomly drawn, from the same mined pool) v_hat as its conditioning direction -- the only
    directions the model is ever conditioned on at inference are real feature directions, so
    the anchor is only informative if it is trained against them too, not against an arbitrary
    or zero vector.

Loss is MSE over the last axis normalized by scale^2, exactly as in src/train_denoiser.py.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common import CKPT, D_MODEL, DATA, DEVICE, RESULTS, ActStats, load_ceilings, load_sae, open_memmap, seed_all
from concept_data import forbidden_feature_indices
from denoiser_cond import MLPDenoiserCondDir

C_MIN, C_MAX = 0.3, 3.5
NOISE_REL = 0.05  # fraction of median_norm used as the Gaussian noise sigma on concept rows


def train(args) -> None:
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    stats = ActStats.load()
    scale = stats.median_norm
    acts_mm = open_memmap()
    n_total = acts_mm.shape[0]

    blob = np.load(DATA / "concept_positions.npz")
    features = blob["features"].astype(np.int64)
    positions = blob["positions"].astype(np.int64)
    acts_topk = blob["acts"].astype(np.float32)
    n_feat, top_k = positions.shape

    leaked = sorted(set(features.tolist()) & forbidden_feature_indices())
    if leaked:
        raise SystemExit(
            f"FATAL: forbidden (test/dev/test_r3) features present in concept_positions.npz: {leaked}"
        )

    sae = load_sae()
    ceilings = load_ceilings()
    features_t = torch.as_tensor(features, dtype=torch.long, device=DEVICE)
    wdec_all = sae.W_dec[features_t].detach().float()  # [n_feat, 768], raw decoder rows
    v_hat_arr = wdec_all / wdec_all.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    natural_s_arr = ceilings[features_t] * wdec_all.norm(dim=-1)  # [n_feat]
    del sae
    torch.cuda.empty_cache()

    model = MLPDenoiserCondDir(hidden=args.hidden, mean=stats.mean, scale=scale).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    # At least ~10 log points regardless of --steps, so a short smoke run still shows a curve.
    log_every = max(1, min(args.log_every, max(1, args.steps // 10)))

    t0 = time.time()
    log = []
    for step in range(args.steps):
        n_concept = int(round(args.batch * (1.0 - args.p_clean)))
        n_clean = args.batch - n_concept

        fi = rng.integers(0, n_feat, size=n_concept)
        ki = rng.integers(0, top_k, size=n_concept)
        fi_t = torch.from_numpy(fi).long().to(DEVICE)
        pos = positions[fi, ki]
        a_f = acts_topk[fi, ki]
        h_plus = torch.from_numpy(np.ascontiguousarray(acts_mm[pos])).to(DEVICE).float()
        a_f_t = torch.from_numpy(a_f).to(DEVICE).float()
        wdec_rows = wdec_all[fi_t]
        v_hat_rows = v_hat_arr[fi_t]
        nat_s_rows = natural_s_arr[fi_t]
        h_minus = h_plus - a_f_t.unsqueeze(-1) * wdec_rows

        r = torch.rand(n_concept, device=DEVICE, generator=gen)
        c = torch.exp(r * (np.log(C_MAX) - np.log(C_MIN)) + np.log(C_MIN))
        s_concept = c * nat_s_rows
        noise = torch.randn(n_concept, D_MODEL, device=DEVICE, generator=gen) * (NOISE_REL * scale)
        x_concept = h_minus + s_concept.unsqueeze(-1) * v_hat_rows + noise
        y_concept = h_plus

        clean_pos = rng.integers(0, n_total, size=n_clean)
        h_clean = torch.from_numpy(np.ascontiguousarray(acts_mm[clean_pos])).to(DEVICE).float()
        clean_fi = torch.from_numpy(rng.integers(0, n_feat, size=n_clean)).long().to(DEVICE)
        dir_clean = v_hat_arr[clean_fi]
        s_clean = torch.zeros(n_clean, device=DEVICE)

        x = torch.cat([x_concept, h_clean], dim=0)
        y = torch.cat([y_concept, h_clean], dim=0)
        v = torch.cat([v_hat_rows, dir_clean], dim=0)
        s = torch.cat([s_concept, s_clean], dim=0)

        pred = model(x, v, s)
        loss = ((pred - y) ** 2).sum(-1).mean() / (scale**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % log_every == 0 or step == 0:
            log.append({"step": step + 1, "loss": float(loss.detach())})
            print(f"step {step + 1}/{args.steps} loss {float(loss):.5f}", flush=True)

    name = args.name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "mlp_cond_dir",
            "hidden": args.hidden,
            "scale": scale,
            "cond": True,
            "steps": args.steps,
            "seed": args.seed,
            "n_params": n_params,
        },
        CKPT / f"{name}.pt",
    )
    out = {
        "name": name,
        "n_params": n_params,
        "minutes": round((time.time() - t0) / 60, 2),
        "train_log": log,
    }
    (RESULTS / f"train_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "train_log"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--p-clean", dest="p_clean", type=float, default=0.15)
    ap.add_argument("--log-every", dest="log_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="cond_mlp_s0")
    train(ap.parse_args())
