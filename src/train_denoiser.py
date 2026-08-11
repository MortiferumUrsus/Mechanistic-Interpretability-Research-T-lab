"""Train a denoiser D with L = ||h - D(h + s*u)||^2 on FIT directions only."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from common import CKPT, DATA, DEVICE, RESULTS, ActStats, load_sae, open_memmap, seed_all
from denoiser import build_denoiser
from noise import NoiseConfig, perturb

HOLDOUT = 50_000


def fit_dictionary(device: str = DEVICE) -> torch.Tensor:
    sae = load_sae(device)
    fit = np.load(DATA / "splits.npz")["fit"]
    w = sae.W_dec[torch.as_tensor(fit, device=device, dtype=torch.long)].detach().float()
    del sae
    torch.cuda.empty_cache()
    return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def train(args) -> None:
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    stats = ActStats.load()
    acts = open_memmap()
    n_total = acts.shape[0]
    n_train = n_total - HOLDOUT
    scale = stats.median_norm

    dictionary = fit_dictionary()
    cfg = NoiseConfig(
        kind=args.noise,
        p_dict=args.p_dict,
        s_min_rel=args.s_min_rel,
        s_max_rel=args.s_max_rel,
        p_zero=args.p_zero,
    )
    model = build_denoiser(
        args.arch, stats, hidden=args.hidden, scale=scale, cond=bool(args.cond)
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    log = []
    for step in range(args.steps):
        idx = np.sort(rng.integers(0, n_train, size=args.batch))
        h = torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEVICE).float()
        x, s = perturb(h, dictionary, scale, cfg, gen)
        pred = model(x, s if args.cond else None)
        loss = ((pred - h) ** 2).sum(-1).mean() / (scale**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.log_every == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"step {step + 1}/{args.steps} loss {float(loss):.5f}", flush=True)

    metrics = evaluate(model, acts, n_train, dictionary, stats, cfg, args, gen)
    name = args.name or f"{args.arch}_{args.noise}_cond{int(bool(args.cond))}_s{args.seed}"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "hidden": args.hidden,
            "cond": bool(args.cond),
            "scale": scale,
            "noise": asdict(cfg),
            "seed": args.seed,
            "steps": args.steps,
            "n_params": n_params,
            "metrics": metrics,
        },
        CKPT / f"{name}.pt",
    )
    out = {
        "name": name,
        "n_params": n_params,
        "minutes": round((time.time() - t0) / 60, 2),
        "train_log": log,
        **metrics,
    }
    (RESULTS / f"train_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "train_log"}, indent=2))


@torch.no_grad()
def evaluate(model, acts, n_train, dictionary, stats, cfg, args, gen) -> dict:
    """Held-out reconstruction error, reported per perturbation magnitude."""
    model.eval()
    scale = stats.median_norm
    rng = np.random.default_rng(args.seed + 1)
    idx = np.sort(rng.choice(np.arange(n_train, acts.shape[0]), size=8192, replace=False))
    h = torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEVICE).float()
    res = {}
    for c in [0.0, 0.5, 1.0, 2.0, 3.0]:
        u = dictionary[torch.randint(0, dictionary.shape[0], (h.shape[0],), device=DEVICE, generator=gen)]
        s = torch.full((h.shape[0],), c * scale, device=DEVICE)
        x = h + s.unsqueeze(-1) * u
        pred = model(x, s if args.cond else None)
        res[f"nrmse_c{c}"] = round(float(((pred - h).norm(dim=-1) / scale).mean()), 5)
        res[f"nrmse_raw_c{c}"] = round(float(((x - h).norm(dim=-1) / scale).mean()), 5)
    model.train()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--noise", default="mix", choices=["gauss", "dict", "mix"])
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--cond", type=int, default=1)
    ap.add_argument("--p-dict", dest="p_dict", type=float, default=0.5)
    ap.add_argument("--s-min-rel", dest="s_min_rel", type=float, default=0.05)
    ap.add_argument("--s-max-rel", dest="s_max_rel", type=float, default=3.5)
    ap.add_argument("--p-zero", dest="p_zero", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--log-every", dest="log_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    train(ap.parse_args())
