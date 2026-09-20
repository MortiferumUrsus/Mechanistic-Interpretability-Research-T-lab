"""CTR: train the repair against the frozen upper half instead of an activation-space proxy.

Activation MSE is a proxy for what we actually care about: that layers 7..11 keep computing
something sane while the concept still gets delivered. Both are differentiable through the
frozen suffix, so they can be optimised directly.

For a clean activation h and a direction u, the small-signal causal response of the final
logits is g = (F(h + eps u) - F(h)) / eps. The part of a large-alpha response that is aligned
with g is declared useful; the off-tangent residue is declared damage. The teacher target is
therefore z* = F(h) + a * g with a the aligned amplitude, and the repair is trained to make
F(h_tilde) match it.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, DATA, DEVICE, HOOK, RESULTS, ActStats, load_model, load_sae, seed_all

from common import HOOK as _HOOK
LAYER = int(_HOOK.split(".")[1])  # intervention after this block; the frozen suffix is the blocks after it


class Repair(nn.Module):
    """Residual repair conditioned on the steering direction and magnitude.

    The output is projected orthogonally to u, so the concept coordinate u^T h_tilde equals the
    naive steering value by construction and the arm cannot win by quietly reducing the signal.
    """

    def __init__(self, d: int, hidden: int, mean: torch.Tensor, scale: float):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.dir_proj = nn.Linear(d, hidden, bias=False)
        self.mag_proj = nn.Linear(1, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d)
        for m in (self.fc2, self.dir_proj, self.mag_proj):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        while u.dim() < x.dim():
            u = u.unsqueeze(0)
        if s.dim() < x.dim():
            s = s.view(*s.shape, *([1] * (x.dim() - s.dim())))
        z = self.fc1(self.norm((x - self.mean) / self.scale))
        z = z + self.dir_proj(u) + self.mag_proj(s / self.scale)
        out = self.fc2(F.gelu(z)) * self.scale
        return out - (out * u).sum(-1, keepdim=True) * u


def make_suffix(model):
    blocks = model.blocks[LAYER + 1 :]

    def suffix(resid: torch.Tensor) -> torch.Tensor:
        for blk in blocks:
            resid = blk(resid)
        logits = model.unembed(model.ln_final(resid))
        return logits - logits.mean(-1, keepdim=True)

    return suffix


def token_windows(model, n_docs: int, ctx: int):
    from datasets import load_dataset

    from activations import ACT_SHARD

    ds = load_dataset("parquet", data_files=ACT_SHARD, split="train")
    out = []
    for i in range(n_docs):
        t = ds[i]["text"]
        if len(t) < 600:
            continue
        toks = model.to_tokens(t[:4000], truncate=True)[0, :ctx]
        if toks.shape[0] == ctx:
            out.append(toks)
    return torch.stack(out)


def train(args) -> None:
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    stats = ActStats.load()
    scale = stats.median_norm
    model = load_model()
    sae = load_sae()
    fit = np.load(DATA / "splits.npz")["fit"]
    dictionary = sae.W_dec[torch.as_tensor(fit, device=DEVICE, dtype=torch.long)].detach().float()
    dictionary = dictionary / dictionary.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    del sae
    torch.cuda.empty_cache()

    windows = token_windows(model, args.n_docs, args.ctx).to(DEVICE)
    suffix = make_suffix(model)
    repair = Repair(model.cfg.d_model, args.hidden, stats.mean, scale).to(DEVICE)
    n_params = sum(p.numel() for p in repair.parameters())
    opt = torch.optim.AdamW(repair.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    rng = np.random.default_rng(args.seed)
    eps = args.eps_c * scale
    t0, log = time.time(), []

    for step in range(args.steps):
        sel = rng.integers(0, windows.shape[0], size=args.batch)
        toks = windows[torch.as_tensor(sel, device=DEVICE)]
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=HOOK)
            h = cache[HOOK].detach()
            del cache
            u = dictionary[torch.randint(0, dictionary.shape[0], (1,), device=DEVICE, generator=gen)][0]
            c = args.c_min + (args.c_max - args.c_min) * torch.rand(1, device=DEVICE, generator=gen)
            s = (c * scale).expand(h.shape[0])
            z0 = suffix(h)
            g = (suffix(h + eps * u) - z0) / eps
            zb = suffix(h + s.view(-1, 1, 1) * u)
            gn = (g * g).sum(-1, keepdim=True).clamp_min(1e-6)
            a = ((zb - z0) * g).sum(-1, keepdim=True) / gn
            a = a.clamp(0.0, args.a_max)
            target = z0 + a * g

        x = h + s.view(-1, 1, 1) * u
        h_tilde = x + repair(x, u, s)
        z = suffix(h_tilde)
        t = args.temp
        kl = F.kl_div(
            F.log_softmax(z / t, dim=-1), F.log_softmax(target / t, dim=-1),
            reduction="batchmean", log_target=True,
        ) * (t * t)
        zero = repair(h, u, torch.zeros_like(s))
        loss = kl + args.xi * (zero.pow(2).sum(-1).mean() / scale**2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % args.log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "kl": float(kl), "c": float(c)})
            print(f"step {step + 1}/{args.steps} loss {float(loss):.4f} kl {float(kl):.4f}", flush=True)

    name = args.name or f"ctr_h{args.hidden}_s{args.seed}"
    torch.save(
        {
            "state_dict": repair.state_dict(),
            "hidden": args.hidden,
            "scale": scale,
            "eps_c": args.eps_c,
            "c_range": [args.c_min, args.c_max],
            "a_max": args.a_max,
            "temp": args.temp,
            "xi": args.xi,
            "seed": args.seed,
            "steps": args.steps,
            "n_params": n_params,
        },
        CKPT / f"{name}.pt",
    )
    (RESULTS / f"train_{name}.json").write_text(
        json.dumps(
            {"name": name, "n_params": n_params, "minutes": round((time.time() - t0) / 60, 2), "log": log},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {name}, {n_params} params, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--n-docs", dest="n_docs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eps-c", dest="eps_c", type=float, default=0.25)
    ap.add_argument("--c-min", dest="c_min", type=float, default=0.5)
    ap.add_argument("--c-max", dest="c_max", type=float, default=3.5)
    ap.add_argument("--a-max", dest="a_max", type=float, default=8.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    train(ap.parse_args())
