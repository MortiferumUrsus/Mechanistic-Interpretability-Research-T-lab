"""Round two: correct the steering direction instead of repairing the activation afterwards.

Round one established that the naive injection already has the lowest off-tangent damage of every arm
tried, and that each repair increased it. If that is true, no post-hoc repair can move the front, and
the only remaining lever is what gets injected.

So learn a direction correction rather than a repair. A shared low-rank map produces
`w(v) = normalise(v + M v)`, and the intervention becomes `h + s * w` — the same perturbation norm as
naive steering, only pointed somewhere slightly different. The objective is measured through the frozen
upper half of the network: keep as much of the concept's small-signal causal response as possible while
suppressing the nonlinear residue that shows up as broken text.

`M` is trained only on FIT feature directions and applied to unseen ones, so nothing about the
evaluation directions enters training.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import (
    CKPT,
    DATA,
    DEVICE,
    HOOK,
    RESULTS,
    SKIP_POS,
    ActStats,
    load_ceilings,
    load_features,
    load_model,
    load_sae,
    natural_strength,
    seed_all,
)
from train_ctr import make_suffix, token_windows


class DirectionCorrection(nn.Module):
    """w = normalise(v + A B^T v). Low rank keeps it a shared, transferable correction."""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        w = v + self.up(self.down(v))
        return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def train(args) -> None:
    seed_all(args.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    stats = ActStats.load()
    model = load_model()
    sae = load_sae()
    ceilings = load_ceilings()

    fit = np.load(DATA / "splits.npz")["fit"]
    # Round three needs features the correction has never seen, and there is no way to select them AFTER
    # training: `fit` was defined as every latent decorrelated from test and dev, so anything outside it is
    # by construction correlated with the very features rounds one and two used. Holding a slice of `fit`
    # out here is the only construction that gives round three both properties at once -- unseen by the
    # correction, and decorrelated from the earlier rounds. Excluding it is unconditional rather than a
    # flag, so a checkpoint trained without the holdout cannot be produced by accident.
    holdout_path = DATA / "splits_r3.npz"
    if holdout_path.exists():
        holdout = set(int(x) for x in np.load(holdout_path)["test_r3"])
        keep = np.array([f for f in fit if int(f) not in holdout], dtype=fit.dtype)
        if len(keep) == len(fit):
            raise SystemExit(
                f"{holdout_path.name} exists but none of its features are in `fit`; round three would not "
                "be held out from this checkpoint. Re-select it from `fit`."
            )
        print(f"fit pool {len(fit)} minus the round-three holdout {len(fit) - len(keep)} -> {len(keep)}")
        fit = keep
    else:
        print("no round-three holdout found; the correction trains on the whole fit pool")

    # Round four is 48 fresh features on disjoint prompts (see select_round4.py); the same
    # unconditional-exclusion argument as round three applies, plus a buffer: any remaining FIT
    # direction whose decoder cosine to ANY round-four direction is >= 0.3 is dropped too, the same
    # gate `features.fit_indices` applies for test/dev, so the pool stays decorrelated from the new
    # split and not just disjoint from it. Computed in chunks so the (|fit| x |test_r4|) cosine
    # matrix is never materialised in one shot.
    # The exclusion is opt-in (`--exclude-r4`): round two and round three checkpoints (dir_hot) were
    # trained before round four existed, and re-running their scripts must keep reproducing them even
    # though configs/features_r4.yaml is now present. The indices come from data/splits_r4.npz when
    # select_round4.py has just written it, and from configs/features_r4.yaml otherwise.
    r4_path = DATA / "splits_r4.npz"
    if args.exclude_r4:
        if r4_path.exists():
            r4_holdout = sorted(int(x) for x in np.load(r4_path)["test_r4"])
        else:
            r4_holdout = sorted(int(rec["index"]) for rec in load_features().get("test_r4", []))
            if not r4_holdout:
                raise SystemExit("--exclude-r4 given but neither data/splits_r4.npz nor configs/features_r4.yaml defines test_r4")
        keep = np.array([f for f in fit if int(f) not in r4_holdout], dtype=fit.dtype)
        if len(keep) == len(fit):
            raise SystemExit(
                f"{r4_path.name} exists but none of its features are in `fit`; round four would not "
                "be held out from this checkpoint. Re-select it from `fit`."
            )
        print(f"fit pool {len(fit)} minus the round-four holdout {len(fit) - len(keep)} -> {len(keep)}")

        w_dec = sae.W_dec.detach().float()
        vh = w_dec / w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r4_v = vh[torch.as_tensor(r4_holdout, device=DEVICE, dtype=torch.long)]
        keep_t = torch.as_tensor(keep, device=DEVICE, dtype=torch.long)
        keep_mask = torch.zeros(keep_t.shape[0], dtype=torch.bool, device=DEVICE)
        buffer_chunk = 2048
        for i in range(0, keep_t.shape[0], buffer_chunk):
            block = vh[keep_t[i : i + buffer_chunk]]
            cos = (block @ r4_v.T).abs().max(dim=1).values
            keep_mask[i : i + buffer_chunk] = cos < 0.3
        n_before = keep_t.shape[0]
        fit = keep_t[keep_mask].cpu().numpy().astype(fit.dtype)
        print(
            f"fit pool {n_before} minus {n_before - fit.shape[0]} within |cos| >= 0.3 of the "
            f"round-four set -> {fit.shape[0]} directions remain in the training pool"
        )
    else:
        print("round-four holdout not excluded (pass --exclude-r4 to hold out test_r4 and its cosine buffer)")

    dirs = sae.W_dec[torch.as_tensor(fit, device=DEVICE, dtype=torch.long)].detach().float()
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    natural = torch.tensor(
        [natural_strength(sae, ceilings, int(f)) for f in fit], device=DEVICE, dtype=torch.float32
    )
    del sae
    torch.cuda.empty_cache()

    windows = token_windows(model, args.n_docs, args.ctx).to(DEVICE)
    suffix = make_suffix(model)
    corr = DirectionCorrection(model.cfg.d_model, args.rank).to(DEVICE)
    n_params = sum(p.numel() for p in corr.parameters())
    opt = torch.optim.AdamW(corr.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    rng = np.random.default_rng(args.seed)
    t0, log = time.time(), []
    for step in range(args.steps):
        toks = windows[torch.as_tensor(rng.integers(0, windows.shape[0], size=args.batch), device=DEVICE)]
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=HOOK)
            h = cache[HOOK].detach()
            del cache
            j = int(torch.randint(0, dirs.shape[0], (1,), device=DEVICE, generator=gen))
            v = dirs[j]
            scale = float(natural[j])
            c = args.c_min + (args.c_max - args.c_min) * float(
                torch.rand(1, device=DEVICE, generator=gen)
            )
            s = c * scale
            z0 = suffix(h)[:, SKIP_POS:]
            g = (suffix(h + args.eps_c * scale * v)[:, SKIP_POS:] - z0) / (args.eps_c * scale)
            gn = (g * g).sum(-1, keepdim=True).clamp_min(1e-6)
            gnorm = g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # the naive arm measured in the same batch, so the objective is a matched comparison
            dz0 = suffix(h + s * v)[:, SKIP_POS:] - z0
            a0 = ((dz0 * g).sum(-1, keepdim=True) / gn).mean().clamp_min(1e-3)
            c0 = ((dz0 - ((dz0 * g).sum(-1, keepdim=True) / gn) * g).norm(dim=-1, keepdim=True) / gnorm).mean()

        w = corr(v.unsqueeze(0))[0]
        dz = suffix(h + s * w)[:, SKIP_POS:] - z0
        a = (dz * g).sum(-1, keepdim=True) / gn
        resid = dz - a * g
        c_val = (resid.norm(dim=-1, keepdim=True) / gnorm).mean()
        # Maximise the concept-aligned response relative to naive steering, penalised only for making
        # the off-tangent residue worse than naive. Without the matched reference the objective is
        # minimised by pointing somewhere harmless, which delivers no concept at all.
        loss = -(a.mean() / a0) + args.gamma * torch.relu(c_val / c0.clamp_min(1e-6) - 1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                cos = float(torch.nn.functional.cosine_similarity(w, v, dim=0))
                a_rel = float(a.mean() / a0)
                c_rel = float(c_val / c0.clamp_min(1e-6))
            log.append(
                {"step": step + 1, "loss": float(loss), "a_rel": a_rel, "c_rel": c_rel, "cos": cos}
            )
            print(
                f"step {step + 1}/{args.steps} loss {float(loss):.4f} "
                f"A/A_naive {a_rel:.3f} C/C_naive {c_rel:.3f} cos(w,v) {cos:.4f}",
                flush=True,
            )

    name = args.name or f"dir_r{args.rank}_g{args.gamma}_s{args.seed}"
    torch.save(
        {
            "state_dict": corr.state_dict(),
            "rank": args.rank,
            "gamma": args.gamma,
            "eps_c": args.eps_c,
            "c_range": [args.c_min, args.c_max],
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


@torch.no_grad()
def evaluate(args) -> None:
    """Measure the correction on held-out directions, averaged over features and strengths.

    Per-step training numbers are single-feature samples and cannot show convergence; this is the
    quantity the selection is made on.
    """
    import pandas as pd
    import yaml

    from common import ROOT

    stats = ActStats.load()
    model = load_model()
    sae = load_sae()
    ceilings = load_ceilings()
    suffix = make_suffix(model)
    prompts = json.loads((DATA / "prompts.json").read_text(encoding="utf-8"))[: args.n_prompts]
    toks = model.to_tokens(prompts).to(DEVICE)
    _, cache = model.run_with_cache(toks, names_filter=HOOK)
    h = cache[HOOK].detach()
    del cache
    z0 = suffix(h)[:, SKIP_POS:]

    recs = load_features(args.split)
    rows = []
    for name in args.checkpoints.split(","):
        corr = load_correction(CKPT / f"{name}.pt")
        for r in recs:
            f = int(r["index"])
            v = sae.W_dec[f].detach().float()
            v = v / v.norm()
            scale = natural_strength(sae, ceilings, f)
            w = corr(v.unsqueeze(0))[0]
            g = (suffix(h + args.eps_c * scale * v)[:, SKIP_POS:] - z0) / (args.eps_c * scale)
            gn = (g * g).sum(-1, keepdim=True).clamp_min(1e-6)
            gnorm = g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            for c in [0.5, 1.0, 1.5, 2.0, 2.5]:
                s = c * scale
                out = {}
                for tag, direction in (("naive", v), ("dirfix", w)):
                    dz = suffix(h + s * direction)[:, SKIP_POS:] - z0
                    a = (dz * g).sum(-1, keepdim=True) / gn
                    resid = dz - a * g
                    out[tag] = (
                        float(a.mean()),
                        float((resid.norm(dim=-1, keepdim=True) / gnorm).mean()),
                    )
                rows.append(
                    {
                        "checkpoint": name,
                        "feature": f,
                        "c": c,
                        "cos_w_v": float(torch.nn.functional.cosine_similarity(w, v, dim=0)),
                        "a_naive": out["naive"][0],
                        "a_dirfix": out["dirfix"][0],
                        "c_naive": out["naive"][1],
                        "c_dirfix": out["dirfix"][1],
                        "a_ratio": out["dirfix"][0] / max(out["naive"][0], 1e-6),
                        "c_ratio": out["dirfix"][1] / max(out["naive"][1], 1e-6),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"dirfix_eval_{args.split}.csv", index=False)
    print(
        df.groupby(["checkpoint", "c"])[["a_ratio", "c_ratio", "cos_w_v"]].mean().round(3).to_string()
    )
    print("\nper checkpoint, averaged over features and strengths:")
    print(df.groupby("checkpoint")[["a_ratio", "c_ratio", "cos_w_v"]].agg(["mean", "std"]).round(3).to_string())


def load_correction(path):
    blob = torch.load(path, map_location=DEVICE)
    corr = DirectionCorrection(768, blob["rank"]).to(DEVICE)
    corr.load_state_dict(blob["state_dict"])
    corr.eval()
    return corr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="train", choices=["train", "eval"])
    ap.add_argument("--checkpoints", default="dir_hot")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--n-docs", dest="n_docs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eps-c", dest="eps_c", type=float, default=0.25)
    ap.add_argument("--c-min", dest="c_min", type=float, default=0.5)
    ap.add_argument("--c-max", dest="c_max", type=float, default=2.5)
    ap.add_argument("--log-every", dest="log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--exclude-r4",
        dest="exclude_r4",
        action="store_true",
        help="hold out the round-four features (configs/features_r4.yaml or data/splits_r4.npz) and their "
        "|cos| >= 0.3 buffer from the training pool; used to train dir_r4",
    )
    a = ap.parse_args()
    if a.stage == "eval":
        evaluate(a)
    else:
        train(a)
