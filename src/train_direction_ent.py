"""Round two, entropy-penalised: same direction-correction objective as train_direction.py, but
augmented with a stability penalty on the suffix's output entropy.

Round two's objective (-A/A_naive + gamma * relu(C/C_naive - 1)) never looks at the confidence of
the steered distribution -- only its alignment with the naive small-signal response and the
off-tangent residue. An objective blind to entropy would not resist a correction that collapses
onto a shared "confidence handle" direction: crushing next-token entropy is a cheap way to inflate
a coherent-looking response, and the hypothesis under test is that this is exactly what happened.
This script adds

    + ent_weight * mean |H(logits_steered) - H(logits_clean)|

to the loss, measured through the same frozen suffix used for A and C, at the same positions, to
see whether the shared component the correction converges on disappears once confidence drift is
penalised. logits_clean is the suffix's output on the untouched activation h (no injection);
logits_steered is its output on h + s * w(v). H is Shannon entropy (nats) of softmax over the
vocabulary, computed via log_softmax for numerical stability.

Everything else -- holdout exclusion (round three and round four, with the cosine buffer),
natural-strength scaling, direction/strength sampling, logging and checkpoint format -- is
unchanged from train_direction.py; DirectionCorrection, load_correction and evaluate are imported
from there rather than redefined. train() is a copy of train_direction.py's train() because the
loss step is threaded through a single local scope that cannot be overridden piecemeal; the lines
that differ from the original are marked "# ENT:".
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

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
from train_direction import DirectionCorrection, evaluate, load_correction  # noqa: F401  (re-exported)


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
            # ENT: entropy of the clean (non-injected) suffix output, same positions as A/C, no
            # ENT: grad needed -- z0 is already computed above without injection, under no_grad.
            logp_clean = F.log_softmax(z0, dim=-1)
            h_clean = -(logp_clean.exp() * logp_clean).sum(-1)

        w = corr(v.unsqueeze(0))[0]
        # ENT: named so it also serves as logits_steered for the entropy penalty below.
        z_s = suffix(h + s * w)[:, SKIP_POS:]
        dz = z_s - z0
        a = (dz * g).sum(-1, keepdim=True) / gn
        resid = dz - a * g
        c_val = (resid.norm(dim=-1, keepdim=True) / gnorm).mean()
        # ENT: entropy of the steered suffix output, same frozen suffix, same positions, with grad.
        logp_steered = F.log_softmax(z_s, dim=-1)
        h_steered = -(logp_steered.exp() * logp_steered).sum(-1)
        ent_diff = (h_steered - h_clean).abs().mean()
        # Maximise the concept-aligned response relative to naive steering, penalised for making
        # the off-tangent residue worse than naive, and for drifting the suffix's confidence away
        # from what the untouched activation would have produced.
        loss = (
            -(a.mean() / a0)
            + args.gamma * torch.relu(c_val / c0.clamp_min(1e-6) - 1.0)
            + args.ent_weight * ent_diff  # ENT: the added stability penalty
        )

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
                ent_val = float(ent_diff)  # ENT: logged alongside the existing metrics
            log.append(
                {
                    "step": step + 1,
                    "loss": float(loss),
                    "a_rel": a_rel,
                    "c_rel": c_rel,
                    "cos": cos,
                    "ent_diff": ent_val,  # ENT: mean |Delta H| curve, in nats
                }
            )
            print(
                f"step {step + 1}/{args.steps} loss {float(loss):.4f} "
                f"A/A_naive {a_rel:.3f} C/C_naive {c_rel:.3f} cos(w,v) {cos:.4f} "
                f"|dH| {ent_val:.4f}",  # ENT: added to the printed line
                flush=True,
            )

    name = args.name or "dir_ent"
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
            "ent_weight": args.ent_weight,  # ENT: recorded; load_correction ignores unknown keys
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
    ap.add_argument("--name", default="dir_ent")
    ap.add_argument(
        "--exclude-r4",
        dest="exclude_r4",
        action="store_true",
        help="hold out the round-four features (configs/features_r4.yaml or data/splits_r4.npz) and their "
        "|cos| >= 0.3 buffer from the training pool; used to train dir_r4",
    )
    ap.add_argument("--ent-weight", dest="ent_weight", type=float, default=1.0)  # ENT: new flag
    a = ap.parse_args()
    if a.stage == "eval":
        evaluate(a)
    else:
        train(a)
