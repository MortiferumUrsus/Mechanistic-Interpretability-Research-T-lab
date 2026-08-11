"""Pick every inference knob on DEV directions, before TEST is ever touched.

Selection uses a generation-free proxy: run each candidate through the frozen suffix and
measure the tangent decomposition of the final-logit response. A is how much of the concept's
small-signal causal direction survives, C is the relative size of the off-tangent residue,
which is what shows up as broken text. A candidate is admissible if it keeps at least
`min_a_ratio` of the naive arm's A, and among admissible candidates the lowest C wins.

This costs no sampling and no judge, so the DEV loop is cheap enough to cover every knob.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import yaml

from common import DATA, DEVICE, HOOK, RESULTS, ROOT, ActStats, load_model, load_sae, seed_all
from denoiser import WienerDenoiser, build_denoiser, sigma_for_norm, transported_direction
from train_ctr import make_suffix

CONFIGS = ROOT / "configs"
C_GRID = [1.0, 2.0, 3.0]


def load_trained(name: str, stats: ActStats):
    blob = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location=DEVICE)
    d = build_denoiser(
        blob["arch"], stats, hidden=blob["hidden"], scale=blob["scale"], cond=blob["cond"]
    ).to(DEVICE)
    d.load_state_dict(blob["state_dict"])
    d.eval()
    return d


@torch.no_grad()
def run(args) -> None:
    import steering as S

    seed_all(args.seed)
    stats = ActStats.load()
    scale = stats.median_norm
    model = load_model()
    sae = load_sae()
    suffix = make_suffix(model)

    prompts = json.loads((DATA / "prompts.json").read_text(encoding="utf-8"))[: args.n_prompts]
    toks = model.to_tokens(prompts).to(DEVICE)
    _, cache = model.run_with_cache(toks, names_filter=HOOK)
    h = cache[HOOK].detach()
    del cache
    z0 = suffix(h)

    dev = yaml.safe_load((CONFIGS / "features.yaml").read_text(encoding="utf-8"))["dev"]
    ceiling = torch.load(DATA / "sae_feature_stats.pt", map_location=DEVICE)["max_act"].to(DEVICE)
    w_norm = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    trained = {n: load_trained(n, stats) for n in args.denoisers.split(",")}
    eps = args.eps_c * scale

    rows = []
    for rec in dev:
        f = int(rec["index"])
        v = sae.W_dec[f].detach().float()
        v_hat = v / v.norm()
        g = (suffix(h + eps * v_hat) - z0) / eps
        gn = (g * g).sum(-1, keepdim=True).clamp_min(1e-6)

        exempt = (w_norm @ v_hat).abs() > args.exempt_cos
        exempt[f] = True

        cands: dict[str, S.Intervention] = {"naive": S.naive, "norm_preserving": S.norm_preserving}
        for sh in [float(x) for x in args.shrinks.split(",")]:
            cands[f"mts|shrink={sh}"] = S.TransportedArm(
                direction=transported_direction(stats, v_hat, sh)
            )
        for k in [float(x) for x in args.fsr_ks.split(",")]:
            cands[f"fsr|k={k}"] = S.FeatureSurgeryArm(
                sae=sae, ceiling=ceiling, exempt=exempt, k=k
            )
        for name, d in trained.items():
            for eta in [float(x) for x in args.etas.split(",")]:
                cands[f"denoise_naive|{name}|eta={eta}"] = S.DenoiserArm(
                    denoiser=d, mode="naive", eta=eta
                )
            for lam in [float(x) for x in args.lams.split(",")]:
                cands[f"cds|{name}|lam={lam}"] = S.DenoiserArm(denoiser=d, mode="cds", lam=lam)
        for sig, lam in itertools.product(
            [float(x) for x in args.wiener_sigmas.split(",")],
            [float(x) for x in args.lams.split(",")],
        ):
            wd = WienerDenoiser(
                stats, sigma=sigma_for_norm(sig * scale), shrink=args.wiener_shrink
            )
            cands[f"wiener_cds|sigma={sig}|lam={lam}"] = S.DenoiserArm(
                denoiser=wd, mode="cds", lam=lam
            )

        for cname, fn in cands.items():
            for c in C_GRID:
                s = c * scale
                dz = suffix(fn(h, v_hat, s)) - z0
                a = (dz * g).sum(-1, keepdim=True) / gn
                resid = dz - a * g
                cval = resid.norm(dim=-1) / (a.squeeze(-1).abs() * g.norm(dim=-1)).clamp_min(1e-6)
                rows.append(
                    {
                        "feature": f,
                        "candidate": cname,
                        "c": c,
                        "A": float(a.mean()),
                        "C": float(cval.mean()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "dev_sweep_raw.csv", index=False)

    base = (
        df[df["candidate"] == "naive"].groupby(["feature", "c"])["A"].mean().rename("A_naive").reset_index()
    )
    m = df.merge(base, on=["feature", "c"])
    m["a_ratio"] = m["A"] / m["A_naive"]
    agg = (
        m.groupby("candidate")
        .agg(A=("A", "mean"), C=("C", "mean"), a_ratio=("a_ratio", "mean"))
        .reset_index()
        .sort_values("C")
    )
    agg["admissible"] = agg["a_ratio"] >= args.min_a_ratio
    agg.to_csv(RESULTS / "dev_sweep_summary.csv", index=False)
    print(agg.to_string(index=False))

    def pick(prefix: str) -> str | None:
        sub = agg[agg["candidate"].str.startswith(prefix) & agg["admissible"]]
        return None if sub.empty else str(sub.iloc[0]["candidate"])

    chosen = {p: pick(p) for p in ("cds|", "denoise_naive|", "mts|", "fsr|", "wiener_cds|")}
    (RESULTS / "dev_sweep_choice.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoisers", default="mlp_mix_cond1_s0,mlp_gauss_cond1_s0,linear_mix_cond0_s0")
    ap.add_argument("--lams", default="0.5,1.0,1.5")
    ap.add_argument("--etas", default="0.5,1.0")
    ap.add_argument("--shrinks", default="0.01,0.05,0.2")
    ap.add_argument("--fsr-ks", dest="fsr_ks", default="0.3,0.5,0.7,1.0")
    ap.add_argument("--wiener-sigmas", dest="wiener_sigmas", default="0.5,1.0,2.0")
    ap.add_argument("--wiener-shrink", dest="wiener_shrink", type=float, default=0.05)
    ap.add_argument("--exempt-cos", dest="exempt_cos", type=float, default=0.3)
    ap.add_argument("--eps-c", dest="eps_c", type=float, default=0.25)
    ap.add_argument("--min-a-ratio", dest="min_a_ratio", type=float, default=0.9)
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
