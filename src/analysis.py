"""Mechanism measurements: what the repair does, and why it helps when it helps."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from common import DATA, DEVICE, HOOK, RESULTS, ROOT, ActStats, load_model, load_sae, seed_all
from denoiser import WienerDenoiser, build_denoiser, sigma_for_norm, transported_direction

CONFIGS = ROOT / "configs"
C_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def load_trained(name: str, stats: ActStats):
    blob = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location=DEVICE)
    d = build_denoiser(
        blob["arch"], stats, hidden=blob["hidden"], scale=blob["scale"], cond=blob["cond"]
    ).to(DEVICE)
    d.load_state_dict(blob["state_dict"])
    d.eval()
    return d, blob


def sample_acts(n: int, seed: int) -> torch.Tensor:
    from common import open_memmap

    acts = open_memmap()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(acts.shape[0], size=n, replace=False))
    return torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEVICE).float()


def feature_dirs(split: str, sae) -> list[tuple[int, torch.Tensor]]:
    recs = yaml.safe_load((CONFIGS / "features.yaml").read_text(encoding="utf-8"))[split]
    out = []
    for r in recs:
        f = int(r["index"])
        v = sae.W_dec[f].detach().float()
        out.append((f, v / v.norm()))
    return out


@torch.no_grad()
def transmission(args) -> None:
    """tau(c) = <D(h + s v) - D(h), v> / s. Baseline-subtracted, so it is defined and unbiased.

    Also reports the split of the correction into the component along v and the rest, which is
    exactly what the direction-preserving arm keeps and drops.
    """
    stats = ActStats.load()
    sae = load_sae()
    h = sample_acts(args.n_tokens, args.seed)
    scale = stats.median_norm

    denoisers = {}
    for name in args.denoisers.split(","):
        if name.startswith("wiener"):
            sigma = sigma_for_norm(float(name.split(":")[1]) * scale)
            denoisers[name] = WienerDenoiser(stats, sigma=sigma, shrink=args.shrink)
        else:
            denoisers[name], _ = load_trained(name, stats)

    rows = []
    for dname, D in denoisers.items():
        base = D(h)
        for f, v_hat in feature_dirs(args.split, sae):
            for c in C_GRID:
                s = c * scale
                delta = D(h + s * v_hat) - base - s * v_hat
                along = delta @ v_hat
                perp = delta - along.unsqueeze(-1) * v_hat
                rows.append(
                    {
                        "denoiser": dname,
                        "feature": f,
                        "c": c,
                        "tau": float(1.0 + (along / s).mean()),
                        "tau_sd": float((along / s).std()),
                        "erase": float(-(along / s).mean()),
                        "perp_frac": float(
                            (perp.norm(dim=-1) ** 2 / delta.norm(dim=-1).clamp_min(1e-6) ** 2).mean()
                        ),
                        "delta_rel_norm": float((delta.norm(dim=-1) / s).mean()),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "transmission.csv", index=False)
    print(df.groupby(["denoiser", "c"])[["tau", "erase", "perp_frac"]].mean().to_string())


@torch.no_grad()
def spectral(args) -> None:
    """Energy of the perturbation along the eigenbasis of the activation covariance.

    H4 predicts that steering injects energy into low-variance directions and that the repair
    removes exactly that energy.
    """
    stats = ActStats.load()
    sae = load_sae()
    evals, evecs = torch.linalg.eigh(stats.cov.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].float(), evecs[:, order].float()
    h = sample_acts(args.n_tokens, args.seed)
    scale = stats.median_norm
    D, _ = load_trained(args.denoiser, stats)
    base = D(h)

    n_bins = 16
    edges = np.linspace(0, len(evals), n_bins + 1).astype(int)
    rows = []
    for f, v_hat in feature_dirs(args.split, sae):
        tdir = transported_direction(stats, v_hat, args.shrink)
        for c in [1.0, 2.0]:
            s = c * scale
            variants = {
                "naive": h + s * v_hat,
                "denoise_naive": D(h + s * v_hat),
                "cds": h
                + s * v_hat
                + (lambda d: d - (d @ v_hat).unsqueeze(-1) * v_hat)(D(h + s * v_hat) - base - s * v_hat),
                "mts": h + s * tdir,
            }
            for name, x in variants.items():
                proj = (x - h) @ evecs
                energy = (proj**2).mean(0)
                binned = [float(energy[edges[i] : edges[i + 1]].sum()) for i in range(n_bins)]
                rows.append(
                    {
                        "feature": f,
                        "c": c,
                        "arm": name,
                        "total_energy": float(energy.sum()),
                        **{f"bin{i}": b for i, b in enumerate(binned)},
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "spectral.csv", index=False)
    tail = [f"bin{i}" for i in range(n_bins // 2, n_bins)]
    df["tail_frac"] = df[tail].sum(1) / df["total_energy"]
    print(df.groupby(["arm", "c"])[["total_energy", "tail_frac"]].mean().to_string())


@torch.no_grad()
def surgery(args) -> None:
    """Which non-target latents exceed their natural ceiling under steering, and how many.

    This is the feature-level statement of the damage mechanism: if the number and identity of
    hijacked latents predicts the repair's gain, the mechanism is named, not just localised.
    """
    from steering import FeatureSurgeryArm

    stats = ActStats.load()
    sae = load_sae()
    blob = torch.load(DATA / "sae_feature_stats.pt", map_location=DEVICE)
    ceiling = blob["max_act"].to(DEVICE)
    model = load_model()
    tokenizer = model.tokenizer
    del model
    torch.cuda.empty_cache()

    h = sample_acts(args.n_tokens, args.seed)
    scale = stats.median_norm
    w = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rows, detail = [], {}
    for f, v_hat in feature_dirs(args.split, sae):
        exempt = (w @ v_hat).abs() > args.exempt_cos
        exempt[f] = True
        arm = FeatureSurgeryArm(sae=sae, ceiling=ceiling, exempt=exempt, k=args.k)
        for c in C_GRID:
            rep = arm.clamp_report(h, v_hat, c * scale)
            rows.append({"feature": f, "c": c, "n_exempt": int(exempt.sum()), **{k: v for k, v in rep.items() if k != "top_latents"}})
            if c == 2.0:
                detail[str(f)] = rep["top_latents"]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "surgery_clamp.csv", index=False)
    (RESULTS / "surgery_top_latents.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(df.groupby("c")[["n_over_per_token", "total_excess"]].mean().to_string())


@torch.no_grad()
def causal(args) -> None:
    """Tangent-aligned versus off-tangent response of the frozen upper half.

    A is how much of the small-signal causal direction survives, C is the relative size of the
    nonlinear residue. The claim to test is that gains track reductions in C at matched A.
    """
    stats = ActStats.load()
    sae = load_sae()
    model = load_model()
    prompts = json.loads((DATA / "prompts.json").read_text(encoding="utf-8"))[: args.n_prompts]
    toks = model.to_tokens(prompts).to(DEVICE)
    scale = stats.median_norm
    D, _ = load_trained(args.denoiser, stats)

    _, cache = model.run_with_cache(toks, names_filter=HOOK)
    h0 = cache[HOOK].clone()
    del cache

    def suffix(hval: torch.Tensor) -> torch.Tensor:
        def hook(resid, hook):
            return hval

        with model.hooks(fwd_hooks=[(HOOK, hook)]):
            logits = model(toks)
        z = logits[:, -1, :].float()
        return z - z.mean(-1, keepdim=True)

    z0 = suffix(h0)
    base = D(h0)
    rows = []
    for f, v_hat in feature_dirs(args.split, sae):
        tdir = transported_direction(stats, v_hat, args.shrink)
        eps = args.eps_c * scale
        g = (suffix(h0 + eps * v_hat) - z0) / eps
        gn = (g**2).sum(-1).clamp_min(1e-9)
        for c in C_GRID:
            s = c * scale
            d_steer = D(h0 + s * v_hat) - base - s * v_hat
            perp = d_steer - (d_steer @ v_hat).unsqueeze(-1) * v_hat
            variants = {
                "naive": h0 + s * v_hat,
                "denoise_naive": D(h0 + s * v_hat),
                "cds": h0 + s * v_hat + perp,
                "mts": h0 + s * tdir,
            }
            for name, hv in variants.items():
                dz = suffix(hv) - z0
                a = (dz * g).sum(-1) / gn
                resid = dz - a.unsqueeze(-1) * g
                cval = resid.norm(dim=-1) / (a.abs() * g.norm(dim=-1)).clamp_min(1e-9)
                rows.append(
                    {
                        "feature": f,
                        "c": c,
                        "arm": name,
                        "A": float(a.mean()),
                        "C": float(cval.mean()),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "causal_AC.csv", index=False)
    print(df.groupby(["arm", "c"])[["A", "C"]].mean().to_string())


def predictors(args) -> None:
    """Per-feature gain against the two registered geometric predictors."""
    from scipy.stats import spearmanr

    tbl = pd.read_csv(RESULTS / "endpoint_per_feature.csv")
    feats = yaml.safe_load((CONFIGS / "features.yaml").read_text(encoding="utf-8"))["test"]
    geo = pd.DataFrame(
        [{"feature": int(r["index"]), "var_along": r["var_along"], "kappa_tilt": r["kappa_tilt"]} for r in feats]
    )
    base = tbl[tbl["arm"] == "naive"][["feature", "concept_at_budget"]].rename(
        columns={"concept_at_budget": "base"}
    )
    rows = []
    for arm, g in tbl.groupby("arm"):
        if arm == "naive":
            continue
        m = g.merge(base, on="feature").merge(geo, on="feature").dropna()
        if len(m) < 4:
            continue
        m["gain"] = m["concept_at_budget"] - m["base"]
        for pred in ("var_along", "kappa_tilt"):
            rho, p = spearmanr(m[pred], m["gain"])
            rows.append({"arm": arm, "predictor": pred, "spearman": rho, "p": p, "n": len(m)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "predictors.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["transmission", "spectral", "surgery", "causal", "predictors"])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--denoiser", default="mlp_mix_cond1_s0")
    ap.add_argument("--denoisers", default="mlp_mix_cond1_s0,wiener:0.5,wiener:1.0,wiener:2.0")
    ap.add_argument("--n-tokens", dest="n_tokens", type=int, default=4096)
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=16)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--exempt-cos", dest="exempt_cos", type=float, default=0.3)
    ap.add_argument("--eps-c", dest="eps_c", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    globals()[a.stage](a)
