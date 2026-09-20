"""Generate continuations for every (arm, feature, strength) cell of the grid.

All arms see the same prompts and the same per-prompt seed, so comparisons are paired.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from common import (
    DATA,
    DEVICE,
    HOOK,
    RESULTS,
    ROOT,
    SHARED_DIRECTION,
    ActStats,
    load_ceilings,
    load_features,
    load_model,
    load_sae,
    natural_strength,
    seed_all,
)
from steering import HookState, make_hook

CONFIGS = ROOT / "configs"


def load_prompts(model, path: Path = DATA / "prompts.json", want_len: int = 8) -> list[str]:
    """Keep only prompts that re-tokenise to a fixed length, so batches need no padding."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    keep = [p for p in raw if model.to_tokens(p).shape[1] == want_len + 1]
    if not keep:
        raise RuntimeError("no prompt round-trips to the requested token length")
    return keep


def load_expA_config() -> dict:
    """expA.yaml hyper-parameters, with defaults when the sweep hasn't written it yet."""
    path = CONFIGS / "expA.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    cfg = dict(cfg or {})
    cfg.setdefault("kappa_shared", 1.0)
    cfg.setdefault("kappa_anti", 1.0)
    return cfg


def lookup_concept_mu(blob: dict, feat_idx: int) -> torch.Tensor:
    """mu_f for a feature: eval_mu if it was mined as a held-out eval latent, else the fit-pool mu.

    Same fallback order as denoiser_cond._lookup_mu.
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


def build_arms(names: list[str], model, sae, stats: ActStats, frozen: dict) -> dict:
    """Instantiate the requested arms from frozen hyper-parameters."""
    import steering as S
    from denoiser import WienerDenoiser, build_denoiser, transported_direction

    arms: dict = {}
    for name in names:
        if name == "clean":
            arms[name] = S.clean
        elif name == "naive":
            arms[name] = S.naive
        elif name == "norm_preserving":
            arms[name] = S.norm_preserving
        elif name == "ln_match":
            arms[name] = S.ln_stat_matching
        elif name in ("denoise_naive", "cds", "par_only", "perp_only", "clean_then"):
            # the task's literal arm gets whichever denoiser won for it on DEV, not the one that
            # won for the contrastive arm: otherwise the baseline is handicapped by our choice
            key = "denoise_naive_denoiser" if name == "denoise_naive" else "denoiser"
            ckpt = frozen.get(key) or frozen["denoiser"]
            blob = torch.load(ROOT / "checkpoints" / f"{ckpt}.pt", map_location=DEVICE)
            d = build_denoiser(
                blob["arch"], stats, hidden=blob["hidden"], scale=blob["scale"], cond=blob["cond"]
            ).to(DEVICE)
            d.load_state_dict(blob["state_dict"])
            d.eval()
            mode = {"denoise_naive": "naive"}.get(name, name)
            arms[name] = S.DenoiserArm(
                denoiser=d, mode=mode, eta=frozen["eta"], lam=frozen["lam"]
            )
        elif name == "wiener_cds":
            d = WienerDenoiser(stats, sigma=frozen["wiener_sigma"], shrink=frozen["shrink"])
            arms[name] = S.DenoiserArm(denoiser=d, mode="cds", lam=frozen["lam"])
        elif name == "dirfix":
            from train_direction import load_correction

            arms[name] = S.CorrectedDirectionArm(
                correction=load_correction(ROOT / "checkpoints" / f"{frozen['direction']}.pt")
            )
        elif name == "randrot":
            arms[name] = S.RandomRotationArm(
                cos_target=frozen.get("randrot_cos", 0.866), seed=frozen.get("randrot_seed", 0)
            )
        elif name == "mts":
            arms[name] = ("per_feature", "mts")
        elif name == "fsr":
            arms[name] = ("per_feature", "fsr")
        elif name in ("diffmeans", "diffmeans_purified"):
            arms[name] = ("per_feature", name)
        elif name == "rotate":
            arms[name] = S.RotateArm()
        elif name in ("shared", "residual", "antimanifold", "shared_only", "centred", "purified"):
            expA = load_expA_config()
            d_bar = torch.load(SHARED_DIRECTION, map_location=DEVICE)["d_bar"].to(DEVICE).float()
            if name == "shared":
                arms[name] = S.SharedDirectionArm(d_bar=d_bar, kappa=float(expA["kappa_shared"]))
            elif name == "shared_only":
                arms[name] = S.SharedOnlyArm(d_bar=d_bar)
            elif name == "purified":
                arms[name] = S.PurifiedArm(d_bar=d_bar)
            elif name == "centred":
                arms[name] = S.CentredArm(mu_hat=stats.mean / stats.mean.norm().clamp_min(1e-6))
            elif name == "residual":
                from train_direction import load_correction

                arms[name] = S.ResidualDirectionArm(
                    correction=load_correction(ROOT / "checkpoints" / f"{frozen['direction']}.pt"),
                    d_bar=d_bar,
                )
            else:  # antimanifold
                arms[name] = S.AntiManifoldArm(
                    cov=stats.shrunk_cov(frozen["shrink"]).to(DEVICE), kappa=float(expA["kappa_anti"])
                )
        elif name in ("cond_wiener", "cond_denoise"):
            arms[name] = ("per_feature", name)
        else:
            raise ValueError(name)
    return arms


def resolve_per_feature(tag: str, sae, stats, frozen, v_hat, feat_idx):
    import steering as S
    from denoiser import transported_direction

    if tag == "mts":
        return S.TransportedArm(direction=transported_direction(stats, v_hat, frozen["shrink"]))
    if tag == "fsr":
        blob = torch.load(DATA / "sae_feature_stats.pt", map_location=DEVICE)
        ceiling = blob["max_act"].to(DEVICE)
        w = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        exempt = (w @ v_hat).abs() > frozen["fsr_exempt_cos"]
        exempt[feat_idx] = True
        return S.FeatureSurgeryArm(sae=sae, ceiling=ceiling, exempt=exempt, k=frozen["fsr_k"])
    if tag == "cond_wiener":
        from denoiser_cond import ConditionalWienerArm

        return ConditionalWienerArm.load(DATA / "concept_stats.pt", stats, feat_idx, v_hat, lam=frozen.get("cond_lam", 1.0))
    if tag == "cond_denoise":
        from denoiser_cond import ConditionalDenoiserArm

        return ConditionalDenoiserArm.load(
            ROOT / "checkpoints" / f"{frozen.get('cond_denoiser', 'cond_mlp_s0')}.pt",
            stats,
            feat_idx,
            v_hat,
            eta=frozen.get("cond_eta", 1.0),
        )
    if tag == "diffmeans":
        blob = torch.load(DATA / "concept_stats.pt", map_location=DEVICE)
        mu_f = lookup_concept_mu(blob, feat_idx).to(device=DEVICE, dtype=torch.float32)
        u = mu_f - stats.mean
        u = u / u.norm().clamp_min(1e-6)
        return S.DiffMeansArm(u=u)
    if tag == "diffmeans_purified":
        blob = torch.load(DATA / "concept_stats.pt", map_location=DEVICE)
        mu_f = lookup_concept_mu(blob, feat_idx).to(device=DEVICE, dtype=torch.float32)
        u_dm = mu_f - stats.mean
        u_dm = u_dm / u_dm.norm().clamp_min(1e-6)
        d_bar = torch.load(SHARED_DIRECTION, map_location=DEVICE)["d_bar"].to(DEVICE).float()
        u = u_dm - (u_dm @ d_bar) * d_bar
        u = u / u.norm().clamp_min(1e-6)
        return S.DiffMeansPurifiedArm(u=u)
    raise ValueError(tag)


@torch.no_grad()
def run(args) -> None:
    seed_all(args.seed)
    model = load_model()
    sae = load_sae()
    stats = ActStats.load()
    frozen = yaml.safe_load((CONFIGS / "frozen.yaml").read_text(encoding="utf-8"))
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        frozen[k] = yaml.safe_load(v)
    feats = load_features(args.split)
    if args.max_features:
        feats = feats[: args.max_features]

    prompts = load_prompts(model, path=Path(args.prompts))[: args.n_prompts]
    grid = [float(c) for c in args.c_grid.split(",")]
    arms = build_arms(args.arms.split(","), model, sae, stats, frozen)
    state = HookState()
    ceilings = load_ceilings()

    out_path = RESULTS / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")
    n_cells = len(feats) * len(grid) * len(arms)
    t0 = time.time()
    pbar = tqdm(total=n_cells, unit="cell")

    for rec in feats:
        f = int(rec["index"])
        v = sae.W_dec[f].detach().float()
        v_hat = v / v.norm()
        scale = natural_strength(sae, ceilings, f)
        for arm_name, arm in arms.items():
            fn = arm
            if isinstance(arm, tuple):
                fn = resolve_per_feature(arm[1], sae, stats, frozen, v_hat, f)
            for c in grid:
                # the unsteered arm does not depend on strength; generating it once per feature
                # keeps the null identical across the grid instead of resampling it
                if arm_name == "clean" and c != grid[0]:
                    pbar.update(1)
                    continue
                s = c * scale
                for i in range(0, len(prompts), args.batch):
                    chunk = prompts[i : i + args.batch]
                    toks = model.to_tokens(chunk).to(DEVICE)
                    torch.manual_seed(args.seed * 100003 + i)
                    state.reset()
                    with model.hooks(fwd_hooks=[(HOOK, make_hook(fn, v_hat, s, state))]):
                        gen = model.generate(
                            toks,
                            max_new_tokens=args.new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            # distinct-n is only comparable at fixed length, so never stop early
                            stop_at_eos=False,
                            verbose=False,
                        )
                    cont = gen[:, toks.shape[1] :]
                    for j in range(cont.shape[0]):
                        fh.write(
                            json.dumps(
                                {
                                    "arm": arm_name + (f"|{args.tag}" if args.tag else ""),
                                    "feature": f,
                                    "c": c,
                                    "prompt_idx": i + j,
                                    "prompt": chunk[j],
                                    "text": model.to_string(cont[j]),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                pbar.update(1)
    pbar.close()
    fh.close()
    n = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"wrote {n} generations to {out_path} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "test_r3", "test_r4"])
    ap.add_argument("--arms", default="clean,naive,norm_preserving,denoise_naive,cds,mts,fsr")
    # strength in units of the latent's own natural ceiling, not of the global activation norm
    ap.add_argument("--c-grid", dest="c_grid", default="0,0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=30)
    ap.add_argument("--new-tokens", dest="new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--max-features", dest="max_features", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen_test.jsonl")
    ap.add_argument("--prompts", default=str(DATA / "prompts.json"))
    ap.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a frozen hyper-parameter; DEV selection only",
    )
    ap.add_argument("--tag", default="", help="suffix appended to arm names; DEV selection only")
    run(ap.parse_args())
