"""Select validation directions by an objective rule and freeze the FIT / DEV / TEST split.

Hand-picking a few pretty features would only license a few case studies, so features are
ranked by how lexically crisp their top activations are and then sampled with a fixed seed.
The same top-activation statistics also produce each feature's keyword set, which makes the
model-free concept metric reproducible rather than hand-written.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from common import DATA, DEVICE, RESULTS, ROOT, load_model, load_sae, open_memmap, seed_all

CONFIGS = ROOT / "configs"
TOPK = 256
FREQ_LO, FREQ_HI = 1e-4, 5e-2
N_TEST, N_DEV = 12, 6
MAX_COS = 0.3
KEYWORD_MASS = 0.8
PUNCT = re.compile(r"^[\W\d_]+$")

# A concept feature has to fire on several related content words. One distinct token means a
# token detector; dozens of them means the feature is not about a nameable concept. Function
# words are excluded outright: steering "the" cannot show up as a concept in generated text.
MIN_DISTINCT, MAX_DISTINCT = 3, 25
MIN_CONTENT_MASS = 0.7
STOPWORDS = set(
    """a about above after again against all also am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does doesn doing
    don down during each even few for from further had hadn has hasn have haven having he her here
    hers herself him himself his how however i if in into is isn it its itself just let ll me more
    most much must mustn my myself no nor not now of off on once only or other others ought our
    ours ourselves out over own same shan she should shouldn so some such than that the their
    theirs them themselves then there these they this those through to too under until up very
    was wasn we were weren what when where which while who whom why will with won would wouldn
    yet you your yours yourself yourselves thus hence therefore though although upon among
    within without across along around behind beyond during except inside outside since toward
    towards unless whereas whether""".split()
)


# Explicit latents are skipped: the judge model refuses to score them, which would silently
# turn one of the twelve test cells into noise, and they do not belong in a submitted report.
BLOCKED = set(
    """cock cocks dick dicks penis vagina sperm semen anal anus nsfw porn porno pornography
    fuck fucks fucked fucking shit cunt slut whore rape raped raping nude nudes naked
    masturb orgasm ejacul genital breasts boobs tits""".split()
)


def is_content(tok: str) -> bool:
    t = tok.strip().lower()
    return t.isascii() and t.isalpha() and len(t) >= 4 and t not in STOPWORDS


def is_blocked(tok: str) -> bool:
    t = tok.strip().lower()
    return any(b in t for b in BLOCKED) if t.isascii() else False


@torch.no_grad()
def top_activations(chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One streaming pass: per-latent frequency, mean, and top-K (value, token id)."""
    sae = load_sae()
    f = sae.cfg.d_sae
    acts = open_memmap()
    toks = np.memmap(DATA / "toks.i32", dtype=np.int32, mode="r")
    n = acts.shape[0]

    top_val = torch.full((TOPK, f), -1.0, device=DEVICE)
    top_tok = torch.zeros((TOPK, f), device=DEVICE, dtype=torch.long)
    n_active = torch.zeros(f, device=DEVICE, dtype=torch.long)
    sum_active = torch.zeros(f, device=DEVICE, dtype=torch.float64)

    for i in tqdm(range(0, n, chunk), unit="chunk"):
        h = torch.from_numpy(np.ascontiguousarray(acts[i : i + chunk])).to(DEVICE).float()
        t = torch.from_numpy(np.ascontiguousarray(toks[i : i + chunk])).to(DEVICE).long()
        z = sae.encode(h)
        n_active += (z > 0).sum(0)
        sum_active += z.sum(0).double()
        k = min(TOPK, z.shape[0])
        cv, ci = torch.topk(z, k, dim=0)
        ct = t[ci]
        merged_v = torch.cat([top_val, cv], 0)
        merged_t = torch.cat([top_tok, ct], 0)
        sel = torch.topk(merged_v, TOPK, dim=0).indices
        top_val = merged_v.gather(0, sel)
        top_tok = merged_t.gather(0, sel)
        del h, t, z, cv, ci, ct, merged_v, merged_t, sel

    del sae
    torch.cuda.empty_cache()
    freq = (n_active.double() / n).float()
    mean_act = (sum_active / n_active.clamp_min(1)).float()
    return top_val, top_tok, freq, mean_act


def token_mass(vals: np.ndarray, tids: np.ndarray) -> list[tuple[int, float]]:
    """Total top-K activation mass per distinct token id, descending."""
    keep = vals > 0
    vals, tids = vals[keep], tids[keep]
    if len(vals) == 0:
        return []
    order = np.argsort(tids)
    tids, vals = tids[order], vals[order]
    uniq, start = np.unique(tids, return_index=True)
    sums = np.add.reduceat(vals, start)
    idx = np.argsort(-sums)
    return [(int(uniq[i]), float(sums[i])) for i in idx]


def mass_prefix(mass: list[tuple[int, float]], frac: float = KEYWORD_MASS) -> list[tuple[int, float]]:
    """The shortest prefix of the descending token list carrying `frac` of the mass."""
    total = sum(m for _, m in mass)
    out, acc = [], 0.0
    for item in mass:
        out.append(item)
        acc += item[1]
        if acc >= frac * total:
            break
    return out


def profile(mass: list[tuple[int, float]], decode) -> dict:
    """Shape of a latent's lexical footprint, used as the objective selection rule."""
    if not mass:
        return {"n_distinct": 0, "content_mass": 0.0, "blocked": False}
    head = mass_prefix(mass)
    tot = sum(m for _, m in head) or 1e-9
    content = sum(m for t, m in head if is_content(decode(t)))
    blocked = any(is_blocked(decode(t)) for t, _ in head)
    return {"n_distinct": len(head), "content_mass": content / tot, "blocked": blocked}


def geometry(w_dec: torch.Tensor, cov: torch.Tensor, idx: torch.Tensor) -> pd.DataFrame:
    v = w_dec[idx]
    dec_norm = v.norm(dim=-1)
    vh = v / dec_norm.unsqueeze(-1).clamp_min(1e-6)
    sv = vh @ cov
    var_along = (sv * vh).sum(-1)
    tilt = (sv - var_along.unsqueeze(-1) * vh).norm(dim=-1) / var_along.clamp_min(1e-9)
    return pd.DataFrame(
        {
            "index": idx.cpu().numpy(),
            "dec_norm": dec_norm.cpu().numpy(),
            "var_along": var_along.cpu().numpy(),
            "kappa_tilt": tilt.cpu().numpy(),
        }
    )


def greedy_decorrelated(pool: list[int], vh: torch.Tensor, want: int) -> list[int]:
    chosen: list[int] = []
    for f in pool:
        if not chosen:
            chosen.append(f)
            continue
        cos = (vh[torch.tensor(chosen, device=vh.device)] @ vh[f]).abs().max()
        if float(cos) < MAX_COS:
            chosen.append(f)
        if len(chosen) == want:
            break
    return chosen


def fit_indices(vh: torch.Tensor, held: list[int], chunk: int = 2048) -> np.ndarray:
    held_t = torch.tensor(held, device=vh.device)
    held_v = vh[held_t]
    keep = []
    for i in range(0, vh.shape[0], chunk):
        block = vh[i : i + chunk]
        cos = (block @ held_v.T).abs().max(dim=1).values
        ok = (cos < MAX_COS).nonzero(as_tuple=True)[0] + i
        keep.append(ok.cpu().numpy())
    out = np.concatenate(keep)
    return np.sort(np.setdiff1d(out, np.array(held)))


def clean_keywords(strings: list[str]) -> list[str]:
    out: list[str] = []
    for s in strings:
        t = s.strip().lower()
        if len(t) < 3 or PUNCT.match(t):
            continue
        if t not in out:
            out.append(t)
    return out


def select(args) -> None:
    seed_all(args.seed)
    CONFIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    top_val, top_tok, freq, mean_act = top_activations(args.chunk)
    from common import ActStats

    stats = ActStats.load()
    sae = load_sae()
    w_dec = sae.W_dec.detach().float()
    vh_all = w_dec / w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    model = load_model(device="cpu")
    tokenizer = model.tokenizer
    cache: dict[int, str] = {}

    def decode(t: int) -> str:
        if t not in cache:
            cache[t] = tokenizer.decode([t])
        return cache[t]

    alive = ((freq >= FREQ_LO) & (freq <= FREQ_HI)).nonzero(as_tuple=True)[0]
    tv = top_val[:, alive].T.cpu().numpy()
    tt = top_tok[:, alive].T.cpu().numpy()
    masses = [token_mass(tv[i], tt[i]) for i in range(len(alive))]
    profs = [profile(m, decode) for m in masses]

    geo = geometry(w_dec, stats.cov, alive)
    surv = geo.assign(
        freq=freq[alive].cpu().numpy(),
        mean_act=mean_act[alive].cpu().numpy(),
        n_distinct=[p["n_distinct"] for p in profs],
        content_mass=[p["content_mass"] for p in profs],
        blocked=[p["blocked"] for p in profs],
    )
    surv.to_csv(RESULTS / "feature_stats.csv", index=False)

    qualify = surv[
        (surv["n_distinct"] >= MIN_DISTINCT)
        & (surv["n_distinct"] <= MAX_DISTINCT)
        & (surv["content_mass"] >= MIN_CONTENT_MASS)
        & (~surv["blocked"])
    ]
    print(
        f"latents={len(freq)} alive={len(alive)} qualifying={len(qualify)} "
        f"(distinct in [{MIN_DISTINCT},{MAX_DISTINCT}], content mass >= {MIN_CONTENT_MASS})"
    )

    pool_global = [int(x) for x in qualify["index"].to_numpy()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool_global)
    chosen = greedy_decorrelated(pool_global, vh_all, N_TEST + N_DEV)
    if len(chosen) < N_TEST + N_DEV:
        raise RuntimeError(f"only {len(chosen)} decorrelated features found in the pool")
    test_idx, dev_idx = chosen[:N_TEST], chosen[N_TEST:]

    fit = fit_indices(vh_all, chosen)
    np.savez(
        DATA / "splits.npz",
        test=np.array(test_idx, dtype=np.int32),
        dev=np.array(dev_idx, dtype=np.int32),
        fit=fit.astype(np.int32),
    )

    local_of = {int(alive[i]): i for i in range(len(alive))}

    def record(f: int) -> dict:
        mass = masses[local_of[f]]
        strings = [decode(t) for t, _ in mass]
        shown = [s.replace(" ", "_", 1) if s.startswith(" ") else s for s in strings[:15]]
        kw_raw = [decode(t) for t, _ in mass_prefix(mass)]
        row = surv[surv["index"] == f].iloc[0]
        return {
            "index": int(f),
            "freq": float(f"{row.freq:.6g}"),
            "mean_act": float(f"{row.mean_act:.6g}"),
            "n_distinct": int(row.n_distinct),
            "content_mass": float(f"{row.content_mass:.6g}"),
            "dec_norm": float(f"{row.dec_norm:.6g}"),
            "var_along": float(f"{row.var_along:.6g}"),
            "kappa_tilt": float(f"{row.kappa_tilt:.6g}"),
            "top_tokens": shown,
            "keywords": clean_keywords(kw_raw),
        }

    payload = {
        "test": [record(f) for f in test_idx],
        "dev": [record(f) for f in dev_idx],
    }
    (CONFIGS / "features.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    fit_v = vh_all[torch.as_tensor(fit, device=DEVICE, dtype=torch.long)]
    rows = []
    for split, ids in (("test", test_idx), ("dev", dev_idx)):
        for f in ids:
            cos = (fit_v @ vh_all[f]).abs()
            j = int(cos.argmax())
            rows.append(
                {
                    "split": split,
                    "index": f,
                    "max_abs_cos_vs_fit": float(cos[j]),
                    "argmax_fit_index": int(fit[j]),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "leakage.csv", index=False)

    print(f"\nfit={len(fit)} test={test_idx} dev={dev_idx}")
    print(f"max leakage cosine = {max(r['max_abs_cos_vs_fit'] for r in rows):.4f}\n")
    for r in payload["test"]:
        print(
            f"{r['index']:>6} freq={r['freq']:.5f} nd={r['n_distinct']:>2} "
            f"cm={r['content_mass']:.2f} var={r['var_along']:.2f} tilt={r['kappa_tilt']:.2f} "
            f"| {' '.join(r['top_tokens'][:8])}"
        )
    print(f"peak_gpu_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select"])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    select(a)
