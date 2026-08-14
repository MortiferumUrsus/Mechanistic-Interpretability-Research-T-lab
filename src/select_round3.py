"""Select a fresh, never-seen feature set for round three, without touching the frozen splits.

The protocol says TEST opens once. For round one that held; for round two it did not, because the method was
chosen after seeing round one's TEST numbers. The report says so, and the honest way to remove the caveat is
not better wording but a set of features that nothing in this work has looked at.

That set has to be genuinely fresh and cheap to produce, and both are available: the selection rule is
objective (`features.py`, 4912 latents qualify out of 24576), and the per-feature statistics it needs are
already computed in `results/feature_stats.csv`. So round three costs one generation pass, not a re-run of the
pipeline.

This script is deliberately **additive**. It appends a `test_r3` key to `configs/features.yaml` and writes
`data/splits_r3.npz`; it never rewrites `splits.npz`, `test` or `dev`, because those are what rounds one and
two were run against and overwriting them would destroy the reproducibility the report claims. It also
refuses to pick anything correlated with the frozen features, on the same `|cos| < 0.3` rule.

    python select_round3.py --seed 777 --n 12
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from common import DATA, RESULTS, ROOT, load_model, load_sae

CONFIGS = ROOT / "configs"

COS_MAX = 0.3
MIN_DISTINCT, MAX_DISTINCT, MIN_CONTENT_MASS = 3, 25, 0.7


def main(args) -> None:
    stats_path = RESULTS / "feature_stats.csv"
    if not stats_path.exists():
        raise SystemExit("results/feature_stats.csv is missing; run features.py select first")
    surv = pd.read_csv(stats_path)

    frozen = np.load(DATA / "splits.npz")
    # `fit` belongs in the exclusion set, and leaving it out was the whole point of failure: round 3 exists
    # to be a set of features that nothing in this work has touched, and `fit` is exactly the pool the
    # direction correction was TRAINED on. Excluding only test and dev produced twelve "fresh" features
    # every one of which the correction had already seen, so the transfer claim it was meant to support was
    # measuring memorisation. The assert below makes the omission impossible to repeat silently.
    used = set()
    for name in ("test", "dev", "fit"):
        if name in frozen:
            used |= set(int(x) for x in frozen[name])
    for name in ("test", "dev", "fit"):
        if name not in frozen:
            raise SystemExit(f"splits.npz has no '{name}' split; round 3 cannot prove freshness without it")
    print(f"frozen features: {len(used)} "
          f"(test {len(frozen['test'])}, dev {len(frozen['dev'])}, fit {len(frozen['fit'])})")

    qualify = surv[
        (surv["n_distinct"] >= MIN_DISTINCT)
        & (surv["n_distinct"] <= MAX_DISTINCT)
        & (surv["content_mass"] >= MIN_CONTENT_MASS)
        & (~surv["blocked"].astype(bool))
    ]
    pool = [int(x) for x in qualify["index"].to_numpy() if int(x) not in used]
    print(f"qualifying under the same rule: {len(qualify)}; available after excluding the frozen: {len(pool)}")

    sae = load_sae(device="cpu")
    w = sae.W_dec.detach().float()
    vh = (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)).cpu().numpy()

    def decorrelated(candidates: list[int], forbidden: list[int], n: int) -> list[int]:
        """Greedy pick with |cos| < COS_MAX against the frozen features and against each other."""
        chosen: list[int] = []
        block = np.asarray(forbidden, dtype=int)
        for f in candidates:
            v = vh[f]
            if block.size and np.abs(vh[block] @ v).max() >= COS_MAX:
                continue
            if chosen and np.abs(vh[np.asarray(chosen)] @ v).max() >= COS_MAX:
                continue
            chosen.append(f)
            if len(chosen) == n:
                break
        return chosen

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool)
    picked = decorrelated(pool, sorted(used), args.n)
    if len(picked) < args.n:
        raise SystemExit(f"only {len(picked)} decorrelated fresh features found; lower --n or relax the rule")

    # Prove the two properties the round depends on, rather than assuming them.
    assert not (set(picked) & used), "a round-three feature is already in the frozen splits"
    cross = np.abs(vh[np.asarray(picked)] @ vh[np.asarray(sorted(used))].T).max()
    within = 0.0
    if len(picked) > 1:
        g = np.abs(vh[np.asarray(picked)] @ vh[np.asarray(picked)].T)
        np.fill_diagonal(g, 0.0)
        within = float(g.max())
    print(f"max |cos| to the frozen features: {cross:.4f}; within the new set: {within:.4f} (limit {COS_MAX})")
    assert cross < COS_MAX and within < COS_MAX

    from features import clean_keywords, mass_prefix, token_mass, top_activations

    top_val, top_tok, freq, mean_act = top_activations(args.chunk)
    model = load_model(device="cpu")
    tok = model.tokenizer
    cache: dict[int, str] = {}

    def decode(t: int) -> str:
        if t not in cache:
            cache[t] = tok.decode([t])
        return cache[t]

    records = []
    for f in picked:
        mass = token_mass(top_val[:, f].cpu().numpy(), top_tok[:, f].cpu().numpy())
        row = surv[surv["index"] == f].iloc[0]
        strings = [decode(t) for t, _ in mass]
        records.append(
            {
                "index": int(f),
                "freq": float(f"{row.freq:.6g}"),
                "mean_act": float(f"{row.mean_act:.6g}"),
                "n_distinct": int(row.n_distinct),
                "content_mass": float(f"{row.content_mass:.6g}"),
                "dec_norm": float(f"{row.dec_norm:.6g}"),
                "var_along": float(f"{row.var_along:.6g}"),
                "kappa_tilt": float(f"{row.kappa_tilt:.6g}"),
                "top_tokens": [s.replace(" ", "_", 1) if s.startswith(" ") else s for s in strings[:15]],
                "keywords": clean_keywords([decode(t) for t, _ in mass_prefix(mass)]),
            }
        )

    path = CONFIGS / "features.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "test_r3" in payload and not args.overwrite:
        raise SystemExit("configs/features.yaml already has test_r3; pass --overwrite to replace it")
    before = {k: len(v) for k, v in payload.items()}
    payload["test_r3"] = records
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    after = {k: len(v) for k, v in payload.items()}
    assert all(before[k] == after[k] for k in before), "an existing split was modified; aborting"
    np.savez(DATA / "splits_r3.npz", test_r3=np.asarray(picked, dtype=np.int32))

    print(f"\nappended test_r3 with {len(records)} features; existing splits unchanged {before}")
    for r in records:
        print(f"  {r['index']:6d}  {', '.join(r['keywords'][:6])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
