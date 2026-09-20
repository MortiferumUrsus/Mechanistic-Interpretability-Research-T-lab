"""Select a fresh, never-seen feature set for round four, decorrelated from rounds one AND three.

Round three fixed the "TEST opened twice" caveat with twelve fresh features, but the interval on
twelve features is wide and the DEV prompts used to freeze hyper-parameters are a prefix of the
TEST prompts, so nothing about the evaluation prompts is actually disjoint from selection. Round
four widens the feature set to 48 and (via `scripts/run_expE.py`) pairs it with a prompt pool that
does not overlap DEV/TEST at all.

Like round three, this is additive: it writes the `test_r4` split to `configs/features_r4.yaml`
(features.yaml itself is frozen) and `data/splits_r4.npz`, never touching `splits.npz`, `test`, `dev` or `test_r3`. Candidates
are drawn from the same objective rule as rounds one and three, restricted to `fit` (already
decorrelated from test/dev by construction) and additionally required to have |cos| < 0.3 against
every round-three feature too -- round three was frozen after round one's TEST numbers were public,
so round four must not lean on anything that could be entangled with it.

Around the 48 selected features there is also a buffer: any other FIT direction (net of the round-
three holdout) within |cos| >= 0.3 of one of the 48 cannot be used to train the correction either,
or evaluation would leak into training. That exclusion is applied by `train_direction.py` (an
unconditional gate, mirroring its round-three holdout), not here; this script only proves the
buffer is small enough to leave a training pool and reports its size.

    python select_round4.py --seed 4242 --n 48
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
    for name in ("test", "dev", "fit"):
        if name not in frozen:
            raise SystemExit(f"splits.npz has no '{name}' split; round four cannot be constructed")
    r3_path = DATA / "splits_r3.npz"
    if not r3_path.exists():
        raise SystemExit(
            "data/splits_r3.npz is missing; round four must be decorrelated from round three too, "
            "run select_round3.py first"
        )
    r3 = np.load(r3_path)
    if "test_r3" not in r3:
        raise SystemExit("splits_r3.npz has no 'test_r3' split")

    test_idx = set(int(x) for x in frozen["test"])
    dev_idx = set(int(x) for x in frozen["dev"])
    r3_idx = set(int(x) for x in r3["test_r3"])
    used = test_idx | dev_idx | r3_idx
    eligible = set(int(x) for x in frozen["fit"])
    print(
        f"frozen: test {len(test_idx)}, dev {len(dev_idx)}, test_r3 {len(r3_idx)}, "
        f"fit {len(eligible)}"
    )

    qualify = surv[
        (surv["n_distinct"] >= MIN_DISTINCT)
        & (surv["n_distinct"] <= MAX_DISTINCT)
        & (surv["content_mass"] >= MIN_CONTENT_MASS)
        & (~surv["blocked"].astype(bool))
    ]
    pool = [int(x) for x in qualify["index"].to_numpy() if int(x) in eligible and int(x) not in used]
    print(
        f"qualifying under the same rule: {len(qualify)}; available after excluding "
        f"test/dev/test_r3: {len(pool)}"
    )

    sae = load_sae(device="cpu")
    w = sae.W_dec.detach().float()
    vh_t = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vh = vh_t.cpu().numpy()

    def decorrelated(candidates: list[int], forbidden: list[int], n: int) -> list[int]:
        """Greedy pick with |cos| < COS_MAX against test/dev/test_r3 and against each other."""
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

    # Prove the properties the round depends on, rather than assuming them.
    assert not (set(picked) & used), "a round-four feature is already in test/dev/test_r3"
    assert set(picked) <= eligible, "a round-four feature is outside `fit`, so it cannot be held out of training"
    cross = np.abs(vh[np.asarray(picked)] @ vh[np.asarray(sorted(used))].T).max()
    within = 0.0
    if len(picked) > 1:
        g = np.abs(vh[np.asarray(picked)] @ vh[np.asarray(picked)].T)
        np.fill_diagonal(g, 0.0)
        within = float(g.max())
    print(f"max |cos| to test/dev/test_r3: {cross:.4f}; within the new set: {within:.4f} (limit {COS_MAX})")
    assert cross < COS_MAX and within < COS_MAX

    # Buffer: of the pool `train_direction.py` would otherwise train on (fit, minus the round-three
    # holdout), how many directions fall within |cos| >= 0.3 of one of the 48 and would therefore be
    # excluded from training too. This does not write anything -- the exclusion itself lives in
    # train_direction.py -- it only proves a buffer of this kind leaves a nonempty training pool.
    from features import clean_keywords, fit_indices, mass_prefix, token_mass, top_activations

    safe_from_r4 = set(int(x) for x in fit_indices(vh_t, sorted(picked), args.chunk))
    fit_minus_r3 = eligible - r3_idx
    buffer = fit_minus_r3 - set(picked) - safe_from_r4
    remaining = fit_minus_r3 - set(picked) - buffer
    assert len(buffer) < len(fit_minus_r3), "the round-four buffer would swallow the entire training pool"
    assert len(remaining) > 0, "no training pool would remain after the round-four buffer"
    print(
        f"buffer: {len(buffer)} fit-pool features (after the round-three holdout, {len(fit_minus_r3)} "
        f"total) fall within |cos| >= {COS_MAX} of the round-four set and would be excluded from "
        f"training by train_direction.py; training pool remaining ~ {len(remaining)}"
    )

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

    print(f"\n{'[dry-run] ' if args.dry_run else ''}picked {len(picked)} round-four features: {picked}")
    for r in records:
        print(f"  {r['index']:6d}  {', '.join(r['keywords'][:6])}")

    if args.dry_run:
        print("\n--dry-run: nothing written to disk")
        return

    # Written to its own file: configs/features.yaml is frozen (hashed by evaluation/ai_annotation) and
    # src/common.py:load_features merges configs/features_*.yaml into it when a split is requested.
    path = CONFIGS / "features_r4.yaml"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} already exists; pass --overwrite to replace it")
    header = (
        "# Round four: 48 fresh SAE features selected by src/select_round4.py from the FIT pool,\n"
        "# decorrelated (|cos| < 0.3) from every test, dev and test_r3 direction. Kept in a separate file so that\n"
        "# configs/features.yaml stays byte-identical to the version hashed by evaluation/ai_annotation/.\n"
        "# src/common.py:load_features merges this file with features.yaml when a script asks for the 'test_r4' split.\n"
    )
    path.write_text(header + yaml.safe_dump({"test_r4": records}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    np.savez(DATA / "splits_r4.npz", test_r4=np.asarray(picked, dtype=np.int32))

    print(f"\nwrote test_r4 with {len(records)} features to {path.relative_to(ROOT)}; configs/features.yaml untouched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="compute selection and buffer, print them, write nothing to disk",
    )
    main(ap.parse_args())
