"""Does the linear-Gaussian closed form reproduce the measured transmission? Recomputed, not remembered.

The Wiener arm is `D(x) = mu + Sigma(Sigma + sigma^2 I)^-1 (x - mu)` (`denoiser.py`, `WienerDenoiser`), so the
transmission it produces along a direction must equal `v^T Sigma (Sigma + sigma^2 I)^-1 v` identically. That
makes the comparison a genuine parameter-free check: `sigma` comes from the arm's own definition -- `wiener:1.0`
is noise of total norm `median||h||`, hence `sigma^2 = median_norm^2 / d` -- and `Sigma` comes from the corpus.

Two things are worth separating, and the report's section 7.1 got them confused once:

- the **exact** form, which matches the measurement to four decimals per feature;
- the **eigendirection shortcut** `s/(s + sigma^2)`, which is what the report originally quoted. It holds only
  when `v` is an eigenvector of `Sigma`, and SAE decoder columns are not: `cos(Sigma v, v)` is about 0.5. Since
  `lambda/(lambda + sigma^2)` is concave, substituting one "average" variance for the average over the spectrum
  overstates the transmission -- one-sidedly, by algebra rather than by chance.

So a mismatch of the shortcut is a measurement of how non-eigen the decoder directions are, not a failure of
the model. This script prints both, and asserts the exact form's agreement, so the claim in the report cannot
drift away from the artefacts again.

    python closed_form_check.py --denoiser wiener:1.0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from activations import ActStats
from common import DATA, RESULTS, load_sae

# With the shrinkage matched to the measurement this is an algebraic identity of the linear denoiser, so
# the residual is float32 rounding of the stored operator (~1e-7), not an approximation error. The old
# tolerance of 0.005 was four orders of magnitude loose: it passed while the two scripts were using
# different covariances, which is exactly the regression it was supposed to catch.
TOL = 1e-4


def main(args) -> None:
    tr_path = RESULTS / "transmission.csv"
    if not tr_path.exists():
        raise SystemExit("results/transmission.csv is missing; run analysis.py transmission first")
    tr = pd.read_csv(tr_path)
    rows = tr[tr["denoiser"] == args.denoiser]
    if rows.empty:
        raise SystemExit(
            f"no rows for {args.denoiser!r}; present: {sorted(tr['denoiser'].unique())}"
        )
    measured = rows.groupby("feature")["tau"].mean()
    features = [int(f) for f in measured.index]

    # The measurement is on whichever split it was run for; resolve it rather than assuming TEST.
    splits = np.load(DATA / "splits.npz")
    which = [k for k in splits.files if set(features) <= {int(x) for x in splits[k]}]
    print(f"{args.denoiser}: {len(features)} features, split {which or ['unknown']}")

    stats = ActStats.load()
    import analysis as A

    scale = float(args.denoiser.split(":")[1])
    sigma = A.sigma_for_norm(scale * float(stats.median_norm))
    s2 = sigma**2
    print(f"median_norm {float(stats.median_norm):.4f} -> sigma {sigma:.4f}, sigma^2 {s2:.4f} (not fitted)")

    # The shrinkage comes from the table being checked, never from a default here. Two hand-maintained
    # defaults in two scripts is what produced the mismatch in the first place; `--shrink` is an override
    # for deliberate experiments, and its absence from the table is a hard failure rather than a guess.
    if "shrink" in rows.columns:
        gammas = sorted(set(round(float(g), 12) for g in rows["shrink"]))
        if len(gammas) != 1:
            raise SystemExit(f"transmission.csv mixes shrinkage values {gammas}; cannot check against one")
        shrink = gammas[0] if args.shrink is None else args.shrink
        if args.shrink is not None and abs(args.shrink - gammas[0]) > 1e-12:
            print(f"WARNING: overriding the table's shrinkage {gammas[0]} with {args.shrink}")
    elif args.shrink is not None:
        shrink = args.shrink
        print(f"transmission.csv predates the shrink column; using the value passed in: {shrink}")
    else:
        raise SystemExit(
            "transmission.csv has no `shrink` column, so the covariance it was measured with is unknown. "
            "Re-run `analysis.py transmission` (it records it now), or pass --shrink explicitly if you "
            "know the value it was produced with."
        )
    print(f"shrinkage taken from the measurement: {shrink}")
    cov = stats.shrunk_cov(shrink).double().cpu().numpy()
    d = cov.shape[0]
    M = np.linalg.solve(cov + s2 * np.eye(d), cov)

    sae = load_sae(device="cpu")
    w = sae.W_dec.detach().float().cpu()
    vh = (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)).numpy()

    out = []
    for f in features:
        v = vh[f].astype(np.float64)
        sv = cov @ v
        s_along = float(v @ sv)
        out.append(
            {
                "feature": f,
                "measured": float(measured[f]),
                "exact": float(v @ (M @ v)),
                "shortcut": s_along / (s_along + s2),
                "var_along": s_along,
                "cos_sigma_v": float(sv @ v / (np.linalg.norm(sv) * np.linalg.norm(v))),
            }
        )
    df = pd.DataFrame(out)
    df["err_exact"] = df["exact"] - df["measured"]
    df["err_shortcut"] = df["shortcut"] - df["measured"]
    df.to_csv(RESULTS / f"closed_form_{args.denoiser.replace(':', '_')}.csv", index=False)

    print()
    print(df.round(4).to_string(index=False))
    print()
    print(
        f"measured mean {df['measured'].mean():.4f} | exact {df['exact'].mean():.4f} "
        f"(max |err| {df['err_exact'].abs().max():.4f}) | shortcut {df['shortcut'].mean():.4f} "
        f"(max |err| {df['err_shortcut'].abs().max():.4f})"
    )
    print(
        f"cos(Sigma v, v) {df['cos_sigma_v'].min():.3f}..{df['cos_sigma_v'].max():.3f} "
        "-- an eigenvector, which the shortcut assumes, would give 1.000"
    )

    worst = df["err_exact"].abs().max()
    assert worst < TOL, (
        f"the exact closed form no longer reproduces the measurement (max error {worst:.4f} > {TOL}); "
        "the report's section 7.1 states that it does, so one of them is wrong"
    )
    # The shortcut must err upward everywhere, by Jensen; if it ever errs downward the argument is incomplete.
    assert (df["err_shortcut"] > 0).all(), (
        "the eigendirection shortcut under-predicts on some feature, which the concavity argument in the "
        "report says cannot happen -- check the argument before trusting the section"
    )
    print("\nexact form reproduces the measurement; shortcut overstates it on every feature, as the algebra says")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", default="wiener:1.0")
    ap.add_argument("--shrink", type=float, default=None,
                    help="override the shrinkage recorded in transmission.csv; normally read from the table")
    main(ap.parse_args())
