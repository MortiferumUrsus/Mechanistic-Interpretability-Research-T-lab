"""Assemble the portable Kaggle payload (and, only with --push, upload it as a private dataset).

The payload is everything the Kaggle kernel needs to reproduce the pipeline except the two huge
activation-dump files (acts.f16 is 2.3 GB, toks.i32 a few MB) -- those are rebuilt inside the kernel
by the data-restore step (see make_kernel.py), not shipped.

    python build_payload.py                       # dry-run (default): stage + print sizes, no upload
    python build_payload.py --out /tmp/stage       # stage into a specific directory instead of temp
    python build_payload.py --push                 # actually create/version the private dataset

--push needs a Kaggle API token: either KAGGLE_API_TOKEN in the environment, or --token-file pointing
at a one-line file holding it (the environment variable wins if both are given). The token is set only
on this script's subprocess environment, never printed and never written anywhere. The Kaggle account
name comes from KAGGLE_USERNAME, defaulting to mortiferumursus.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
USERNAME = os.environ.get("KAGGLE_USERNAME", "mortiferumursus")
DATASET_SLUG = "tlab-mi-payload"

# Whole directories copied verbatim.
COPY_DIRS = ["src", "scripts", "configs"]

# From data/: only the small artifacts. acts.f16 and toks.i32 (2.3 GB / a few MB) are deliberately
# excluded -- the kernel regenerates them via activations.py dump + sae_stats.py.
DATA_FILES = [
    "act_stats.pt",
    "act_stats.json",
    "splits.npz",
    "splits_r3.npz",
    "prompts.json",
    "sae_feature_stats.pt",
]

# From results/: the few artifacts later stages read as INPUT rather than write as output.
# feature_stats.csv is the corpus profile of every SAE latent; select_round4.py refuses to run
# without it, which is exactly how round four died on the first night.
RESULT_FILES = [
    "feature_stats.csv",
    "gen_expE_r4.jsonl",  # round-four generations, scored in a follow-up kernel
]

# From checkpoints/: only what downstream stages need as a starting point.
CHECKPOINT_FILES = [
    "dir_hot.pt",
    "mlp_gauss_cond1_s0.pt",
    "shared_direction.pt",  # layer-7 d_bar (anatomy.py output), reference for the second-layer replication
]

# Directories that must never end up in the payload, even by accident (defensive; COPY_DIRS above
# already excludes them by construction, but a future edit to COPY_DIRS should not silently start
# pulling these in).
NEVER_COPY = {".venv", "results", "kb", "artifacts", "lifelong-agents-fix", ".git", "__pycache__"}


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _copytree_clean(src: Path, dst: Path) -> None:
    """Copy a directory tree, skipping NEVER_COPY entries and __pycache__/*.pyc anywhere inside."""

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        return {n for n in names if n in NEVER_COPY or n.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in COPY_DIRS:
        src = ROOT / name
        if not src.is_dir():
            raise SystemExit(f"expected directory missing: {src}")
        _copytree_clean(src, out / name)

    data_out = out / "data"
    data_out.mkdir(exist_ok=True)
    for name in DATA_FILES:
        src = ROOT / "data" / name
        if src.is_file():
            shutil.copy2(src, data_out / name)
        else:
            print(f"  warning: {src} not found, skipping")

    results_out = out / "results"
    results_out.mkdir(exist_ok=True)
    for name in RESULT_FILES:
        src = ROOT / "results" / name
        if src.is_file():
            shutil.copy2(src, results_out / name)
        else:
            print(f"  warning: {src} not found, skipping")

    ckpt_out = out / "checkpoints"
    ckpt_out.mkdir(exist_ok=True)
    for name in CHECKPOINT_FILES:
        src = ROOT / "checkpoints" / name
        if src.is_file():
            shutil.copy2(src, ckpt_out / name)
        else:
            print(f"  warning: {src} not found, skipping")

    # kaggle/smoke_check.py rides along in scripts/ so the smoke kernel can run it without a
    # separate upload -- it is generated code from this same kaggle/ tree, not repo source.
    smoke_check = HERE / "smoke_check.py"
    if smoke_check.is_file():
        shutil.copy2(smoke_check, out / "scripts" / "smoke_check.py")
    else:
        print(f"  warning: {smoke_check} not found, smoke kernel will have nothing to run")

    for guard in NEVER_COPY:
        if guard == "results":
            # results/ is 47 MB of run output and must never ship wholesale, but a few files in it
            # are INPUTS to later stages. Allow exactly the whitelist, reject anything else.
            staged = {p.name for p in (out / "results").iterdir()} if (out / "results").is_dir() else set()
            extra = staged - set(RESULT_FILES)
            if extra:
                raise SystemExit(
                    f"payload safety check failed: unexpected files staged under results/: {sorted(extra)}"
                )
            continue
        if (out / guard).exists():
            raise SystemExit(f"payload safety check failed: {guard!r} ended up in the staged payload")

    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": DATASET_SLUG,
                "id": f"{USERNAME}/{DATASET_SLUG}",
                "licenses": [{"name": "unknown"}],
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def print_sizes(out: Path) -> None:
    print(f"\npayload staged at {out}")
    total = 0
    for child in sorted(out.iterdir()):
        size = _dir_size(child) if child.is_dir() else child.stat().st_size
        total += size
        print(f"  {child.name:20s} {size / 1e6:8.2f} MB")
    print(f"  {'TOTAL':20s} {total / 1e6:8.2f} MB")
    if total > 100_000_000:
        print(
            "\n  WARNING: payload exceeds 100 MB -- check COPY_DIRS / DATA_FILES / CHECKPOINT_FILES "
            "for something that should not be there (acts.f16 and toks.i32 must never be included)."
        )


def _load_token_env(token_file: Path | None) -> dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token and token_file is not None:
        if not token_file.is_file():
            raise SystemExit(f"token file not found: {token_file}")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise SystemExit(f"token file is empty: {token_file}")
    if not token:
        raise SystemExit("no Kaggle API token: set KAGGLE_API_TOKEN or pass --token-file")
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = token
    return env


def _verify_private_by_default(env: dict[str, str]) -> None:
    """Fail closed unless `kaggle datasets create --help` documents -u/--public as opt-in.

    CLI 2.2.4's own --help text says '-u, --public  Create publicly (default is private)', i.e. a
    plain `datasets create` with no -u/--public flag is private. If a future CLI version changes
    this wording we can no longer be sure of the default, so refuse to upload rather than guess.
    """
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "create", "--help"],
        capture_output=True, text=True, env=env,
    )
    help_text = (r.stdout + r.stderr).lower()
    if "--public" not in help_text or "default is private" not in help_text:
        raise SystemExit(
            "cannot confirm this Kaggle CLI defaults datasets to private "
            "(expected '--public ... default is private' in `kaggle datasets create --help`); "
            "refusing to upload. Inspect the CLI docs and update this check before pushing."
        )
    print("confirmed: this Kaggle CLI creates datasets private by default (no -u/--public passed).")


def _dataset_exists(env: dict[str, str]) -> bool:
    dataset_id = f"{USERNAME}/{DATASET_SLUG}"
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "files", dataset_id],
        capture_output=True, text=True, env=env,
    )
    # Observed behaviour on CLI 2.2.4: a dataset that does not exist yet under our own account
    # returns a non-zero exit (403 Forbidden from the ListDatasetFiles endpoint), not a clean 404.
    return r.returncode == 0


def push(out: Path, message: str, token_file: Path | None) -> None:
    env = _load_token_env(token_file)
    _verify_private_by_default(env)
    exists = _dataset_exists(env)
    if exists:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "version",
               "-p", str(out), "-m", message, "--dir-mode", "zip"]
        print(f"dataset {USERNAME}/{DATASET_SLUG} exists -> creating a new version")
    else:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "create",
               "-p", str(out), "--dir-mode", "zip"]
        print(f"dataset {USERNAME}/{DATASET_SLUG} does not exist yet -> creating it (private)")
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"kaggle CLI exited with code {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "tlab-mi-payload",
                    help="staging directory (default: a folder under the system temp dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and print sizes only; this is the default behaviour")
    ap.add_argument("--push", action="store_true",
                    help="actually create/version the private Kaggle dataset from the staged payload")
    ap.add_argument("--message", default="update payload",
                    help="version message used when the dataset already exists")
    ap.add_argument("--token-file", type=Path, default=None,
                    help="file holding the Kaggle API token, one line (never printed); "
                         "used when KAGGLE_API_TOKEN is not set in the environment")
    args = ap.parse_args()

    dry_run = args.dry_run or not args.push
    if args.dry_run and args.push:
        print("both --dry-run and --push given; --dry-run wins, nothing will be uploaded")

    stage(args.out)
    print_sizes(args.out)

    if dry_run:
        print("\ndry-run: payload staged, nothing uploaded. Pass --push to actually upload.")
        return

    push(args.out, args.message, args.token_file)


if __name__ == "__main__":
    main()
