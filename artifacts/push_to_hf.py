"""Publish the best checkpoint to a public Hugging Face repository.

The task requires it: "сохраните лучший адаптер или чекпойнт в открытый репозиторий на Huggingface".

Run by the account owner. Nothing in the pipeline calls this automatically, and it never will: publishing
is an outward-facing action under someone's account, so it stays a deliberate manual step. Use `--dry-run`
first -- it prints exactly what would be uploaded and where, and touches nothing.

    python push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction --dry-run
    hf auth login
    python push_to_hf.py --repo <user>/gpt2-small-steering-direction-correction
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

# direction_correction.pt is the round-2 method and the best artefact; the denoiser is included because the
# round-1 arms are reported against it and the card describes both.
FILES = ("direction_correction.pt", "denoiser.pt", "model.py", "README.md", "config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/gpt2-small-steering-direction-correction")
    ap.add_argument("--private", action="store_true", help="the task asks for a public repo; use only to rehearse")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="print what would be uploaded, upload nothing")
    args = ap.parse_args()

    present = [(n, (HERE / n)) for n in FILES if (HERE / n).exists()]
    missing = [n for n in FILES if not (HERE / n).exists()]
    total = sum(p.stat().st_size for _, p in present)

    print(f"target repository: {args.repo}  ({'private' if args.private else 'public'})")
    for name, path in present:
        print(f"  {name:28s} {path.stat().st_size / 1e6:8.2f} MB")
    if missing:
        raise SystemExit(f"required publication files are missing: {', '.join(missing)}")
    print(f"  total {total / 1e6:.2f} MB")

    if args.dry_run:
        print("\ndry run: nothing was created and nothing was uploaded")
        return
    if not present:
        raise SystemExit("nothing to upload")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(HERE),
        repo_id=args.repo,
        allow_patterns=list(FILES),
        commit_message="steering direction correction and denoiser, T-Lab 2026 mech-interp track",
    )
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
