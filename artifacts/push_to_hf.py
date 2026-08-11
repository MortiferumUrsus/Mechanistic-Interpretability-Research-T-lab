"""Publish the best denoiser checkpoint to a public Hugging Face repository.

Run by the account owner; nothing in the pipeline calls this automatically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/gpt2-small-steering-denoiser")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    staging = HERE / "_upload"
    staging.mkdir(exist_ok=True)
    for name in ("direction_correction.pt", "denoiser.pt", "README.md", "config.json"):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, staging / name)
    api.upload_folder(folder_path=str(staging), repo_id=args.repo, commit_message="steering denoiser")
    shutil.rmtree(staging)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
