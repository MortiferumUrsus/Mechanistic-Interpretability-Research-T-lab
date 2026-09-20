"""Push, poll, or fetch output for a kernel directory produced by make_kernel.py.

    python push.py --dir build/smoke                          # dry-run (default): print the command, do nothing
    python push.py --dir build/smoke --push                    # actually `kaggle kernels push`
    python push.py --status tlab-mi-smoke                       # poll run status
    python push.py --fetch tlab-mi-smoke --out build/smoke/out  # download kernel output

Needs a Kaggle API token: either KAGGLE_API_TOKEN in the environment, or --token-file pointing at a
one-line file holding it (the environment variable wins if both are given). The token is set only on
this script's subprocess environment, never printed and never written anywhere. The Kaggle account
name comes from KAGGLE_USERNAME, defaulting to mortiferumursus.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

USERNAME = os.environ.get("KAGGLE_USERNAME", "mortiferumursus")


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


def _run(cmd: list[str], env: dict[str, str]) -> None:
    import subprocess

    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"kaggle CLI exited with code {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, help="kernel directory (with <name>.ipynb + kernel-metadata.json) to push")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the push command without running it; this is the default for --dir")
    ap.add_argument("--push", action="store_true", help="actually run `kaggle kernels push`")
    ap.add_argument("--status", metavar="SLUG", help="poll `kaggle kernels status` for USERNAME/SLUG")
    ap.add_argument("--fetch", metavar="SLUG", help="download output via `kaggle kernels output` for USERNAME/SLUG")
    ap.add_argument("--out", type=Path, help="output directory for --fetch")
    ap.add_argument("--token-file", type=Path, default=None,
                    help="file holding the Kaggle API token, one line (never printed); "
                         "used when KAGGLE_API_TOKEN is not set in the environment")
    args = ap.parse_args()

    actions = [a for a in (args.dir, args.status, args.fetch) if a]
    if len(actions) != 1:
        ap.error("pass exactly one of --dir, --status, --fetch")

    if args.dir:
        if not (args.dir / "kernel-metadata.json").is_file():
            raise SystemExit(f"no kernel-metadata.json in {args.dir}")
        cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(args.dir)]
        dry_run = args.dry_run or not args.push
        if args.dry_run and args.push:
            print("both --dry-run and --push given; --dry-run wins, nothing will be pushed")
        if dry_run:
            print("dry-run: would run:", " ".join(cmd))
            print("pass --push to actually push")
            return
        env = _load_token_env(args.token_file)
        _run(cmd, env)
        return

    if args.status:
        env = _load_token_env(args.token_file)
        cmd = [sys.executable, "-m", "kaggle", "kernels", "status", f"{USERNAME}/{args.status}"]
        _run(cmd, env)
        return

    if args.fetch:
        if not args.out:
            ap.error("--fetch requires --out")
        args.out.mkdir(parents=True, exist_ok=True)
        env = _load_token_env(args.token_file)
        cmd = [sys.executable, "-m", "kaggle", "kernels", "output",
               f"{USERNAME}/{args.fetch}", "-p", str(args.out)]
        _run(cmd, env)
        return


if __name__ == "__main__":
    main()
