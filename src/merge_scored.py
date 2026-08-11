"""Merge scored files from several rounds into one table for the combined figure.

The reference arm appears in every round's file; the first occurrence is kept so that all arms are
compared against one measurement of it rather than against per-round copies.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import RESULTS


def main(args) -> None:
    frames = []
    seen: set[str] = set()
    for name in args.inputs.split(","):
        df = pd.read_csv(RESULTS / name)
        keep = df[~df["arm"].isin(seen)]
        seen.update(df["arm"].unique())
        frames.append(keep)
        print(f"{name}: {len(df)} rows, kept {len(keep)} ({sorted(keep['arm'].unique())})")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / args.out, index=False)
    print(f"merged {len(out)} rows -> {args.out}; arms {sorted(out['arm'].unique())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="scored_test.csv,scored_test_r2.csv")
    ap.add_argument("--out", default="scored_test_all.csv")
    main(ap.parse_args())
