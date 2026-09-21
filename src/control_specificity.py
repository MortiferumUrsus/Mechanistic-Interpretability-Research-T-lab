"""Is the gain specific to the steered feature, or a generic effect?

A direction correction that improved both axes could be cheating: if it nudged every direction toward
some generically fluent, generically evocative region, the concept metric would rise without the
intervention being about that feature at all. The control scores every generation twice, once with its
own feature's keyword list and once with the lists of the other features. A feature-specific effect
shows a gain on the matched list and none on the mismatched ones.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from common import RESULTS, ROOT, load_features


def hit(text: str, prompt: str, keywords: list[str]) -> float:
    low, plow = text.lower(), prompt.lower()
    fresh = [k for k in keywords if k not in plow]
    return float(any(k in low for k in fresh))


def main(args) -> None:
    recs = load_features(args.split)
    kw = {int(r["index"]): [k.lower() for k in r["keywords"]] for r in recs}
    rows = [json.loads(l) for l in (RESULTS / args.gen).open(encoding="utf-8-sig") if l.strip()]

    out = []
    for r in rows:
        f = int(r["feature"])
        matched = hit(r["text"], r["prompt"], kw[f])
        others = [hit(r["text"], r["prompt"], kw[g]) for g in kw if g != f]
        out.append(
            {
                "arm": r["arm"],
                "c": r["c"],
                "feature": f,
                "matched": matched,
                "mismatched": sum(others) / max(len(others), 1),
            }
        )
    df = pd.DataFrame(out)
    g = df.groupby(["arm", "c"])[["matched", "mismatched"]].mean().round(4)
    g["specificity"] = (g["matched"] - g["mismatched"]).round(4)
    g.to_csv(RESULTS / args.out)
    print(g.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_dev_r2.jsonl")
    ap.add_argument("--split", default="test", choices=["test", "dev", "test_r3", "test_r4"])
    ap.add_argument("--out", default="control_specificity_dev.csv")
    main(ap.parse_args())
