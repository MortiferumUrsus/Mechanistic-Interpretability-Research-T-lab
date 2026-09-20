"""Cosines between every shared direction saved under checkpoints/shared_direction*.pt.

anatomy.py writes one `d_bar` per checkpoint it is run on (`--tag` picks the file name); this script
reads all of them and writes the pairwise cosine table the report cites when it compares the shared
component of two corrections (for example dir_hot against the entropy-penalised dir_ent).

    python dbar_cosines.py            # -> results/shared_direction_cosines.csv
"""

from __future__ import annotations

import pandas as pd
import torch

from common import CKPT, RESULTS


def main() -> None:
    files = sorted(CKPT.glob("shared_direction*.pt"))
    vecs = {}
    for f in files:
        blob = torch.load(f, map_location="cpu")
        d = blob["d_bar"].float()
        vecs[f.stem] = (d / d.norm().clamp_min(1e-9), blob.get("source"), blob.get("n_fit"))
    rows = []
    names = list(vecs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append(
                {
                    "a": a,
                    "a_source": vecs[a][1],
                    "b": b,
                    "b_source": vecs[b][1],
                    "cosine": float(torch.dot(vecs[a][0], vecs[b][0])),
                    "n_fit_a": vecs[a][2],
                    "n_fit_b": vecs[b][2],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "shared_direction_cosines.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
