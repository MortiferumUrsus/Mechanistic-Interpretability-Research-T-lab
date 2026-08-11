"""Per-feature SAE activation statistics over the corpus dump, for all latents.

Needed by the feature-surgery arm: the natural ceiling of every latent is what defines
"this latent is firing harder than it ever does on real text".
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from common import DATA, DEVICE, load_sae, open_memmap


@torch.no_grad()
def compute(chunk: int) -> None:
    sae = load_sae()
    f = sae.cfg.d_sae
    acts = open_memmap()
    n = acts.shape[0]

    max_act = torch.zeros(f, device=DEVICE)
    n_active = torch.zeros(f, device=DEVICE, dtype=torch.long)
    sum_act = torch.zeros(f, device=DEVICE, dtype=torch.float64)

    for i in tqdm(range(0, n, chunk), unit="chunk"):
        h = torch.from_numpy(np.ascontiguousarray(acts[i : i + chunk])).to(DEVICE).float()
        z = sae.encode(h)
        max_act = torch.maximum(max_act, z.max(0).values)
        n_active += (z > 0).sum(0)
        sum_act += z.sum(0).double()
        del h, z

    torch.save(
        {
            "max_act": max_act.cpu(),
            "freq": (n_active.double() / n).float().cpu(),
            "mean_act": (sum_act / n).float().cpu(),
            "n_tokens": n,
        },
        DATA / "sae_feature_stats.pt",
    )
    print(
        f"latents={f} n_tokens={n} "
        f"median_max={max_act.median().item():.3f} "
        f"dead={(n_active == 0).sum().item()} "
        f"peak_gpu_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=4096)
    compute(ap.parse_args().chunk)
