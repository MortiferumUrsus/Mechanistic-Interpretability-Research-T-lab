import sys

sys.path.insert(0, r"C:\Projects\Work\T-lab\01-mech-interp\src")
import torch
import yaml

from common import ROOT, ActStats, load_sae
from denoiser import transported_direction

stats = ActStats.load()
sae = load_sae()
feats = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))
print(f"{'split':5s} {'feat':>6s} {'var_along':>9s} {'kappa':>6s} {'|delta|/s':>10s} {'cos':>6s}")
for split in ("dev", "test"):
    for r in feats[split]:
        f = int(r["index"])
        v = sae.W_dec[f].detach().float()
        vh = v / v.norm()
        for gamma in (0.01,):
            d = transported_direction(stats, vh, gamma)
            ratio = float(d.norm())
            cos = float(torch.nn.functional.cosine_similarity(d, vh, dim=0))
            print(
                f"{split:5s} {f:6d} {r['var_along']:9.2f} {r['kappa_tilt']:6.2f} {ratio:10.2f} {cos:6.3f}"
            )
