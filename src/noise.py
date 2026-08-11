"""Training perturbations for the denoiser.

The deployment perturbation is rank one, of large norm, and aligned with a dictionary
direction, so the training distribution is built to match those three properties. The
"same direction across the whole sequence" property of steering is deliberately not
modelled: the denoiser is token-wise and cannot observe neighbouring positions, so a
shared direction is statistically indistinguishable from independent ones for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NoiseConfig:
    kind: str = "mix"  # gauss | dict | mix
    p_dict: float = 0.5
    max_atoms: int = 4
    s_min_rel: float = 0.05
    s_max_rel: float = 3.5
    p_zero: float = 0.1  # fraction of clean examples, enforces D(h) ~ h
    fixed_rel: float = 0.0  # > 0 pins the magnitude, for the fixed-sigma ablation


def sample_directions(
    n: int, d: int, dictionary: torch.Tensor, cfg: NoiseConfig, gen: torch.Generator
) -> torch.Tensor:
    dev = dictionary.device
    u = torch.randn(n, d, device=dev, generator=gen)
    if cfg.kind in ("dict", "mix"):
        n_atoms = torch.randint(1, cfg.max_atoms + 1, (1,), device=dev, generator=gen).item()
        idx = torch.randint(0, dictionary.shape[0], (n, n_atoms), device=dev, generator=gen)
        w = torch.randn(n, n_atoms, device=dev, generator=gen)
        combo = torch.einsum("na,nad->nd", w, dictionary[idx])
        if cfg.kind == "dict":
            u = combo
        else:
            take = torch.rand(n, 1, device=dev, generator=gen) < cfg.p_dict
            u = torch.where(take, combo, u)
    return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def sample_magnitudes(
    n: int, scale: float, cfg: NoiseConfig, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    if cfg.fixed_rel > 0:
        return torch.full((n,), cfg.fixed_rel * scale, device=device)
    lo, hi = cfg.s_min_rel * scale, cfg.s_max_rel * scale
    r = torch.rand(n, device=device, generator=gen)
    s = torch.exp(r * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo))).to(device) + torch.log(torch.tensor(lo)).to(device))
    zero = torch.rand(n, device=device, generator=gen) < cfg.p_zero
    return torch.where(zero, torch.zeros_like(s), s)


def perturb(
    h: torch.Tensor, dictionary: torch.Tensor, scale: float, cfg: NoiseConfig, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = h.shape
    u = sample_directions(n, d, dictionary, cfg, gen)
    s = sample_magnitudes(n, scale, cfg, gen, h.device)
    return h + s.unsqueeze(-1) * u, s
