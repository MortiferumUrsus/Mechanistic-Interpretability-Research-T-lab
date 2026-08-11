"""Denoisers D: R^d -> R^d. All learned variants predict the residual D(x) - x."""

from __future__ import annotations

import torch
import torch.nn as nn

from common import ActStats, D_MODEL


class LinearDenoiser(nn.Module):
    def __init__(self, d: int = D_MODEL, mean: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d) if mean is None else mean.clone())
        self.proj = nn.Linear(d, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.proj(x - self.mean)


class MLPDenoiser(nn.Module):
    """Pre-LN residual MLP, optionally conditioned on the perturbation magnitude."""

    def __init__(
        self,
        d: int = D_MODEL,
        hidden: int = 1536,
        mean: torch.Tensor | None = None,
        scale: float = 1.0,
        cond: bool = False,
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d) if mean is None else mean.clone())
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.act = nn.GELU()
        self.cond = cond
        if cond:
            self.cond_emb = nn.Linear(1, hidden, bias=False)
            nn.init.zeros_(self.cond_emb.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        z = self.fc1(self.norm((x - self.mean) / self.scale))
        if self.cond:
            if s is None:
                s = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            elif s.dim() == x.dim() - 1:
                s = s.unsqueeze(-1)
            z = z + self.cond_emb(s / self.scale)
        return x + self.fc2(self.act(z)) * self.scale


class WienerDenoiser(nn.Module):
    """Closed-form linear-Gaussian posterior mean: mu + Sigma (Sigma + s2 I)^-1 (x - mu).

    Exact only for Gaussian activations under isotropic noise of known variance; used as a
    linear baseline and as the analytic handle for the theory, not as a claim of optimality
    for rank-one steering.
    """

    def __init__(self, stats: ActStats, sigma: float, shrink: float = 0.0):
        super().__init__()
        cov = stats.shrunk_cov(shrink).double()
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        a = torch.linalg.solve(cov + (sigma**2) * eye, cov).T
        self.register_buffer("mean", stats.mean.clone())
        self.register_buffer("A", a.float())
        self.sigma = sigma
        self.shrink = shrink

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        return self.mean + (x - self.mean) @ self.A.T


def transported_direction(stats: ActStats, v_hat: torch.Tensor, shrink: float) -> torch.Tensor:
    """Minimum-Mahalanobis shift with a unit increment of the concept coordinate.

    Solves argmin d^T Sigma^-1 d subject to v_hat^T d = 1, giving Sigma v / (v^T Sigma v).
    """
    cov = stats.shrunk_cov(shrink)
    sv = cov @ v_hat
    return sv / torch.dot(v_hat, sv)


def build_denoiser(kind: str, stats: ActStats, **kw) -> nn.Module:
    if kind == "linear":
        return LinearDenoiser(mean=stats.mean)
    if kind == "mlp":
        return MLPDenoiser(
            hidden=kw.get("hidden", 1536),
            mean=stats.mean,
            scale=kw.get("scale", stats.median_norm),
            cond=kw.get("cond", False),
        )
    if kind == "wiener":
        return WienerDenoiser(stats, sigma=kw["sigma"], shrink=kw.get("shrink", 0.0))
    raise ValueError(kind)
