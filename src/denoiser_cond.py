"""Conditional denoiser arms: pull a steered activation toward the manifold of activations
where the *injected concept itself* is naturally expressed, instead of toward the general
corpus manifold (see src/concept_data.py for why that distinction matters and how the
counterfactual pairs h_plus / h_minus are built).

Both arms below implement the fixed steering-arm interface used throughout steering.py:
`f(h, v_hat, s) -> h_tilde`. Both are exact identities at s = 0 via an explicit branch --
this is a hard invariant of the whole study (every arm must agree at zero steering force),
so it is enforced directly rather than left to fall out of the (trained or closed-form)
machinery, which could differ from the identity by floating-point noise.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from common import D_MODEL, ActStats

# Shrinkage applied to the shared conditional covariance before inverting it. The fixed
# `load(...)` signature below has no shrink argument to plumb through (the task interface is
# frozen), so this mirrors -- as a module constant rather than a call-site knob -- the
# magnitude already frozen for the unconditional WienerDenoiser (configs/frozen.yaml: 0.01).
_COV_SHRINK = 0.01


class MLPDenoiserCondDir(nn.Module):
    """MLPDenoiser (src/denoiser.py) conditioned on strength AND on the steering direction.

    Same pre-LN residual MLP predicting a residual, zero-initialised to the identity at the
    start of training. In addition to the existing scalar `cond_emb(s)`, a second
    zero-initialised linear layer folds in v_hat, so the network can learn what a natural
    expression of *this* concept looks like rather than one fixed correction shared by every
    steering direction.
    """

    def __init__(self, d: int = D_MODEL, hidden: int = 1536, mean: Tensor | None = None, scale: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d) if mean is None else mean.clone())
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.act = nn.GELU()
        self.cond_emb = nn.Linear(1, hidden, bias=False)
        self.dir_emb = nn.Linear(d, hidden, bias=False)
        nn.init.zeros_(self.cond_emb.weight)
        nn.init.zeros_(self.dir_emb.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor, v_hat: Tensor, s) -> Tensor:
        z = self.fc1(self.norm((x - self.mean) / self.scale))
        if not torch.is_tensor(s):
            s = torch.full(x.shape[:-1], float(s), device=x.device, dtype=x.dtype)
        if s.dim() == x.dim() - 1:
            s = s.unsqueeze(-1)
        z = z + self.cond_emb(s / self.scale)
        z = z + self.dir_emb(v_hat)
        return x + self.fc2(self.act(z)) * self.scale


def _lookup_mu(blob: dict, feat_idx: int) -> Tensor:
    """eval_mu if the feature was mined as a held-out eval latent, else the fit-pool mu."""
    eval_features = blob["eval_features"]
    hit = (eval_features == feat_idx).nonzero(as_tuple=True)[0]
    if hit.numel() > 0:
        return blob["eval_mu"][hit[0]]
    features = blob["features"]
    hit = (features == feat_idx).nonzero(as_tuple=True)[0]
    if hit.numel() > 0:
        return blob["mu"][hit[0]]
    raise ValueError(
        f"feature {feat_idx} has no mu_f in concept_stats.pt; mine it "
        "(src/concept_data.py mine) or add it to configs/features.yaml first"
    )


class ConditionalWienerArm:
    """Closed-form posterior mean under a concept-conditional Gaussian; no training.

    x = h + s*v_hat, x_out = x + lam*(mu_f + A(x - mu_f) - x), A = Sigma_cond (Sigma_cond +
    s^2 I)^-1 -- the same Wiener form as denoiser.WienerDenoiser, but conditioned on the
    feature: mu_f and Sigma_cond come from src/concept_data.py's `stats` output. mu_f is the
    mean h_plus over the feature's own top-activating corpus positions; Sigma_cond is ONE
    covariance shared across features (each feature has only top_k ~ O(100) points, too few
    to fit a 768x768 covariance per feature).
    """

    def __init__(self, mu_f: Tensor, cov: Tensor, lam: float):
        self.mu_f = mu_f
        self.cov = cov  # shrunk, float64, already (Sigma_cond)
        self.lam = float(lam)
        self._a_cache: dict[float, Tensor] = {}

    @classmethod
    def load(
        cls, stats_path: Path, stats: ActStats, feat_idx: int, v_hat: Tensor, lam: float
    ) -> "ConditionalWienerArm":
        del stats  # corpus-wide ActStats is not needed here; kept for interface symmetry
        blob = torch.load(stats_path, map_location="cpu")
        device = v_hat.device
        mu_f = _lookup_mu(blob, feat_idx).to(device=device, dtype=torch.float32)
        cov = blob["cov"].to(device=device, dtype=torch.float64)
        tau = torch.diagonal(cov).mean()
        eye = torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
        cov = (1.0 - _COV_SHRINK) * cov + _COV_SHRINK * tau * eye
        return cls(mu_f=mu_f, cov=cov, lam=lam)

    def _A(self, sigma: float) -> Tensor:
        key = round(float(sigma), 8)
        a = self._a_cache.get(key)
        if a is None:
            eye = torch.eye(self.cov.shape[0], device=self.cov.device, dtype=self.cov.dtype)
            a = torch.linalg.solve(self.cov + (sigma**2) * eye, self.cov).T.float()
            self._a_cache[key] = a
        return a

    def __call__(self, h: Tensor, v_hat: Tensor, s: float) -> Tensor:
        if s == 0.0:
            return h
        x = h + s * v_hat
        a = self._A(float(s))
        mu_f = self.mu_f.to(dtype=x.dtype)
        m = mu_f + (x - mu_f) @ a.T
        return x + self.lam * (m - x)


class ConditionalDenoiserArm:
    """Learned counterpart of ConditionalWienerArm: x + eta * (D(x, v_hat, s) - x)."""

    def __init__(self, model: MLPDenoiserCondDir, eta: float):
        self.model = model
        self.eta = float(eta)

    @classmethod
    def load(
        cls, ckpt_path: Path, stats: ActStats, feat_idx: int, v_hat: Tensor, eta: float
    ) -> "ConditionalDenoiserArm":
        del feat_idx  # v_hat fully determines the conditioning; kept for interface symmetry
        device = stats.mean.device
        blob = torch.load(ckpt_path, map_location=device)
        model = MLPDenoiserCondDir(hidden=blob["hidden"], mean=stats.mean, scale=blob["scale"]).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model=model, eta=eta)

    @torch.no_grad()
    def __call__(self, h: Tensor, v_hat: Tensor, s: float) -> Tensor:
        if s == 0.0:
            return h
        x = h + s * v_hat
        out = self.model(x, v_hat, s)
        return x + self.eta * (out - x)
