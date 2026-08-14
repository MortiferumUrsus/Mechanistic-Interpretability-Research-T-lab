"""Interventions at HOOK and the TransformerLens hook that applies them.

Every intervention has the signature f(h, v_hat, s) -> h_tilde, where `s` is the intended
increment of the concept coordinate v_hat^T h. Parameterising by `s` rather than by a raw
alpha keeps all arms comparable at equal concept coordinate and makes the grid transferable
across features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from common import SKIP_POS

Intervention = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def _proj_out(delta: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
    return delta - (delta @ v_hat).unsqueeze(-1) * v_hat


def clean(h, v_hat, s):
    return h


def naive(h, v_hat, s):
    return h + s * v_hat


def norm_preserving(h, v_hat, s):
    x = h + s * v_hat
    return x * (h.norm(dim=-1, keepdim=True) / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def ln_stat_matching(h, v_hat, s):
    x = h + s * v_hat
    xm, xs = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-6)
    hm, hs = h.mean(-1, keepdim=True), h.std(-1, keepdim=True)
    return (x - xm) / xs * hs + hm


@dataclass
class DenoiserArm:
    """Arms that call a denoiser D. `mode` selects how D's output is used.

    naive      x + eta * (D(x) - x)                     -- the task's literal proposal
    cds        x + lam * P_perp(D(x) - D(h) - s v)       -- contrastive, direction preserving
    factorial  x + lam * P_perp(dsteer) + gam * P(dsteer)
    clean_then h_clean = D(h), then add steering         -- ordering control
    """

    denoiser: torch.nn.Module
    mode: str = "cds"
    eta: float = 1.0
    lam: float = 1.0
    gam: float = 0.0
    random_proj: torch.Tensor | None = None

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        from denoiser import call_denoiser

        x = h + s * v_hat
        # The denoiser is conditioned on the perturbation magnitude it is shown, so the steered
        # input is told `s` and the clean baseline is told zero. Feeding both the same value
        # would make the conditioning useless.
        if self.mode == "naive":
            return x + self.eta * (call_denoiser(self.denoiser, x, s) - x)
        if self.mode == "clean_then":
            return call_denoiser(self.denoiser, h, 0.0) + s * v_hat
        if s == 0.0:
            return h
        d_steer = (
            call_denoiser(self.denoiser, x, s)
            - call_denoiser(self.denoiser, h, 0.0)
            - s * v_hat
        )
        axis = v_hat if self.random_proj is None else self.random_proj
        perp = _proj_out(d_steer, axis)
        par = d_steer - perp
        if self.mode == "cds":
            return x + self.lam * perp
        if self.mode == "factorial":
            return x + self.lam * perp + self.gam * par
        if self.mode == "par_only":
            return x + par
        if self.mode == "perp_only":
            return x + perp
        raise ValueError(self.mode)


@dataclass
class TransportedArm:
    """Minimum-Mahalanobis steering: same concept coordinate, no training."""

    direction: torch.Tensor  # Sigma v / (v^T Sigma v), unit increment along v_hat

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        return h + s * self.direction


@dataclass
class RepairArm:
    """Learned repair conditioned on the steering direction and magnitude (CTR)."""

    repair: torch.nn.Module

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        x = h + s * v_hat
        if s == 0.0:
            return h
        return x + self.repair(x, v_hat, s)


@dataclass
class CorrectedDirectionArm:
    """Round two: inject along a learned correction of the direction, at the same norm.

    The perturbation norm equals the naive arm's, so the comparison is at matched injected energy;
    only where it points differs. The correction is trained on FIT directions and never sees the
    directions it is evaluated on.
    """

    correction: torch.nn.Module
    cache: dict = field(default_factory=dict)

    def _w(self, v_hat: torch.Tensor) -> torch.Tensor:
        # The cache holds the direction tensor alongside its correction, and the identity check below is
        # not paranoia -- it is the fix for a real defect. Keying on `id(v_hat)` alone while storing only
        # the result lets the tensor be freed the moment the caller moves to the next feature; CPython then
        # hands the same address to a later feature, the lookup hits, and that feature is steered with an
        # EARLIER feature's direction. Nothing errors and the numbers stay plausible. Keeping the tensor in
        # the cache keeps its address alive, so the collision cannot arise; the `is` check makes the failure
        # loud rather than silent if it somehow does.
        key = id(v_hat)
        hit = self.cache.get(key)
        if hit is not None and hit[0] is v_hat:
            return hit[1]
        with torch.no_grad():
            w = self.correction(v_hat.unsqueeze(0))[0]
        self.cache[key] = (v_hat, w)
        return w

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        if s == 0.0:
            return h
        return h + s * self._w(v_hat)


@dataclass
class RandomRotationArm:
    """Control for the direction correction: same rotation angle, random axis, same norm.

    If rotating the decoder direction by this much in an arbitrary direction helps as well, the gain
    belongs to perturbing the direction at all, not to the learned correction.
    """

    cos_target: float
    seed: int = 0
    cache: dict = field(default_factory=dict)

    def _w(self, v_hat: torch.Tensor) -> torch.Tensor:
        # Same cache discipline as CorrectedDirectionArm, and for the same reason: see the comment there.
        key = id(v_hat)
        hit = self.cache.get(key)
        if hit is not None and hit[0] is v_hat:
            return hit[1]
        g = torch.Generator(device=v_hat.device).manual_seed(self.seed)
        u = torch.randn(v_hat.shape, device=v_hat.device, generator=g)
        u = u - (u @ v_hat) * v_hat
        u = u / u.norm().clamp_min(1e-6)
        k = float(self.cos_target)
        w = k * v_hat + (1.0 - k * k) ** 0.5 * u
        w = w / w.norm().clamp_min(1e-6)
        self.cache[key] = (v_hat, w)
        return w

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        if s == 0.0:
            return h
        return h + s * self._w(v_hat)


@dataclass
class FeatureSurgeryArm:
    """Clamp every non-target SAE latent back under its natural corpus ceiling.

    Reconstruction error is passed through unchanged, so at s = 0 the arm is exactly the
    identity and the SAE's own reconstruction error never enters the comparison.
    """

    sae: torch.nn.Module
    ceiling: torch.Tensor  # [d_sae], per-latent max activation over the corpus
    exempt: torch.Tensor  # [d_sae] bool, target latent and its near-collinear buffer
    k: float = 1.0

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        x = h + s * v_hat
        if s == 0.0:
            return h
        z = self.sae.encode(x)
        recon = self.sae.decode(z)
        capped = torch.minimum(z, self.k * self.ceiling)
        z_new = torch.where(self.exempt, z, capped)
        return self.sae.decode(z_new) + (x - recon)

    def clamp_report(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> dict:
        """Which latents were over their ceiling, and by how much. Drives the mechanism story."""
        z = self.sae.encode(h + s * v_hat)
        over = (z > self.k * self.ceiling) & (~self.exempt)
        excess = ((z - self.k * self.ceiling) * over).clamp_min(0)
        flat = over.reshape(-1, over.shape[-1])
        return {
            "n_over_per_token": float(flat.float().sum(-1).mean()),
            "total_excess": float(excess.sum()),
            "top_latents": excess.reshape(-1, excess.shape[-1]).sum(0).topk(20).indices.tolist(),
        }


def apply_masked(fn: Intervention, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
    """Apply an arm to a [batch, seq, d] tensor, leaving the first SKIP_POS positions alone.

    Offline analyses have to reproduce what the generation hook does. The residual norm at the
    first positions of a GPT-2 sequence is an outlier, so intervening there would dominate any
    average taken over positions.
    """
    out = fn(h, v_hat, s)
    if h.dim() < 3 or h.shape[1] <= SKIP_POS:
        return out
    keep = torch.arange(h.shape[1], device=h.device).view(1, -1, 1) >= SKIP_POS
    return torch.where(keep, out, h)


@dataclass
class HookState:
    """Tracks absolute token position so that the first SKIP_POS tokens stay untouched.

    With a KV cache the hook sees the whole prompt on the first pass and a single token on
    every later pass, so the offset has to be carried explicitly.
    """

    offset: int = 0
    history: list = field(default_factory=list)

    def reset(self):
        self.offset = 0
        self.history.clear()


def make_hook(fn: Intervention, v_hat: torch.Tensor, s: float, state: HookState):
    def hook(resid, hook):
        seq = resid.shape[1]
        pos = torch.arange(state.offset, state.offset + seq, device=resid.device)
        state.offset += seq
        mask = (pos >= SKIP_POS).view(1, seq, 1)
        if not mask.any():
            return resid
        out = fn(resid, v_hat, s)
        return torch.where(mask, out, resid)

    return hook
