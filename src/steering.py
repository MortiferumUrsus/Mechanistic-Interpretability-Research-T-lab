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
        # The seed is mixed with the direction itself. A single fixed seed drew one ambient Gaussian and
        # reused it for every latent, so the "random axes" across features were all rotations of one draw --
        # a control that shares its randomness is not twelve independent controls but one, repeated. Mixing
        # in a checksum of the direction keeps the run reproducible while making the draws independent.
        import zlib

        salt = zlib.crc32(v_hat.detach().to(torch.float32).cpu().numpy().tobytes())
        g = torch.Generator(device=v_hat.device).manual_seed((self.seed * 1_000_003 + salt) % (2**31 - 1))
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


@dataclass
class SharedDirectionArm:
    """Blend the feature direction with one direction shared across every FIT latent.

    `d_bar` is the mean, over all FIT features, of how the trained direction correction nudges v_hat
    (see anatomy.py, checkpoints/shared_direction.pt). Injecting `normalize(v_hat + kappa * d_bar)` at the naive
    norm asks how much of the correction's benefit, if any, is explained by that single shared component
    rather than by a per-feature adjustment.
    """

    d_bar: torch.Tensor
    kappa: float = 1.0

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        w = v_hat + self.kappa * self.d_bar
        w = w / w.norm().clamp_min(1e-6)
        return h + s * w


@dataclass
class ResidualDirectionArm:
    """The trained correction with its shared component removed, isolating the per-feature part.

    `delta = M v_hat` is what the trained correction would add to v; subtracting its projection onto
    `d_bar` leaves only the part of the correction specific to this feature. Comparing this arm to
    `shared` and to `dirfix` (CorrectedDirectionArm) separates how much of the correction's effect is the
    shared direction versus a genuinely per-feature one.
    """

    correction: torch.nn.Module
    d_bar: torch.Tensor
    cache: dict = field(default_factory=dict)

    def _w(self, v_hat: torch.Tensor) -> torch.Tensor:
        # Same cache discipline as CorrectedDirectionArm, and for the same reason: see the comment there.
        key = id(v_hat)
        hit = self.cache.get(key)
        if hit is not None and hit[0] is v_hat:
            return hit[1]
        with torch.no_grad():
            delta = self.correction.up(self.correction.down(v_hat))
        delta_perp = delta - (delta @ self.d_bar) * self.d_bar
        w = v_hat + delta_perp
        w = w / w.norm().clamp_min(1e-6)
        self.cache[key] = (v_hat, w)
        return w

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        if s == 0.0:
            return h
        return h + s * self._w(v_hat)


@dataclass
class AntiManifoldArm:
    """Steer along the direction the inverse activation covariance assigns to v_hat.

    `u = normalize(Sigma^-1 v_hat)` weights components inversely to how much the activation distribution
    varies along them -- the opposite emphasis from TransportedArm's `Sigma v` (which leans into
    high-variance directions to match the concept coordinate cheaply). Blending it in at weight `kappa` is
    a control for whether pointing away from the natural activation manifold helps or hurts.
    """

    cov: torch.Tensor  # already shrunk, [d_model, d_model]
    kappa: float = 1.0
    cache: dict = field(default_factory=dict)

    def _w(self, v_hat: torch.Tensor) -> torch.Tensor:
        # Same cache discipline as CorrectedDirectionArm, and for the same reason: see the comment there.
        key = id(v_hat)
        hit = self.cache.get(key)
        if hit is not None and hit[0] is v_hat:
            return hit[1]
        sol = torch.linalg.solve(self.cov.double(), v_hat.double())
        u = (sol / sol.norm().clamp_min(1e-9)).float()
        w = v_hat + self.kappa * u
        w = w / w.norm().clamp_min(1e-6)
        self.cache[key] = (v_hat, w)
        return w

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        if s == 0.0:
            return h
        return h + s * self._w(v_hat)


@dataclass
class SharedOnlyArm:
    """Control: inject the shared direction alone, with no feature direction at all.

    Isolates how much of any `shared`/`residual` gain comes from d_bar by itself versus from combining
    it with the feature's own direction.
    """

    d_bar: torch.Tensor

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        return h + s * self.d_bar


@dataclass
class DiffMeansArm:
    """Difference-of-means direction: normalize(mu_f - mu) for one feature.

    `mu_f` is the mean activation while the feature is active (data/concept_stats.pt, mined by
    concept_data.py); `mu` is the corpus-wide mean (ActStats.mean). This is the standard
    "diff-of-means" direction used in Persona Vectors / CAA, as an alternative to the SAE decoder
    column. `u` is fixed per feature and computed by the caller (generate.py), so this arm ignores
    the `v_hat` passed at call time.
    """

    u: torch.Tensor  # unit direction, precomputed per feature

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        return h + s * self.u


@dataclass
class CentredArm:
    """Decoder direction with its component along the mean activation removed.

    u = normalize(v_hat - (v_hat . mu_hat) mu_hat), mu_hat = normalize(ActStats.mean). The
    mean-centring motif from Jorgensen et al. 2023, applied to the SAE decoder column. Recomputed
    from `v_hat` on every call since one arm instance is reused across every feature in a run.
    """

    mu_hat: torch.Tensor  # unit corpus-mean direction

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        u = v_hat - (v_hat @ self.mu_hat) * self.mu_hat
        u = u / u.norm().clamp_min(1e-6)
        return h + s * u


@dataclass
class PurifiedArm:
    """Decoder direction with its component along the shared confidence direction removed.

    u = normalize(v_hat - (v_hat . d_bar) d_bar). `d_bar` is the shared component of the trained
    direction correction (anatomy.py, checkpoints/shared_direction.pt), identified as a confidence
    regulator rather than a concept-specific direction; this strips it out of the raw decoder
    column instead of blending it in (contrast with SharedDirectionArm).
    """

    d_bar: torch.Tensor

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        u = v_hat - (v_hat @ self.d_bar) * self.d_bar
        u = u / u.norm().clamp_min(1e-6)
        return h + s * u


@dataclass
class DiffMeansPurifiedArm:
    """Diff-of-means direction with its shared confidence component removed.

    u = normalize(u_dm - (u_dm . d_bar) d_bar), where u_dm is the DiffMeansArm direction for this
    feature. Same purification as PurifiedArm, applied to the diff-of-means direction instead of
    the decoder column. `u` is precomputed per feature by the caller.
    """

    u: torch.Tensor  # already-purified unit direction, precomputed per feature

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        return h + s * self.u


@dataclass
class RotateArm:
    """Norm-preserving rotation instead of additive injection.

    Rotates h within the plane spanned by h and v_hat, by exactly the angle that makes the chord
    length ||h' - h|| equal to s -- so this arm is comparable to the additive arms at matched
    perturbation norm, without ever changing ||h||. Per-row: broadcasts over any leading
    batch/seq axes, the last axis is d_model.
    """

    def __call__(self, h: torch.Tensor, v_hat: torch.Tensor, s: float) -> torch.Tensor:
        h_norm = h.norm(dim=-1, keepdim=True)
        h_hat = h / h_norm.clamp_min(1e-6)
        v_perp = v_hat - (h_hat * v_hat).sum(-1, keepdim=True) * h_hat
        v_perp = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        theta = 2.0 * torch.asin((s / (2.0 * h_norm.clamp_min(1e-6))).clamp(0.0, 1.0))
        return h_norm * (torch.cos(theta) * h_hat + torch.sin(theta) * v_perp)


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
