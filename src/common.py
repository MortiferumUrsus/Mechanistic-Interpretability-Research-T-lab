"""Shared setup: paths, seeding, model and SAE loading, hook conventions."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
CONFIGS = ROOT / "configs"
for _d in (DATA, RESULTS, CKPT):
    _d.mkdir(exist_ok=True)

# The shared component of the learned direction correction (anatomy.py writes it, the `shared`,
# `shared_only`, `purified` and `residual` arms read it). It lives under checkpoints/ so that it is
# tracked with the checkpoint it was derived from; data/ is git-ignored.
SHARED_DIRECTION = CKPT / "shared_direction.pt"

MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
# Second-layer replication: `TLAB_LAYER=L` moves the whole pipeline to the SAE trained on
# blocks.L.hook_resid_pre (identical to blocks.(L-1).hook_resid_post). Unset = the frozen layer
# above, so every existing result is untouched by this hook.
_LAYER_OVERRIDE = os.environ.get("TLAB_LAYER")
if _LAYER_OVERRIDE:
    SAE_ID = f"blocks.{int(_LAYER_OVERRIDE)}.hook_resid_pre"
# Intervention site required by the task: "after the middle layer" of a 12-layer model.
# blocks.6.hook_resid_post and blocks.7.hook_resid_pre are the same tensor; the SAE
# dictionary is defined on the latter, so steering directions need no basis change.
HOOK = "blocks.6.hook_resid_post"
if _LAYER_OVERRIDE:
    HOOK = f"blocks.{int(_LAYER_OVERRIDE) - 1}.hook_resid_post"
D_MODEL = 768
N_LAYERS = 12
# GPT-2 residual norms at the first positions are outliers (the BOS/attention-sink
# effect); they distort mean/covariance estimates and the alpha scale, so all
# statistics and interventions skip them.
SKIP_POS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device: str = DEVICE):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_sae(device: str = DEVICE):
    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


@dataclass
class ActStats:
    """Mean, covariance and norm scale of the activation distribution at HOOK."""

    mean: torch.Tensor
    cov: torch.Tensor
    median_norm: float
    n_tokens: int

    @classmethod
    def load(cls, path: Path = DATA / "act_stats.pt", device: str = DEVICE) -> "ActStats":
        blob = torch.load(path, map_location=device)
        return cls(
            mean=blob["mean"].to(device),
            cov=blob["cov"].to(device),
            median_norm=float(blob["median_norm"]),
            n_tokens=int(blob["n_tokens"]),
        )

    def save(self, path: Path = DATA / "act_stats.pt") -> None:
        torch.save(
            {
                "mean": self.mean.cpu(),
                "cov": self.cov.cpu(),
                "median_norm": self.median_norm,
                "n_tokens": self.n_tokens,
            },
            path,
        )

    def shrunk_cov(self, gamma: float) -> torch.Tensor:
        """Ledoit-Wolf style shrinkage towards a scaled identity."""
        tau = torch.diagonal(self.cov).mean()
        eye = torch.eye(self.cov.shape[0], device=self.cov.device, dtype=self.cov.dtype)
        return (1.0 - gamma) * self.cov + gamma * tau * eye


def load_features(split: str | None = None):
    """Feature records from configs/features.yaml, merged with any configs/features_*.yaml.

    features.yaml holds the frozen splits (test, dev, test_r3) and is hashed by the annotation
    package under evaluation/, so later rounds add their splits in separate files instead of
    editing it: features_r4.yaml holds `test_r4`. Returns the whole mapping, or one split's list.
    """
    import yaml

    payload = yaml.safe_load((CONFIGS / "features.yaml").read_text(encoding="utf-8")) or {}
    for extra in sorted(CONFIGS.glob("features_*.yaml")):
        more = yaml.safe_load(extra.read_text(encoding="utf-8")) or {}
        clash = set(more) & set(payload)
        if clash:
            raise ValueError(f"{extra.name} redefines split(s) {sorted(clash)} already present in features.yaml")
        payload.update(more)
    if split is None:
        return payload
    if split not in payload:
        raise KeyError(f"split {split!r} not found in configs/features.yaml or configs/features_*.yaml")
    return payload[split]


def natural_strength(sae, ceiling: torch.Tensor, feature: int) -> float:
    """Steering strength that reproduces the latent's strongest natural activation.

    Decoder rows of this release are unit norm, so this is essentially the latent's corpus
    ceiling. Using it as the unit makes strengths comparable across latents whose natural
    scales differ by a factor of six, which a global activation-norm unit would hide.
    """
    return float(ceiling[feature] * sae.W_dec[feature].norm())


def load_ceilings(device: str = DEVICE) -> torch.Tensor:
    return torch.load(DATA / "sae_feature_stats.pt", map_location=device)["max_act"].to(device)


def open_memmap(path: Path = DATA / "acts.f16", n: int | None = None, mode: str = "r"):
    if mode == "r":
        size = os.path.getsize(path) // (2 * D_MODEL)
        return np.memmap(path, dtype=np.float16, mode="r", shape=(size, D_MODEL))
    return np.memmap(path, dtype=np.float16, mode=mode, shape=(n, D_MODEL))
