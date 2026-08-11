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
for _d in (DATA, RESULTS, CKPT):
    _d.mkdir(exist_ok=True)

MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
# Intervention site required by the task: "after the middle layer" of a 12-layer model.
# blocks.6.hook_resid_post and blocks.7.hook_resid_pre are the same tensor; the SAE
# dictionary is defined on the latter, so steering directions need no basis change.
HOOK = "blocks.6.hook_resid_post"
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


def open_memmap(path: Path = DATA / "acts.f16", n: int | None = None, mode: str = "r"):
    if mode == "r":
        size = os.path.getsize(path) // (2 * D_MODEL)
        return np.memmap(path, dtype=np.float16, mode="r", shape=(size, D_MODEL))
    return np.memmap(path, dtype=np.float16, mode=mode, shape=(n, D_MODEL))
