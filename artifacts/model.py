"""Minimal standalone loader for direction_correction.pt.

This file is uploaded with the checkpoint so the advertised example does not
depend on importing the full experiment repository.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class DirectionCorrection(nn.Module):
    """Shared low-rank map w(v) = normalize(v + up(down(v)))."""

    def __init__(self, d_model: int = 768, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        corrected = direction + self.up(self.down(direction))
        return corrected / corrected.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def load_direction_correction(
    path: str | Path = "direction_correction.pt",
    *,
    device: str | torch.device = "cpu",
) -> DirectionCorrection:
    """Load the published checkpoint and return an eval-mode module."""

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location=device)

    rank = int(payload["rank"])
    model = DirectionCorrection(d_model=768, rank=rank).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model

