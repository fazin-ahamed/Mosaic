"""Adaptive Expert Collapse: rank-64 surrogate for omitted expert mass."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertSurrogate(nn.Module):
    """Rank-64: Linear(H, 64) - GELU - Linear(64, H).

    Forward(x, omitted_mass) -> omitted_mass * mlp(x).
    """

    def __init__(self, hidden: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)

    def forward(self, x: torch.Tensor, omitted_mass: torch.Tensor) -> torch.Tensor:
        y = self.up(F.gelu(self.down(x)))
        while omitted_mass.dim() < y.dim():
            omitted_mass = omitted_mass.unsqueeze(-1)
        return omitted_mass * y
