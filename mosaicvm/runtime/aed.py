"""Adaptive Expert Depth: per-token k from original router masses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AEDDecision:
    k: Tensor  # [bt] int
    masses: Tensor  # [bt, max_k] renormalized (zeros in unused slots)


def adaptive_k(original_scores: Tensor, max_k: int = 6) -> AEDDecision:
    """original_scores [bt, E] pre-topk sqrtsoftplus scores (no bias).

    CODE-mode defaults (Fidelity calls Router with active_k=6 and skips AED):
      k=2 if mass(2)/mass(6) >= 0.95 and (r2-r3) >= 0.15
      elif k=3 if mass(3)/mass(6) >= 0.95
      elif k=4 if mass(4)/mass(6) >= 0.90
      else k=max_k
    mass(k) = sum of top-k of original_scores.
    Surviving weights are renormalized to sum 1; routed_scaling_factor is
    applied at the Router, not here.
    """
    if original_scores.dim() != 2:
        raise ValueError(f"original_scores must be [bt, E], got {tuple(original_scores.shape)}")
    bt = original_scores.shape[0]
    take = min(max_k, original_scores.shape[-1])
    vals, _idx = torch.topk(original_scores, take, dim=-1)
    csum = vals.cumsum(dim=-1)
    ref_k = min(6, take)
    mass_ref = csum[:, ref_k - 1].clamp_min(1e-9)

    def mass(k: int) -> Tensor:
        return csum[:, min(k, take) - 1]

    k = torch.full((bt,), max_k, device=original_scores.device, dtype=torch.long)
    if take >= 4:
        k = torch.where(mass(4) / mass_ref >= 0.90, torch.tensor(4, device=k.device, dtype=k.dtype), k)
    if take >= 3:
        k = torch.where(mass(3) / mass_ref >= 0.95, torch.tensor(3, device=k.device, dtype=k.dtype), k)
    if take >= 3:
        gap = vals[:, 1] - vals[:, 2]
        k = torch.where(
            (mass(2) / mass_ref >= 0.95) & (gap >= 0.15),
            torch.tensor(2, device=k.device, dtype=k.dtype),
            k,
        )
    slot = torch.arange(take, device=original_scores.device)
    keep = slot.unsqueeze(0) < k.unsqueeze(-1)
    kept = vals * keep.to(vals.dtype)
    masses = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    if take < max_k:
        pad = torch.zeros(bt, max_k - take, device=masses.device, dtype=masses.dtype)
        masses = torch.cat([masses, pad], dim=-1)
    return AEDDecision(k=k, masses=masses)
