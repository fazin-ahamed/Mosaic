"""GPU-side MoE assignment layout. Expert ids never leave the device."""
from __future__ import annotations

import torch
from torch import Tensor


def layout_assignments(
    expert_ids: Tensor, n_experts: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """expert_ids [A] int64, A = k*top_k.

    Returns:
      sorted_ids [A]     expert ids sorted
      perm [A]           argsort permutation (token-slot -> packed row)
      offsets [E+1]      prefix sums, int32, zeros for unused experts
      counts [E]         tokens per expert, int32
    """
    if expert_ids.device.type != "cuda":
        raise ValueError("layout_assignments requires CUDA expert_ids")
    ids = expert_ids.reshape(-1).to(torch.int64)
    perm = torch.argsort(ids)
    sorted_ids = ids[perm]
    counts = torch.bincount(ids, minlength=n_experts).to(torch.int32)
    zeros = torch.zeros(1, dtype=torch.int32, device=ids.device)
    offsets = torch.cat([zeros, torch.cumsum(counts, dim=0).to(torch.int32)])
    return sorted_ids, perm, offsets, counts
