from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.moe_layout import layout_assignments


def test_layout_assignments_counts_and_segments():
    expert_ids = torch.tensor([3, 0, 3, 1], device="cuda", dtype=torch.int64)
    sorted_ids, perm, offsets, counts = layout_assignments(expert_ids, n_experts=4)
    assert counts.tolist() == [1, 1, 0, 2]
    assert int(offsets[-1].item()) == 4
    assert perm.device.type == "cuda"
    assert sorted_ids.device.type == "cuda"
    for e in range(4):
        lo = int(offsets[e].item())
        hi = int(offsets[e + 1].item())
        seg = sorted_ids[lo:hi]
        assert (seg == e).all(), f"expert {e} segment {seg.tolist()}"
