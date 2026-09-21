"""emit_tiles tile_rows correctness (Phase 2.8 Step 5)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.w4a16_grouped import emit_tiles


@pytest.mark.parametrize("block_m", [1, 8, 16])
def test_tile_rows_exact(block_m):
    counts = torch.tensor([0, 1, block_m - 1, block_m, block_m + 1, 3 * block_m + 2], device="cuda", dtype=torch.int32)
    ei, ms, nt, rows = emit_tiles(counts, block_m=block_m, max_tiles=64)
    nt = int(nt.item())
    # rebuild expected
    exp = []
    for e, c in enumerate(counts.tolist()):
        n = (c + block_m - 1) // block_m
        for t in range(n):
            exp.append((e, t * block_m, min(block_m, c - t * block_m)))
    assert nt == len(exp)
    got = list(zip(ei[:nt].tolist(), ms[:nt].tolist(), rows[:nt].tolist()))
    assert got == exp


@pytest.mark.parametrize("block_m", [1, 8, 16])
def test_tile_rows_all_empty(block_m):
    counts = torch.zeros(5, dtype=torch.int32, device="cuda")
    ei, ms, nt, rows = emit_tiles(counts, block_m=block_m, max_tiles=8)
    assert int(nt.item()) == 0
