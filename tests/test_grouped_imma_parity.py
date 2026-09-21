"""Permanent regression coverage for GROUPED_IMMA_TILE_BUG (AGENTS §10).

The 2.9R suite exercised K8-mid (mean M ~1.6, mostly DP4A queue) and the
golden-greedy lineage is K1 — the grouped-IMMA path at M >= 4 was never
covered. These tests pin it against exact_moe (QPA reference) across
multiplicities, tile splits, paddings, both weight reprs, and both crossover
configs. A regression here is a CORRECTNESS_BLOCKER: no wide-K numeric result
may ship while any of these fails.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


from mosaicvm.kernels.exact_moe import exact_moe
from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
from mosaicvm.kernels.mxint import fused_moe_mxint


@pytest.fixture(autouse=True)
def _exact_target(monkeypatch):
    """Some sibling modules scrub this env var; set it per-test, not at import."""
    monkeypatch.setenv("MOSAICVM_EXACT_TARGET", "1")

H, INTER = 512, 256

def _banks(e: int, seed: int = 7):
    torch.manual_seed(seed)
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, s = pack_mxfp4_nk(
            (torch.randn(2 * INTER, H, device="cuda") / math.sqrt(H)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(s)
        pd, s2 = pack_mxfp4_nk(
            (torch.randn(H, INTER, device="cuda") / math.sqrt(INTER)).to(torch.bfloat16))
        pdu.append(pd); sdu.append(s2)
    return torch.stack(pgu), torch.stack(sgu), torch.stack(pdu), torch.stack(sdu)


def _check(x, ids, wts, PG, SG, PD, SD, repr_, crossover):
    y_ref = exact_moe(x, ids, wts, PG, SG, PD, SD)
    y = fused_moe_mxint(x, ids, wts, PG, SG, PD, SD,
                        weight_repr=repr_, crossover=crossover)
    assert torch.equal(y, y_ref), (
        f"GROUPED_IMMA parity FAIL repr={repr_} crossover={crossover} "
        f"maxdiff={(y.float() - y_ref.float()).abs().max().item()}")


@pytest.mark.parametrize("m", [4, 5, 6, 7, 8, 12, 16, 24, 32, 56])
@pytest.mark.parametrize("crossover", [0, 3])
def test_grouped_imma_single_expert_m_ge_4(m, crossover):
    """Single expert, top-1, weight=1 — the original bug ladder (M>=4 IMMA)."""
    PG, SG, PD, SD = _banks(1, seed=7)
    torch.manual_seed(11)
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(m, 1, device="cuda", dtype=torch.int64)
    wts = torch.ones(m, 1, device="cuda", dtype=torch.float32)
    _check(x, ids, wts, PG, SG, PD, SD, "hot8", crossover)


@pytest.mark.parametrize("m", [9, 17])  # multi-tile: block_m=8 splits
def test_grouped_imma_multitile(m):
    """Counts > 8 exercise multiple IMMA tiles per expert."""
    PG, SG, PD, SD = _banks(1, seed=7)
    torch.manual_seed(13)
    x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(m, 1, device="cuda", dtype=torch.int64)
    wts = torch.ones(m, 1, device="cuda", dtype=torch.float32)
    for cx in (0, 3):
        _check(x, ids, wts, PG, SG, PD, SD, "hot8", cx)


@pytest.mark.parametrize("repr_", ["hot8", "packed4"])
@pytest.mark.parametrize("crossover", [0, 3])
def test_grouped_imma_multi_expert_real_weights(repr_, crossover):
    """64 experts, distinct top-6, real normalized weights, uniform + skewed."""
    e = 64
    PG, SG, PD, SD = _banks(e, seed=7)
    torch.manual_seed(11)
    k = 56
    x = torch.randn(k, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack([torch.randperm(e, device="cuda")[:6] for _ in range(k)])
    wts = torch.rand(k, 6, device="cuda", dtype=torch.float32)
    wts = wts / wts.sum(-1, keepdim=True)
    _check(x, ids, wts, PG, SG, PD, SD, repr_, crossover)
    # skewed: high multiplicity on a few experts
    hot = torch.arange(6, device="cuda")
    for t in range(0, k, 2):
        ids[t] = torch.cat([hot, torch.randperm(e, device="cuda")[:3]])[:6]
    _check(x, ids, wts, PG, SG, PD, SD, repr_, crossover)
