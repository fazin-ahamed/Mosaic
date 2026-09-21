from __future__ import annotations

import os

os.environ["MOSAICVM_EXACT_TARGET"] = "0"


import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.layers import DenseMoELayer, Router, grouped_expert_forward
from mosaicvm.runtime.aed import adaptive_k


def test_adaptive_k_picks_2_when_mass_and_gap():
    # two dominant experts, third far below
    scores = torch.tensor([[0.50, 0.48, 0.01, 0.005, 0.003, 0.002, 0.0, 0.0]])
    d = adaptive_k(scores, max_k=6)
    assert int(d.k[0]) == 2
    assert d.masses.shape == (1, 6)
    assert torch.isclose(d.masses[0, :2].sum(), torch.tensor(1.0), atol=1e-5)
    assert float(d.masses[0, 2:].abs().sum()) == 0.0


def test_adaptive_k_falls_back_to_max_k():
    scores = torch.ones(1, 8) / 8
    d = adaptive_k(scores, max_k=6)
    assert int(d.k[0]) == 6


def test_router_active_k_without_rebuild():
    cfg = V4Config(n_routed_experts=8, num_experts_per_tok=6, hidden_size=16)
    r = Router(cfg)
    r.weight.data.normal_(0, 0.1)
    x = torch.randn(3, 16)
    s6, i6 = r(x)
    assert s6.shape[-1] == 6
    r.active_k = 4
    s4, i4 = r(x)
    assert s4.shape[-1] == 4
    assert i4.shape[-1] == 4


def test_router_aed_zero_pads_to_max_k():
    cfg = V4Config(n_routed_experts=8, num_experts_per_tok=6, hidden_size=16)
    r = Router(cfg)
    r.aed = True
    r.weight.data.zero_()
    r.weight.data[0] = 10.0
    r.weight.data[1] = 9.5
    x = torch.ones(2, 16)
    scores, idx = r(x)
    assert scores.shape[-1] == 6
    assert idx.shape[-1] == 6


def test_grouped_expert_forward_skips_zero_scores():
    torch.manual_seed(0)
    cfg = V4Config(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=8, hidden_size=16)
    layer = DenseMoELayer(cfg)
    x = torch.randn(2, cfg.hidden_size)
    scores = torch.tensor([[0.6, 0.0], [1.0, 0.0]])
    idx = torch.tensor([[0, 2], [1, 3]])
    got = grouped_expert_forward(x, scores, idx, layer.experts_gate_up(), layer.experts_down(), cfg.n_routed_experts)
    want = grouped_expert_forward(
        x, scores[:, :1], idx[:, :1], layer.experts_gate_up(), layer.experts_down(), cfg.n_routed_experts
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
