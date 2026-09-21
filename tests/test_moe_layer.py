from __future__ import annotations

import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.layers import DenseMoELayer, grouped_expert_forward


def test_grouped_expert_forward_matches_loop():
    torch.manual_seed(0)
    cfg = V4Config(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=8, hidden_size=16)
    layer = DenseMoELayer(cfg)
    x = torch.randn(3, cfg.hidden_size)
    scores = torch.tensor([[0.6, 0.4], [1.0, 0.5], [0.3, 0.2]])
    idx = torch.tensor([[0, 2], [1, 3], [2, 0]])

    got = grouped_expert_forward(x, scores, idx, layer.experts_gate_up(), layer.experts_down(), cfg.n_routed_experts)

    want = torch.zeros_like(x)
    wg, wu = layer.experts_gate_up()
    wd = layer.experts_down()
    for ti in range(x.shape[0]):
        for k_ in range(cfg.num_experts_per_tok):
            e = idx[ti, k_].item()
            g = torch.nn.functional.silu(x[ti] @ wg[e])
            u = x[ti] @ wu[e]
            want[ti] += scores[ti, k_] * ((g * u) @ wd[e])

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
