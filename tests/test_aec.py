from __future__ import annotations

import os

os.environ["MOSAICVM_EXACT_TARGET"] = "0"


import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.layers import DenseMoELayer
from mosaicvm.runtime.aec import ExpertSurrogate


def test_surrogate_zero_mass_is_noop():
    torch.manual_seed(0)
    h = 16
    x = torch.randn(4, h)
    s = ExpertSurrogate(h, rank=8)
    y = s(x, omitted_mass=torch.zeros(4, 1))
    assert torch.equal(y, torch.zeros_like(x)) or torch.allclose(y, torch.zeros_like(x), atol=1e-7)


def test_moe_aec_zero_omitted_equals_exact():
    torch.manual_seed(0)
    cfg = V4Config(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=8, hidden_size=16)
    layer = DenseMoELayer(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    with torch.no_grad():
        exact = layer(x)
        layer.surrogate = ExpertSurrogate(cfg.hidden_size, rank=8)
        layer.use_aec = True
        # omitted_mass forced to 0 inside forward when all mass is kept
        got = layer(x)
    torch.testing.assert_close(got, exact, rtol=1e-4, atol=1e-4)
