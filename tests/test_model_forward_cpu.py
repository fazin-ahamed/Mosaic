from __future__ import annotations

import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.model import V4Model


def test_model_forward_cpu_shapes():
    import os

    prev = os.environ.get("MOSAICVM_EXACT_TARGET")
    os.environ["MOSAICVM_EXACT_TARGET"] = "0"  # shape-only; hash layers use salted fallback
    try:
        cfg = V4Config(
            num_hidden_layers=2,
            hidden_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            num_hash_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            q_lora_rank=16,
            o_lora_rank=16,
            vocab_size=64,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            sliding_window=8,
            index_topk=4,
            index_n_heads=4,
            index_head_dim=8,
        )
        m = V4Model(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = m(ids)
        assert logits.shape == (2, 5, cfg.vocab_size)
        out = m.generate(ids[:1], max_new_tokens=2)
        assert out.shape == (1, 7)
    finally:
        if prev is None:
            os.environ.pop("MOSAICVM_EXACT_TARGET", None)
        else:
            os.environ["MOSAICVM_EXACT_TARGET"] = prev
