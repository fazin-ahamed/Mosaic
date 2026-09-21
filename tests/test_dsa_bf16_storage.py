"""BF16-storage DSA residency mode — bitwise equivalence (AGENTS §54–57).

The DSA weights that are BF16-native in the checkpoint (compressor
wkv/wgate, norms, indexer weights) may be stored bf16 on GPU with fp32
math via lazy widening (`Compressor._w` / `RMSNormExact` widen) — halves
the DSA weight footprint for the 12-GB residency budget. For
bf16-representable values (which checkpoint values are, by definition)
the two storages must produce BITWISE-IDENTICAL outputs through the full
lifecycle. fp32-native tensors (ape, attn_sink) stay fp32.
"""
from __future__ import annotations

import pytest
import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention

# module-fp32-stored tensors whose checkpoint values are bf16-native
F32_STORED_BF16_NATIVE = {"wkv.weight", "wgate.weight", "norm.weight",
                          "q_norm.weight", "kv_norm.weight"}


def _quantize_checkpoint_representable(m):
    for name, p in m.named_parameters():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] + "." + parts[-1] in F32_STORED_BF16_NATIVE:
            with torch.no_grad():
                p.data = p.data.to(torch.bfloat16).to(torch.float32)


def _to_bf16_storage(m):
    for name, p in m.named_parameters():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] + "." + parts[-1] in F32_STORED_BF16_NATIVE:
            p.data = p.data.to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("layer", [0, 2, 3, 42])   # ratio 0 / 4 / 128 / L42
def test_bf16_storage_bitwise_identical(layer):
    cfg = V4Config()
    torch.manual_seed(42)
    m = ExactAttention(cfg, layer_id=layer, max_seq=2048, device="cuda")
    cast_exact_attention(m, "cuda")
    _quantize_checkpoint_representable(m)
    xs = torch.randn(300, cfg.hidden_size, dtype=torch.bfloat16,
                     device="cuda") * 0.5

    def lifecycle():
        outs = [m(xs[:130], 0)]
        for sp in range(130, 160):
            outs.append(m(xs[sp:sp + 1], sp))
        outs.append(m(xs[160:180], 160))          # chunked verify
        for sp in range(180, 190):
            outs.append(m(xs[sp:sp + 1], sp))
        return outs

    with torch.no_grad():
        outs_fp32 = lifecycle()
        m.reset_state()
        _to_bf16_storage(m)
        outs_bf16 = lifecycle()

    assert len(outs_fp32) == len(outs_bf16)
    for i, (a, b) in enumerate(zip(outs_fp32, outs_bf16)):
        d = (a.float() - b.float()).abs().max().item()
        assert d == 0.0, (
            f"L{layer} lifecycle step {i}: bf16 storage must be bitwise "
            f"identical to fp32 storage for checkpoint-representable "
            f"values, got {d:.3e}")
