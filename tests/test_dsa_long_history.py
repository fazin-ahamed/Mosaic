"""§81 NEXT B — long-history top-512 selection validation (AGENTS §27, §52).

The source-exact bf16 indexer (round 1) only binds where the compressed
count exceeds the top-k: n_c = end_pos//4 > 512 → context > 2048. The
existing s=256 regressions are non-selective there. This suite validates
top-512 selection identity against the SOURCE reference at real selective
contexts:

  * fp32 mode (passthrough quants): index tensors must be BITWISE
    identical — any selection difference is a port bug.
  * bf16 production mode: both sides run the source bf16 dtype flow;
    set overlap is measured (near-tie score flips from GEMM batching
    order are possible); gate at >=99% mean set overlap with the exact
    fraction reported.

Index comparison is the precise gate (cheaper and sharper than full
attention outputs, which the big prefill gather would OOM at these sizes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tests.test_exact_dsa_source_parity import (
    _load_same_weights, _source_module)

REPO = Path(__file__).resolve().parents[1]


def _indexer_case(S: int, layer: int, bf16: bool):
    """Prefill-time indexer selection on both implementations at context S."""
    prev = torch.get_default_dtype(), torch.get_default_device()
    saved = {}
    try:
        torch.set_default_dtype(torch.bfloat16 if bf16 else torch.float32)
        torch.set_default_device("cuda")
        torch.manual_seed(31 + S + layer)

        import mosaicvm.model.dsa as D
        import mosaicvm.model.dsa_layers as DL
        saved["aq"], saved["f4"] = D.act_quant_sim, D.fp4_act_quant_sim
        saved["dlaq"], saved["dlf4"] = DL.act_quant_sim, DL.fp4_act_quant_sim
        # BOTH sides run passthrough quants in every mode: _source_module()
        # shims the source's act_quant/fp4_act_quant to passthrough, so our
        # side must match to isolate the SCORING dtype flow (fp4-quant path
        # parity is covered by the source-parity suites)
        D.act_quant_sim = lambda x, block=64: x
        D.fp4_act_quant_sim = lambda x, block=32: x
        DL.act_quant_sim, DL.fp4_act_quant_sim = (
            D.act_quant_sim, D.fp4_act_quant_sim)

        SM = _source_module()
        SM.default_dtype = torch.bfloat16 if bf16 else torch.float32
        import json
        src_cfg = json.loads(
            (REPO / ".mosaicvm" / "source" / "inference_config.json")
            .read_text())
        args = SM.ModelArgs()
        args.__dict__.update({
            "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"],
            "head_dim": src_cfg["head_dim"],
            "rope_head_dim": src_cfg["rope_head_dim"],
            "q_lora_rank": src_cfg["q_lora_rank"],
            "o_groups": src_cfg["o_groups"],
            "o_lora_rank": src_cfg["o_lora_rank"],
            "window_size": src_cfg["window_size"],
            "original_seq_len": src_cfg["original_seq_len"],
            "rope_theta": src_cfg["rope_theta"],
            "rope_factor": src_cfg["rope_factor"],
            "beta_fast": src_cfg["beta_fast"],
            "beta_slow": src_cfg["beta_slow"],
            "index_n_heads": src_cfg["index_n_heads"],
            "index_head_dim": src_cfg["index_head_dim"],
            "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6,
            "max_batch_size": 1, "max_seq_len": 8192,
            "compress_rope_theta": src_cfg["compress_rope_theta"],
            "compress_ratios": src_cfg["compress_ratios"]})
        src = SM.Attention(layer, args).cuda()
        if not bf16:
            src.wo_a.float()
            src.indexer.weights_proj.float()
            src = src.float()

        from mosaicvm.model.config import V4Config
        from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention
        ours = ExactAttention(V4Config(), layer_id=layer, max_seq=8192,
                              device="cuda")
        cast_exact_attention(ours, "cuda")
        if not bf16:
            ours = ours.float()
            ours.rebuild_freqs()
            if ours.compressor is not None:
                ours.compressor.rebuild_freqs()
                if ours.indexer is not None:
                    ours.indexer.compressor.rebuild_freqs()
            ours._wire_caches()
        _load_same_weights(src, ours, args)

        torch.manual_seed(900 + S)
        dt = torch.bfloat16 if bf16 else torch.float32
        xs = torch.randn(1, S, args.dim, dtype=dt, device="cuda") * 0.5

        with torch.no_grad():
            # source indexer (prefill): wire its caches like Attention.forward
            ix_s = src.indexer
            ix_s.compressor.kv_cache = ix_s.kv_cache
            ix_s.compressor.freqs_cis = src.freqs_cis
            ix_s.freqs_cis = src.freqs_cis
            q_lora = src.q_norm(src.wq_a(xs))
            src_idx = ix_s(xs, q_lora, 0, 0).int()        # [1, S, k]
            # ours
            xo = xs.squeeze(0)
            qr = ours.q_norm(ours.q_a(xo))
            our_idx = ours.indexer(xo, qr, ours.freqs, 0, 0).int()
        return src_idx[0], our_idx, S
    finally:
        D.act_quant_sim, D.fp4_act_quant_sim = saved["aq"], saved["f4"]
        DL.act_quant_sim, DL.fp4_act_quant_sim = saved["dlaq"], saved["dlf4"]
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S", [2304, 4608])     # n_c = 576 / 1152 > 512
def test_top512_selection_bitwise_fp32(S):
    src_idx, our_idx, _ = _indexer_case(S, layer=2, bf16=False)
    assert src_idx.shape == our_idx.shape, (src_idx.shape, our_idx.shape)
    same = (src_idx == our_idx)
    assert bool(same.all()), (
        f"fp32 top-512 selection must be bitwise identical at S={S}; "
        f"{(~same).sum().item()} of {same.numel()} indices differ")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S", [2304, 4608])
def test_top512_selection_overlap_bf16(S):
    """Production bf16 dtype flow on both sides: set overlap per row."""
    src_idx, our_idx, _ = _indexer_case(S, layer=2, bf16=True)
    assert src_idx.shape == our_idx.shape
    # exact-order identity
    exact = (src_idx == our_idx).float().mean().item()
    # per-row SET overlap (order-insensitive; -1 masks equal by position)
    k = src_idx.shape[-1]
    overlaps = []
    for r in range(0, src_idx.shape[0], max(1, src_idx.shape[0] // 64)):
        a = set(src_idx[r].tolist())
        b = set(our_idx[r].tolist())
        a.discard(-1); b.discard(-1)
        if a or b:
            overlaps.append(len(a & b) / max(1, len(a)))
    mean_ov = sum(overlaps) / len(overlaps) if overlaps else 1.0
    print(f"S={S}: exact-order identity {exact:.4f}, mean set overlap "
          f"{mean_ov:.4f}")
    assert mean_ov >= 0.99, (
        f"bf16 top-512 set overlap {mean_ov:.4f} at S={S} — near-tie flips "
        f"beyond tolerance; investigate before production promotion")
