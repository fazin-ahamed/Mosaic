"""Exact-DSA source parity — permanent regression coverage (AGENTS §38).

Compares mosaicvm ExactAttention against the checkpoint's own reference
implementation (.mosaicvm/source/inference_model.py, executed unmodified
under torch shims for its tilelang leaf kernels — local nvcc/g++ CCCL
incompatibility blocks tilelang JIT on this host).

fp32 boundary-free mode with passthrough quant sims: any nonzero diff is a
PORT BUG, so the assertion is exact equality. Covered: all three layer
regimes (ratio 0 / ratio 4+indexer / ratio 128) at a window-multiple
length AND a remainder length (s=130 → remainder 2 vs ratio 4, exercising
the compressor remainder path). Global torch state (default dtype/device)
is saved and restored around each case.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / ".mosaicvm" / "source"


def _source_module():
    """Import the kit's inference_model under leaf shims; return module."""
    for p in (str(REPO), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mosaicvm.model.dsa as mdsa

    if "fast_hadamard_transform" not in sys.modules:
        fht = types.ModuleType("fast_hadamard_transform")
        fht.hadamard_transform = lambda x, scale=None: mdsa.hadamard(x)
        sys.modules["fast_hadamard_transform"] = fht
    if "kernel" not in sys.modules:
        import inference_kernel  # noqa: F401  (kit's own leaf kernels)
        sys.modules["kernel"] = inference_kernel
    import inference_model as SM

    SM.rotate_activation = lambda x: mdsa.hadamard(x)   # fp32 study: no bf16 assert
    SM.act_quant = lambda x, *a, **k: x                  # passthrough quant
    SM.fp4_act_quant = lambda x, *a, **k: x

    def _sp(q, kv, sink, idxs, scale):
        b, s, h, d = q.size()
        valid = idxs >= 0
        idxc = idxs.clamp(min=0)
        g = kv.unsqueeze(1).expand(b, s, *kv.shape[1:]) \
            .gather(2, idxc.unsqueeze(-1).expand(-1, -1, -1, d))
        sc = torch.einsum("bshd,bstd->bsht", q, g) * scale
        sc = torch.where(valid.unsqueeze(2), sc, torch.full_like(sc, float("-inf")))
        m = sc.max(-1, keepdim=True).values
        p = torch.exp(sc - m)
        den = p.sum(-1) + torch.exp(sink.view(1, 1, h).float() - m.squeeze(-1))
        return torch.einsum("bsht,bstd->bshd", p, g) / den.unsqueeze(-1)

    SM.sparse_attn = _sp
    SM.default_dtype = torch.float32
    return SM


def _build_pair(SM, layer: int, S: int):
    import json

    src_cfg = json.loads((SRC / "inference_config.json").read_text())
    args = SM.ModelArgs()
    args.__dict__.update({
        "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"],
        "head_dim": src_cfg["head_dim"], "rope_head_dim": src_cfg["rope_head_dim"],
        "q_lora_rank": src_cfg["q_lora_rank"], "o_groups": src_cfg["o_groups"],
        "o_lora_rank": src_cfg["o_lora_rank"], "window_size": src_cfg["window_size"],
        "original_seq_len": src_cfg["original_seq_len"],
        "rope_theta": src_cfg["rope_theta"], "rope_factor": src_cfg["rope_factor"],
        "beta_fast": src_cfg["beta_fast"], "beta_slow": src_cfg["beta_slow"],
        "index_n_heads": src_cfg["index_n_heads"],
        "index_head_dim": src_cfg["index_head_dim"],
        "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6,
        "max_batch_size": 1, "max_seq_len": 1024,
        "compress_rope_theta": src_cfg["compress_rope_theta"],
        "compress_ratios": src_cfg["compress_ratios"],
    })
    src = SM.Attention(layer, args).cuda()
    src.wo_a.float()
    if getattr(src, "indexer", None) is not None:
        src.indexer.weights_proj.float()

    from mosaicvm.model.config import V4Config
    from mosaicvm.model.dsa_layers import ExactAttention
    ours = ExactAttention(V4Config(), layer_id=layer, max_seq=1024,
                          device="cuda").float()
    ours.rebuild_freqs()
    if ours.compressor is not None:
        ours.compressor.rebuild_freqs()
        if ours.indexer is not None:
            ours.indexer.compressor.rebuild_freqs()
    return src, ours, args


def _load_same_weights(src, ours, args):
    has_comp = getattr(src, "compressor", None) is not None
    with torch.no_grad():
        def gen(*sh):
            return torch.randn(*sh, device="cuda") * 0.02

        for s_nm, o_nm, inn, out in [
                ("wq_a", "q_a", args.dim, args.q_lora_rank),
                ("wq_b", "q_b", args.q_lora_rank, 64 * 512),
                ("wkv", "kv_proj", args.dim, 512),
                ("wo_b", "wo_b", 8 * 1024, args.dim),
                ("wo_a", "wo_a", 4096, 8 * 1024)]:
            w = gen(inn, out).T.contiguous()
            getattr(src, s_nm).weight.copy_(w)
            getattr(ours, o_nm).weight.copy_(w)
        for nm in ("q_norm", "kv_norm"):
            v = torch.rand(getattr(src, nm).weight.shape, device="cuda") + 0.5
            getattr(src, nm).weight.copy_(v)
            getattr(ours, nm).weight.copy_(v)
        sink = torch.randn(64, device="cuda")
        src.attn_sink.copy_(sink)
        ours.attn_sink.data.copy_(sink)

        def lc(scomp, ocomp):
            ape = torch.randn(scomp.ape.shape, device="cuda") * 0.1
            scomp.ape.copy_(ape); ocomp.ape.copy_(ape)
            for nm in ("wkv", "wgate"):
                w = torch.randn(getattr(scomp, nm).weight.shape,
                                device="cuda") * 0.02
                getattr(scomp, nm).weight.copy_(w)
                getattr(ocomp, nm).weight.copy_(w)
            v = torch.rand(scomp.norm.weight.shape, device="cuda") + 0.5
            scomp.norm.weight.copy_(v); ocomp.norm.weight.copy_(v)

        if has_comp:
            lc(src.compressor, ours.compressor)
            if getattr(src, "indexer", None) is not None and ours.indexer is not None:
                ix, oix = src.indexer, ours.indexer
                wq_ix = gen(args.q_lora_rank, 64 * 128).T.contiguous()
                ix.wq_b.weight.copy_(wq_ix); oix.wq_b.weight.copy_(wq_ix)
                wp = gen(args.dim, 64).T.contiguous()
                ix.weights_proj.weight.copy_(wp); oix.weights_proj.weight.copy_(wp)
                lc(ix.compressor, oix.compressor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("layer,ratio_label", [
    (0, "ratio0"), (2, "ratio4+indexer"), (3, "ratio128")])
@pytest.mark.parametrize("S", [256, 130])     # window multiple + remainder length
def test_exact_dsa_source_parity_fp32(layer, ratio_label, S):
    prev = torch.get_default_dtype(), torch.get_default_device()
    saved = {}
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
        torch.manual_seed(7)

        import mosaicvm.model.dsa as D
        import mosaicvm.model.dsa_layers as DL
        saved["aq"], saved["f4"] = D.act_quant_sim, D.fp4_act_quant_sim
        saved["dlaq"], saved["dlf4"] = DL.act_quant_sim, DL.fp4_act_quant_sim
        D.act_quant_sim = lambda x, block=64: x
        D.fp4_act_quant_sim = lambda x, block=32: x
        DL.act_quant_sim, DL.fp4_act_quant_sim = D.act_quant_sim, D.fp4_act_quant_sim

        SM = _source_module()
        src, ours, args = _build_pair(SM, layer, S)
        _load_same_weights(src, ours, args)
        x = torch.randn(1, S, args.dim, device="cuda") * 0.5
        with torch.no_grad():
            y_src = src(x, 0).float()
            y_ours = ours(x.squeeze(0)).float().unsqueeze(0)

        assert torch.isfinite(y_src).all(), "source side produced non-finite output"
        assert torch.isfinite(y_ours).all(), "our side produced non-finite output"
        max_abs = (y_src - y_ours).abs().max().item()
        assert max_abs == 0.0, (
            f"L{layer} ({ratio_label}) s={S}: fp32 boundary-free comparison "
            f"must be bitwise exact, max_abs={max_abs:.3e}")
    finally:
        D.act_quant_sim, D.fp4_act_quant_sim = saved["aq"], saved["f4"]
        DL.act_quant_sim, DL.fp4_act_quant_sim = saved["dlaq"], saved["dlf4"]
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])
