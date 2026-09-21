"""Fused Triton sparse-attention kernel — permanent correctness gates
(AGENTS §44 G1–G5). Classification: NUMERIC_PARITY vs the eager
BITWISE_MOSAIC_REFERENCE (block-64 online softmax + tensor-core dot
splitting are not bit-identical to the fp32 eager single-pass; the fp32
parity suites keep the bitwise contract on the eager path).
"""
from __future__ import annotations

import pytest
import torch

from mosaicvm.kernels.dsa_sparse import sparse_attn_fused
from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa import sparse_attn_ref
from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention


def _run(fn, reps=20, warm=5):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("s,t", [(32, 160), (56, 288), (80, 352), (112, 640)])
def test_g1_g2_synthetic_and_real_shapes(s, t):
    """G1/G2: fused vs eager reference at verification shapes (window +
    compressed widths). Tolerance: bf16-ulp scale on |o|~1 outputs."""
    torch.manual_seed(s * 7 + t)
    h, d, n = 64, 512, t + 128
    q = torch.randn(s, h, d, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
    idxs = torch.randint(0, n, (s, t), device="cuda", dtype=torch.int32)
    idxs[:, -2:] = -1
    sink = torch.randn(h, dtype=torch.float32, device="cuda")
    scale = d ** -0.5
    ref = sparse_attn_ref(q, kv, sink, idxs, scale)
    out = sparse_attn_fused(q, kv, sink, idxs, scale)
    d_abs = (ref.float() - out.float()).abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert d_abs <= 0.01, f"s={s},t={t}: max abs {d_abs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_g3_long_history_selection_output():
    """G3: full module at a selective context (C/4 > 512) — fused path is
    active (bf16) and output stays within the production tolerance class
    of the eager path."""
    cfg = V4Config()
    torch.manual_seed(11)
    m = ExactAttention(cfg, layer_id=2, max_seq=8192, device="cuda")
    cast_exact_attention(m, "cuda")
    xs = torch.randn(2304, cfg.hidden_size, dtype=torch.bfloat16,
                     device="cuda") * 0.5
    import mosaicvm.model.dsa_layers as DL
    with torch.no_grad():
        DL.USE_FUSED_SPARSE = False
        ref = m(xs, 0).float()
        m.reset_state()
        DL.USE_FUSED_SPARSE = True
        out = m(xs, 0).float()
    DL.USE_FUSED_SPARSE = True
    d_abs = (ref - out).abs().max().item()
    rel = (ref - out).abs() / ref.abs().clamp_min(1e-3)
    assert torch.isfinite(out).all()
    assert d_abs <= 0.02, f"max abs {d_abs}"
    assert (rel <= 0.05).float().mean().item() >= 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_g4_g5_chunk_decode_lifecycle_fused():
    """G4/G5: chunked verification + decode against eager on the same
    module (single-processing — positions consumed once per arm)."""
    cfg = V4Config()
    torch.manual_seed(13)
    m = ExactAttention(cfg, layer_id=2, max_seq=4096, device="cuda")
    cast_exact_attention(m, "cuda")
    S, K, ND = 130, 48, 6
    xs = torch.randn(S + K + ND, cfg.hidden_size, dtype=torch.bfloat16,
                     device="cuda") * 0.5
    import mosaicvm.model.dsa_layers as DL
    outs = {}
    for mode, fused in (("ref", False), ("fused", True)):
        DL.USE_FUSED_SPARSE = fused
        m.reset_state()
        with torch.no_grad():
            seq = [m(xs[:S], 0)]
            for i in range(K):
                sp = S + i
                seq.append(m(xs[sp:sp + 1], sp))
            seq.append(m(xs[S + K:S + K + ND], S + K))
            for i in range(ND):
                sp = S + K + i
                seq.append(m(xs[sp:sp + 1], sp))
        outs[mode] = [t.float() for t in seq]
    DL.USE_FUSED_SPARSE = True
    worst = max((a - b).abs().max().item()
                for a, b in zip(outs["ref"], outs["fused"]))
    assert worst <= 0.02, f"lifecycle max abs {worst}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_kernel_speedup_recorded():
    """Record the fused-vs-eager speedup at the K112 verification shape
    (informational assert: the fused path must actually be faster)."""
    torch.manual_seed(0)
    s, h, d, t, n = 112, 64, 512, 640, 1170
    q = torch.randn(s, h, d, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
    idxs = torch.randint(0, n, (s, t), device="cuda", dtype=torch.int32)
    sink = torch.randn(h, dtype=torch.float32, device="cuda")
    scale = d ** -0.5
    t_ref = _run(lambda: sparse_attn_ref(q, kv, sink, idxs, scale))
    t_k = _run(lambda: sparse_attn_fused(q, kv, sink, idxs, scale))
    print(f"eager {t_ref:.3f} ms -> fused {t_k:.3f} ms ({t_ref/t_k:.2f}x)")
    assert t_k < t_ref, "fused path must beat the eager reference"
