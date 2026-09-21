"""Exact-DSA CHUNKED-PREFILL parity (AGENTS §33 gate extended).

The source has no chunked path (start_pos>0, s>1) — it is our extension
needed for (a) long-context prefill (the O(S·(win+S/ratio)) gather
transient, artifacts/dsa_state_census.md finding 4) and (b) P2
verification blocks (N tokens at a position). Reference gate per §33:
recompute-from-full-prefix — run the SOURCE reference prefill on the
whole sequence and require the chunk's rows to match exactly (fp32
boundary-free mode; any nonzero delta is either a port bug or, if at
GEMM-reduction-order scale, explicitly classified NUMERIC_PARITY).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tests.test_exact_dsa_source_parity import (
    _build_pair, _load_same_weights, _source_module)

REPO = Path(__file__).resolve().parents[1]


def _chunked_case(S: int, K: int, layer: int, label: str):
    prev = torch.get_default_dtype(), torch.get_default_device()
    saved = {}
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
        torch.manual_seed(23 + layer)

        import mosaicvm.model.dsa as D
        import mosaicvm.model.dsa_layers as DL
        saved["aq"], saved["f4"] = D.act_quant_sim, D.fp4_act_quant_sim
        saved["dlaq"], saved["dlf4"] = DL.act_quant_sim, DL.fp4_act_quant_sim
        D.act_quant_sim = lambda x, block=64: x
        D.fp4_act_quant_sim = lambda x, block=32: x
        DL.act_quant_sim, DL.fp4_act_quant_sim = D.act_quant_sim, D.fp4_act_quant_sim

        SM = _source_module()
        src, ours, args = _build_pair(SM, layer, S + K)
        _load_same_weights(src, ours, args)

        torch.manual_seed(300 + S + K + layer)
        xs = torch.randn(1, S + K, args.dim, device="cuda") * 0.5

        with torch.no_grad():
            # reference: full-prefix source prefill
            y_ref = src(xs, 0).float()[0, S:]            # chunk rows
            # ours: prefill S, then chunked K at start_pos=S
            _ = ours(xs[0, :S], 0)
            y_chk = ours(xs[0, S:], S).float()

        assert torch.isfinite(y_ref).all() and torch.isfinite(y_chk).all()
        max_abs = (y_ref - y_chk).abs().max().item()
        return max_abs
    finally:
        D.act_quant_sim, D.fp4_act_quant_sim = saved["aq"], saved["f4"]
        DL.act_quant_sim, DL.fp4_act_quant_sim = saved["dlaq"], saved["dlf4"]
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S,K,label", [
    (130, 32, "remainder-lead"),      # leading 2 rows complete a group
    (128, 32, "aligned"),             # chunk starts exactly on a group
    (129, 33, "lead3+odd"),
    (256, 112, "post-window"),        # history >= window; full ring path
    (130, 130, "chunk-eq-window"),    # chunk as large as the window
    (200, 160, "chunk-gt-window"),    # chunk > window (ring scatter path)
])
def test_chunked_parity_l2(S, K, label):
    d = _chunked_case(S, K, 2, f"ratio4+idx/{label}")
    assert d <= 1e-4, f"L2 chunked {label} (S={S},K={K}): max_abs={d:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S,K,label", [
    (130, 32, "remainder-lead"),
    (256, 112, "post-window"),
])
def test_chunked_parity_l3(S, K, label):
    # ratio 128: chunk emits at most one compressed row; leading loop up
    # to 127 rows
    d = _chunked_case(S, K, 3, f"ratio128/{label}")
    assert d <= 1e-4, f"L3 chunked {label} (S={S},K={K}): max_abs={d:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S,K,label", [
    (130, 32, "remainder-lead"),
    (256, 112, "post-window"),
])
def test_chunked_parity_l0(S, K, label):
    d = _chunked_case(S, K, 0, f"ratio0/{label}")
    assert d <= 1e-4, f"L0 chunked {label} (S={S},K={K}): max_abs={d:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_chunked_then_decode():
    """Chunked prefill must leave state valid for continued decode."""
    prev = torch.get_default_dtype(), torch.get_default_device()
    saved = {}
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
        torch.manual_seed(77)
        import mosaicvm.model.dsa as D
        import mosaicvm.model.dsa_layers as DL
        saved["aq"], saved["f4"] = D.act_quant_sim, D.fp4_act_quant_sim
        saved["dlaq"], saved["dlf4"] = DL.act_quant_sim, DL.fp4_act_quant_sim
        D.act_quant_sim = lambda x, block=64: x
        D.fp4_act_quant_sim = lambda x, block=32: x
        DL.act_quant_sim, DL.fp4_act_quant_sim = D.act_quant_sim, D.fp4_act_quant_sim
        SM = _source_module()
        S, K, NDEC = 130, 34, 6
        src, ours, args = _build_pair(SM, 2, S + K + NDEC)
        _load_same_weights(src, ours, args)
        torch.manual_seed(501)
        xs = torch.randn(1, S + K + NDEC, args.dim, device="cuda") * 0.5
        with torch.no_grad():
            _ = src(xs[:, :S + K], 0)                 # reference prefill
            _ = ours(xs[0, :S], 0)
            _ = ours(xs[0, S:S + K], S)
            for i in range(NDEC):
                sp = S + K + i
                y_s = src(xs[:, sp:sp + 1], sp).float()
                y_o = ours(xs[0, sp:sp + 1], sp).float().unsqueeze(0)
                d = (y_s - y_o).abs().max().item()
                assert d <= 1e-4, f"post-chunk decode step {i}: {d:.3e}"
    finally:
        D.act_quant_sim, D.fp4_act_quant_sim = saved["aq"], saved["f4"]
        DL.act_quant_sim, DL.fp4_act_quant_sim = saved["dlaq"], saved["dlf4"]
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])
