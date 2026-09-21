"""Exact-DSA DECODE parity — permanent regression coverage (AGENTS §32–34).

Gate (§33): prefill S tokens, then decode one token at a time, comparing
against the checkpoint's own reference implementation executed stepwise on
identical clean (seeded random) tensors. fp32 boundary-free mode with
passthrough quant sims → the assertion is exact equality at every decode
step.

Matrix (§34): ratio 0 / 4 / 128; window boundary −1/+1 (S=127/129 vs 128);
remainder lengths (S=130); multiple incremental steps; long-history wrap
(decode well past the 128 window, ring ordering); L0, middle layer (L20),
L42 (the corrected ratio-4 DSpark-adjacent layer). Compression firing at
ratio boundaries is covered by decoding ≥ ratio steps after a remainder
prefill (L2/L42) and ≥ 128 steps for ratio 128 (L3, 127→254).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tests.test_exact_dsa_source_parity import (
    _build_pair, _load_same_weights, _source_module)

REPO = Path(__file__).resolve().parents[1]


def _run_decode_case(S: int, n_steps: int, layer: int, ratio_label: str):
    """Prefill S tokens then decode n_steps one-token steps on both sides."""
    prev = torch.get_default_dtype(), torch.get_default_device()
    saved = {}
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
        torch.manual_seed(11 + layer)

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

        torch.manual_seed(101 + S + layer)
        xs = torch.randn(1, S + n_steps, args.dim, device="cuda") * 0.5

        # --- prefill both sides ---
        with torch.no_grad():
            y_s = src(xs[:, :S], 0).float()
            y_o = ours(xs[0, :S]).float().unsqueeze(0)
        assert (y_s - y_o).abs().max().item() == 0.0, (
            f"L{layer} ({ratio_label}) prefill S={S} diverged")

        # --- stepwise decode ---
        for step in range(n_steps):
            sp = S + step
            with torch.no_grad():
                y_s = src(xs[:, sp:sp + 1], sp).float()
                y_o = ours(xs[0, sp:sp + 1], sp).float().unsqueeze(0)
            d = (y_s - y_o).abs().max().item()
            assert d == 0.0, (
                f"L{layer} ({ratio_label}) decode step {step} "
                f"(start_pos={sp}, S={S}): max_abs={d:.3e}")
    finally:
        D.act_quant_sim, D.fp4_act_quant_sim = saved["aq"], saved["f4"]
        DL.act_quant_sim, DL.fp4_act_quant_sim = saved["dlaq"], saved["dlf4"]
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])


# --- window boundary cases (ratio 4 + indexer, the fullest stack) ---------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S,label", [(127, "win-1"), (128, "win"),
                                     (129, "win+1"), (130, "remainder")])
def test_decode_l2_window_boundaries(S, label):
    _run_decode_case(S, n_steps=10, layer=2, ratio_label=f"ratio4+idx/{label}")


# --- multiple steps + long-history wrap past the 128 window --------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l2_long_wrap():
    # S=130 then 40 steps: start_pos 130..169 — ring wraps twice, several
    # ratio-4 compressions fire mid-stream
    _run_decode_case(130, n_steps=40, layer=2, ratio_label="ratio4+idx/wrap")


# --- ratio 0 (pure sliding window) ----------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("S,label", [(127, "win-1"), (128, "win"),
                                     (129, "win+1")])
def test_decode_l0_ratio0(S, label):
    _run_decode_case(S, n_steps=8, layer=0, ratio_label=f"ratio0/{label}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l0_long_wrap():
    _run_decode_case(96, n_steps=48, layer=0, ratio_label="ratio0/wrap")


# --- ratio 128 (static compress idxs; compression fires at pos 127,255..) --

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l3_compression_fires():
    # S=127, decode 129 steps: start_pos 127..255 — first ratio-128
    # compression fires at start_pos=127 and again at 255
    _run_decode_case(127, n_steps=129, layer=3, ratio_label="ratio128/fire")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l3_partial_chunk():
    # S=130: one compressed row exists from prefill; decode a few steps —
    # no new compression fires (start_pos+1 not divisible by 128)
    _run_decode_case(130, n_steps=6, layer=3, ratio_label="ratio128/partial")


# --- representative middle layer + L42 (corrected ratio-4) -----------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l20_middle():
    _run_decode_case(130, n_steps=12, layer=20, ratio_label="L20/ratio4+idx")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_l42():
    _run_decode_case(130, n_steps=12, layer=42, ratio_label="L42/ratio4+idx")
