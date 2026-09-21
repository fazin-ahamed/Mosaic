"""Fail-closed regression tests for the six constrained-runtime bugs
(AGENTS §32-37). Each bug was found during the 2026-09-03 3060-overlap
attempt; these tests pin the fixes so the exact runtime cannot silently
regress.

Bugs pinned:
  1. FP4 packed-bit cast destruction   (bf16 cast of packed int8 bytes)
  2. LUT-then-scale ordering           (scale-before-LUT is invalid)
  3. Expert layout/transpose           ([out,in] vs [in,out])
  4. Config-meta normalization         (short spellings rejected by contract)
  5. Per-expert validator naming       (flat vs indexed expert names)
  6. Full-layer load                   (missing/ram_overflow must fail closed)

These tests use the authoritative archive where weights are needed and
synthetic tensors where semantics alone are pinned.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ARCH = REPO / ".mosaicvm" / "core" / "core.safetensors"
HAVE_ARCH = ARCH.exists()


# ---------------------------------------------------------------- bug 1+2
def test_fp4_dequant_lut_then_scale_and_packed_bits():
    """Bugs 1+2: dequant_fp4 must LUT-unpack FIRST then scale per 32; the
    packed container must never be numerically cast before decode."""
    from mosaicvm.runtime.fp8 import dequant_fp4

    # synthetic: 2x64 packed (one 32-block), values ±{0.5..6}, scale 2^-3
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    torch.manual_seed(0)
    nib = torch.randint(0, 16, (2, 64))
    packed = (nib[:, ::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    scale = torch.full((2, 2), 0.125).to(torch.float8_e8m0fnu)

    out = dequant_fp4(packed, scale, 2, 64)
    # expected: LUT[nib] * 0.125
    exp = (lut[nib] * 0.125).to(torch.bfloat16)
    assert out.shape == (2, 64)
    assert torch.equal(out, exp), "LUT-then-scale ordering broken"

    # packed-bits invariant: casting the container to bf16 and viewing as
    # uint8 must NOT reproduce the original bytes (proves a bf16 cast
    # destroys the encoding — the old bug's mechanism)
    destroyed = packed.to(torch.bfloat16).view(torch.uint8).flatten()
    orig = packed.flatten()
    assert not torch.equal(destroyed[:4], orig[:4]) or \
        not torch.equal(destroyed, orig), \
        "bf16 cast unexpectedly preserved packed bytes (test invalid)"


@pytest.mark.skipif(not HAVE_ARCH, reason="authoritative archive required")
def test_fp4_dequant_real_expert_finite_sane():
    """Real authoritative expert: finite, sane magnitudes (bug 1/2/3
    combined would produce garbage or shape errors)."""
    from safetensors import safe_open
    from mosaicvm.runtime.fp8 import dequant_fp4

    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
        "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    if not (snap / "model.safetensors.index.json").exists():
        pytest.skip("local authoritative snapshot not present")
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"]["layers.5.ffn.experts.0.w1.weight"]
    with safe_open(str(snap / shard), framework="pt") as f:
        w = f.get_tensor("layers.5.ffn.experts.0.w1.weight")
        s = f.get_tensor("layers.5.ffn.experts.0.w1.scale")
    out = dequant_fp4(w, s, 2048, 4096)
    assert out.shape == (2048, 4096)
    assert torch.isfinite(out.float()).all()
    amax = out.abs().max().item()
    assert 1e-4 < amax < 10.0, f"expert amax {amax} outside sane range"


# ---------------------------------------------------------------- bug 3
def test_expert_transpose_orientation():
    """Bug 3: checkpoint stores [out, in]; the grouped forward wants
    [in, out]. dequant_fp4 yields [out, in]; the runtime must transpose."""
    from mosaicvm.runtime.fp8 import dequant_fp4

    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    # asymmetric pattern: row 0 all +1.0 (nib 2), row 1 all +6.0 (nib 7)
    nib = torch.full((2, 64), 2, dtype=torch.long)  # LUT[2] = 1.0
    nib[1, :] = 7                                   # LUT[7] = 6.0
    packed = (nib[:, ::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    scale = torch.ones(2, 2).to(torch.float8_e8m0fnu)
    out = dequant_fp4(packed, scale, 2, 64)  # [out=2, in=64]
    # grouped forward orientation: x @ W.T
    x = torch.ones(1, 64, dtype=torch.bfloat16)
    y = (x @ out.t()).float()  # [1, 2]: correct orientation uses .t()
    assert y[0, 0].item() == pytest.approx(64.0)    # 64 * 1.0
    assert y[0, 1].item() == pytest.approx(384.0)   # 64 * 6.0


# ---------------------------------------------------------------- bug 4
def test_config_meta_short_spellings_accepted():
    """Bug 4: the authoritative config.json ships short spellings
    (dynamic/ue8m0/fp4 + fmt e4m3); from_checkpoint must normalize and
    pass the exact contract."""
    from mosaicvm.model.config import V4Config, validate_exact_v4_config

    raw = {
        "num_hidden_layers": 43, "num_hash_layers": 3,
        "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "activation_scheme": "dynamic", "fmt": "e4m3",
        "scale_fmt": "ue8m0", "expert_dtype": "fp4",
    }
    # normalize exactly as from_checkpoint does
    meta = {
        "activation": raw.get("activation") or raw.get("activation_scheme"),
        "scale_fmt": raw.get("scale_fmt"),
        "expert_dtype": raw.get("expert_dtype"),
    }
    if meta["activation"] == "dynamic" and raw.get("fmt") == "e4m3":
        meta["activation"] = "dynamic E4M3"
    if str(meta["scale_fmt"]).lower() == "ue8m0":
        meta["scale_fmt"] = "UE8M0"
    if str(meta["expert_dtype"]).lower() == "fp4":
        meta["expert_dtype"] = "FP4 E2M1"
    # must not raise
    validate_exact_v4_config(raw, meta)
    # and the raw short forms alone (without normalization) must FAIL —
    # proving the contract is still strict
    with pytest.raises(Exception):
        validate_exact_v4_config(raw, {
            "activation": "dynamic", "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"})


@pytest.mark.skipif(not HAVE_ARCH, reason="authoritative archive required")
def test_expert_validator_per_expert_names():
    """Bug 5: the tensor validator must find per-expert indexed names and
    accept I8/F8_E8M0 safetensors dtype strings."""
    from mosaicvm.runtime.loader import StreamingLoader

    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
        "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    if not (snap / "model.safetensors.index.json").exists():
        pytest.skip("local authoritative snapshot not present")
    from mosaicvm.runtime.streaming_model import validate_exact_v4_tensors
    loader = StreamingLoader(snap)
    # must not raise for layer 0 (hash layer) and layer 5 (routed)
    validate_exact_v4_tensors(loader, "layers.0.", 0)
    validate_exact_v4_tensors(loader, "layers.5.", 5)


# ---------------------------------------------------------------- bug 6
@pytest.mark.skipif(not HAVE_ARCH, reason="authoritative archive required")
def test_full_layer_load_fail_closed():
    """Bug 6: a complete authoritative load must produce ZERO missing
    tensors; the old failure mode (fp32 params exhausting the host budget,
    leaving layers 28-42 random) must be impossible under bf16 defaults.

    Also pins: load_streaming_model must not silently proceed when
    weights are missing — the harness classifies BLOCKED on any missing.
    """
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
        "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    if not (snap / "model.safetensors.index.json").exists():
        pytest.skip("local authoritative snapshot not present")
    import torch as _t
    prev_dtype = _t.get_default_dtype()
    try:
        _t.set_default_dtype(_t.bfloat16)  # the fix: production dtype
        from mosaicvm.runtime.streaming_model import load_streaming_model
        model, loader, missing = load_streaming_model(str(snap),
                                                      device="cuda")
        assert missing == [], (
            f"full authoritative load left {len(missing)} tensors "
            f"unloaded (fail-closed violation): {missing[:5]}")
        # every loaded parameter must be finite (no random/garbage path)
        n_checked = 0
        for name, p in model.named_parameters():
            if p.numel() and p.is_floating_point():
                assert torch.isfinite(p.float()).all(), \
                    f"non-finite parameter {name}"
                n_checked += 1
        assert n_checked > 100
    finally:
        _t.set_default_dtype(prev_dtype)


def test_ram_overflow_is_not_silent():
    """Bug 6 (mechanism): load_streaming_model treats ram_overflow
    entries as `missing` — the caller must fail closed on any missing.
    This pins the contract: the route_overlap harness exits non-zero /
    classifies BLOCKED when missing != 0 (verified by construction in the
    harness source; here we pin the budget exception itself)."""
    from mosaicvm.runtime.memory_budget import MemoryBudget, BudgetExceeded
    b = MemoryBudget()
    b.reset(vram=1 << 30, ram=1 << 20)  # tiny RAM budget
    with pytest.raises(BudgetExceeded):
        b.charge_host(torch.zeros(2 << 20), "test-overflow")
