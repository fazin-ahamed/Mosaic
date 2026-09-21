"""Source-semantic FP4 layout for Phase 2.7 QPA/XSIMT.

Official V4 linears are not ``bf16_x @ dequant(W_fp4)``.  Weights are
``Linear.weight`` ``[N, K]`` (out, in).  MXFP4 groups 32 consecutive K per
output row:

    packed:  uint8 [N, K // 2]            # low nibble = even K, high = odd K
    scale_w: float8_e8m0fnu [N, K // 32]  # one E8M0 per output row per 32-K group

Activations: DeepSeek ``act_quant`` block-128, ``s = max(abs(x))/448``, store
``float8_e4m3fn``.  ``reference_fp4_linear_source_semantics`` is the
authoritative numeric reference for QPA/XSIMT/MoE tests.
"""
from __future__ import annotations

import os

import torch

from mosaicvm.kernels.mxfp4 import (
    E2M1_LUT,
    MX_BLOCK,
    _f32_to_e2m1,
    _floor_e8m0_scale,
    _pack_uint4,
)

ACT_BLOCK = 128
ACT_FP8_MAX = 448.0
_E8M0_BIAS = 127
_MBITS_F32 = 23
_F32_EXP_BIAS = 127


def pack_mxfp4_nk(w_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w_nk [N, K] -> packed uint8 [N, K/2], scale float8_e8m0fnu [N, K/32].

    FLOOR E8M0 on the last dim (K), block 32.  Prefer
    ``torch._vendor.quack.mx_utils.to_mxfp4`` (already blocks last dim).
    """
    if w_nk.dim() != 2:
        raise ValueError(f"pack_mxfp4_nk expects [N, K], got {tuple(w_nk.shape)}")
    n, k = w_nk.shape
    if k % MX_BLOCK != 0:
        raise ValueError(f"K={k} must be a multiple of MX_BLOCK={MX_BLOCK}")
    w = w_nk.contiguous()
    try:
        from torch._vendor.quack.mx_utils import to_mxfp4  # type: ignore
    except Exception:
        to_mxfp4 = None
    if to_mxfp4 is not None:
        packed, scale = to_mxfp4(w.to(torch.bfloat16), block_size=MX_BLOCK)
        return packed.to(torch.uint8).contiguous(), scale.contiguous()
    blocks = k // MX_BLOCK
    data_hp = w.to(torch.float32).view(n, blocks, MX_BLOCK)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    scale_biased = _floor_e8m0_scale(max_abs)
    scale_fp32 = torch.bitwise_left_shift(scale_biased.to(torch.int32), _MBITS_F32).view(torch.float32)
    scale_fp32 = torch.clamp(scale_fp32, min=2 ** (-_F32_EXP_BIAS + 1))
    data_lp = data_hp / scale_fp32
    codes = _f32_to_e2m1(data_lp.reshape(n, k))
    packed = _pack_uint4(codes)
    scale = scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return packed.contiguous(), scale.contiguous()


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu or uint8 -> fp32 power-of-two."""
    if scale.dtype == torch.float8_e8m0fnu:
        s_biased = scale.view(torch.uint8).to(torch.int32)
    else:
        s_biased = scale.to(torch.int32)
    s_exp = (s_biased - _E8M0_BIAS).clamp(-_E8M0_BIAS, _E8M0_BIAS + 1)
    s_int = (s_exp + _F32_EXP_BIAS) << _MBITS_F32
    return s_int.view(torch.float32)


def decode_e2m1_nk(packed: torch.Tensor) -> torch.Tensor:
    """packed [..., K/2] uint8 -> fp32 E2M1 values [..., K], no weight scale."""
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=packed.device)
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return lut[codes]


def decode_e4m3(x_fp8: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn or uint8-viewed e4m3 -> fp32."""
    if x_fp8.dtype == torch.uint8:
        x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    return x_fp8.to(torch.float32)


def act_quant_legacy_phase27(x: torch.Tensor, block: int = ACT_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase 2.7 act_quant: s = max(abs(x))/448 (non-pow2). Kept for comparison only."""
    if x.dim() != 2:
        raise ValueError(f"act_quant_legacy_phase27 expects [M, K], got {tuple(x.shape)}")
    m, k = x.shape
    if k % block != 0:
        raise ValueError(f"K={k} must be a multiple of act block {block}")
    xf = x.to(torch.float32)
    xb = xf.view(m, k // block, block)
    amax = xb.abs().amax(dim=-1)
    s = amax / ACT_FP8_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)
    y = (xb / s.unsqueeze(-1)).reshape(m, k).to(torch.float8_e4m3fn)
    return y, s.contiguous()


def act_quant_ref(x: torch.Tensor, block: int = ACT_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4-0731 act_quant: per 128-K block, E8M0 power-of-two scale via IEEE bits.

    s = 2^ceil(log2(amax/448)) with amax clamped to >= 1e-4; y = (x/s).e4m3fn.
    x: [M, K] any float. Returns (x_fp8 e4m3fn [M,K], scale_a fp32 [M, K//block]).
    """
    if x.dim() != 2:
        raise ValueError(f"act_quant_ref expects [M, K], got {tuple(x.shape)}")
    m, k = x.shape
    if k % block != 0:
        raise ValueError(f"K={k} must be a multiple of act block {block}")
    xf = x.to(torch.float32).view(m, k // block, block)
    amax = xf.abs().amax(-1)
    a = amax.clamp_min(1e-4)
    val = a * (1.0 / ACT_FP8_MAX)
    bits = val.view(torch.int32)
    pow_exp = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    sA = ((pow_exp + 127) << 23).view(torch.float32)
    y = (xf / sA.unsqueeze(-1)).clamp(-ACT_FP8_MAX, ACT_FP8_MAX).reshape(m, k).to(torch.float8_e4m3fn)
    return y, sA.contiguous()


def reference_fp4_linear_source_semantics(
    x_bf16: torch.Tensor,
    packed: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    """Authoritative source-linear: act_quant then per-K32 A @ B.T * sA * sW.

    x_bf16 [M, K], packed [N, K/2], scale_w [N, K/32] -> [M, N] bf16.
    """
    m, k = x_bf16.shape
    n, k_half = packed.shape
    if k_half * 2 != k:
        raise ValueError(f"packed K {k_half * 2} != x K {k}")
    n_groups = k // MX_BLOCK
    x_fp8, sA = act_quant_ref(x_bf16, ACT_BLOCK)
    acc = torch.zeros(m, n, dtype=torch.float32, device=x_bf16.device)
    sW = e8m0_to_fp32(scale_w)  # [N, K/32]
    for g in range(n_groups):
        k0 = MX_BLOCK * g
        A = decode_e4m3(x_fp8[:, k0 : k0 + MX_BLOCK])  # [M, 32]
        B = decode_e2m1_nk(packed[:, k0 // 2 : k0 // 2 + 16])  # [N, 32]
        partial = A @ B.T  # [M, N]
        partial = partial * sA[:, g // 4, None] * sW[:, g][None, :]
        acc = acc + partial
    return acc.to(torch.bfloat16)


def inspect_checkpoint_expert_layout() -> dict:
    """If MOSAICVM_MODEL_ROOT is set, inspect one expert w1.weight + .scale.

    Does not requant.  Returns layout tag:
      checkpoint_nk  — packed [N, K/2], scale [N, K/32]
      legacy_kn      — 2.6 [K, N/2] along-N
      unknown        — neither, or unset (synthetic)
    """
    root = os.environ.get("MOSAICVM_MODEL_ROOT")
    out = {
        "expert_source": "synthetic",
        "layout": "nk",
        "fp4_scheme": None,
        "packed_shape": None,
        "scale_shape": None,
    }
    if not root:
        return out
    try:
        from mosaicvm.runtime.loader import StreamingLoader

        loader = StreamingLoader(root)
        name = "layers.0.ffn.experts.0.w1.weight"
        scale_name = "layers.0.ffn.experts.0.w1.weight.scale"
        if name not in loader.shards.shards:
            # try alternate
            keys = [k for k in loader.shards.keys() if "experts.0" in k and "w1" in k]
            out["fp4_scheme"] = "missing_w1"
            out["keys_sample"] = keys[:8]
            return out
        t = loader.shards.load_tensor(name)
        scale = None
        if scale_name in loader.shards.shards:
            scale = loader.shards.load_tensor(scale_name)
        out["packed_shape"] = list(t.shape)
        out["scale_shape"] = list(scale.shape) if scale is not None else None
        out["packed_dtype"] = str(t.dtype)
        out["scale_dtype"] = str(scale.dtype) if scale is not None else None
        # Heuristic: NK if last dim * 2 is K (in_features) and scale last == K/32.
        if scale is not None and t.dtype in (torch.uint8, torch.int8):
            n0, n1 = t.shape[0], t.shape[1]
            s0, s1 = scale.shape[0], scale.shape[-1] if scale.dim() >= 2 else 1
            if n1 * 2 % 32 == 0 and s1 == (n1 * 2) // 32 and s0 == n0:
                out["expert_source"] = "checkpoint_nk"
                out["layout"] = "nk"
                out["fp4_scheme"] = "mxfp4_nk"
                return out
            if n0 * 2 % 32 == 0 and scale.dim() == 2 and scale.shape[1] == n1 * 2 // 32:
                # [K, N/2] along-N
                out["expert_source"] = "checkpoint_kn"
                out["layout"] = "legacy_kn"
                out["fp4_scheme"] = "mxfp4_kn_along_n"
                return out
        out["expert_source"] = "checkpoint_other"
        out["layout"] = "unknown"
        out["fp4_scheme"] = str(t.dtype)
        return out
    except Exception as exc:
        out["fp4_scheme"] = f"inspect_error:{type(exc).__name__}"
        out["error"] = str(exc)
        return out
