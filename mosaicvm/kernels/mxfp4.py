"""Source-preserving MXFP4 pack (E2M1 data + E8M0 scale), not INT4.

The Exact-30 MoE/core path keeps source FP4 values: ``pack_mxfp4`` packs a
``[K, N]`` bf16 weight into packed uint8 ``[K, N/2]`` plus an E8M0
``float8_e8m0fnu`` scale ``[K, N/32]``.  Nibble order matches the existing
``pack_w4`` / ``to_mxfp4`` convention: low nibble = even index, high nibble =
odd index.  ``dequant_mxfp4`` is the LEVEL-1 reference the skinny SIMT kernel
is checked against.

Prefer ``torch._vendor.quack.mx_utils.to_mxfp4`` when importable (FLOOR E8M0);
otherwise fall back to the in-tree copy of that exact algorithm.
"""
from __future__ import annotations

import os

import torch

# Locked E2M1FN value table (positive codes 0..7, negative 8..15).
E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
MX_BLOCK = 32  # MXFP4; NVFP4 block-16 is not this increment

_E8M0_BIAS = 127
_F4_E2M1_MAX_POW2 = 2
_F32_EXP_BIAS = 127
_MBITS_F32 = 23


def _floor_e8m0_scale(max_abs: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 scale (uint8) for FLOOR-mode MX quantization.

    Verbatim from ``torch._vendor.quack.mx_utils._compute_e8m0_scale_floor``.
    """
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = (torch.bitwise_right_shift(max_abs_int32, _MBITS_F32) & 0xFF) - _F32_EXP_BIAS
    scale_unbiased = extracted_pow2 - _F4_E2M1_MAX_POW2
    scale_unbiased = torch.clamp(scale_unbiased, min=-_E8M0_BIAS, max=_E8M0_BIAS + 1)
    scale_biased = (scale_unbiased + _E8M0_BIAS).to(torch.uint8)
    scale_biased = torch.where(torch.isnan(max_abs), 255, scale_biased)
    return scale_biased


def _f32_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> E2M1 codes (uint8, code in low 4 bits).

    Round-to-nearest-even; saturation on overflow; no NaN.  Verbatim
    structure of ``_f32_to_floatx_unpacked`` for ebits=2, mbits=1.
    """
    ebits, mbits = 2, 1
    assert x.dtype == torch.float32
    exp_bias = (1 << (ebits - 1)) - 1  # 1
    max_int = (1 << (ebits + mbits)) - 1  # 7
    sign_mask = 1 << (ebits + mbits)  # 8
    magic_adder = (1 << (_MBITS_F32 - mbits - 1)) - 1  # 2**21 - 1
    max_normal = (2 ** ((1 << ebits) - 1 - exp_bias)) * (((1 << (mbits + 1)) - 1) / (2 ** mbits))  # 6.0
    min_normal = 2 ** (1 - exp_bias)  # 1.0
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float32)
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))
    denormal_x = (x + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x = normal_x + val_to_add + mant_odd
    normal_x = (normal_x >> (_MBITS_F32 - mbits)).to(torch.uint8)
    out = torch.full_like(x, max_int, dtype=torch.uint8)
    out = torch.where(denormal_mask, denormal_x, out)
    out = torch.where(normal_mask, normal_x, out)
    sign_lp = (sign >> (_MBITS_F32 + 8 - mbits - ebits)).to(torch.uint8) & sign_mask
    return (out | sign_lp).to(torch.uint8)


def _pack_uint4(codes: torch.Tensor) -> torch.Tensor:
    """Pair 4-bit codes: (a, b) -> byte (b << 4 | a). low nibble = even index."""
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(*codes.shape[:-1], codes.shape[-1] // 2)


def pack_mxfp4(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w_bf16 [K, N] -> packed uint8 [K, N/2], scale float8_e8m0fnu [K, N/32].

    N must be a multiple of 32.  Uses the FLOOR E8M0 algorithm.  Prefer the
    upstream ``torch._vendor.quack.mx_utils.to_mxfp4`` when importable so this
    stays bit-compatible with the locked LUT.
    """
    if w_bf16.dim() != 2:
        raise ValueError(f"pack_mxfp4 expects [K, N], got {tuple(w_bf16.shape)}")
    k, n = w_bf16.shape
    if n % MX_BLOCK != 0:
        raise ValueError(f"N={n} must be a multiple of MX_BLOCK={MX_BLOCK}")
    if w_bf16.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"pack_mxfp4 expects bf16/fp16/fp32, got {w_bf16.dtype}")
    w = w_bf16.contiguous()

    try:
        from torch._vendor.quack.mx_utils import to_mxfp4  # type: ignore
    except Exception:  # pragma: no cover - fallback for envs without quack
        to_mxfp4 = None

    if to_mxfp4 is not None:
        packed, scale = to_mxfp4(w.to(torch.bfloat16), block_size=MX_BLOCK)
        return packed.to(torch.uint8).contiguous(), scale.contiguous()

    # In-tree FLOOR E8M0 copy.
    blocks = n // MX_BLOCK
    data_hp = w.to(torch.float32).view(k, blocks, MX_BLOCK)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)  # [k, blocks, 1]
    scale_biased = _floor_e8m0_scale(max_abs)  # uint8 [k, blocks, 1]
    scale_fp32 = torch.bitwise_left_shift(scale_biased.to(torch.int32), _MBITS_F32).view(torch.float32)
    scale_fp32 = torch.clamp(scale_fp32, min=2 ** (-_F32_EXP_BIAS + 1))
    data_lp = data_hp / scale_fp32
    codes = _f32_to_e2m1(data_lp.reshape(k, n))
    packed = _pack_uint4(codes)
    scale = scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return packed.contiguous(), scale.contiguous()


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """packed [K, N/2] uint8 + e8m0 [K, N/32] -> [K, N] bf16 via E2M1_LUT * scale.

    LEVEL-1 reference.  Broadcasts the per-block E8M0 scale (as fp32) over the
    32-element block and multiplies by the E2M1 codebook value.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"dequant_mxfp4 expects uint8 packed, got {packed.dtype}")
    k, n_half = packed.shape
    n = n_half * 2
    blocks = n // MX_BLOCK
    if scale.shape != (k, blocks):
        raise ValueError(f"scale shape {tuple(scale.shape)} != ({k}, {blocks})")

    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=packed.device)
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    codes = torch.stack((lo, hi), dim=-1).reshape(k, n)
    vals = lut[codes]  # [k, n] fp32

    if scale.dtype == torch.float8_e8m0fnu:
        s_biased = scale.view(torch.uint8).to(torch.int32)
    else:
        s_biased = scale.to(torch.int32)
    # E8M0 scale is a power-of-two: scale_fp32 = 2 ** (biased - 127).
    # Reconstruct by writing the exponent into the FP32 exponent field.
    s_exp = s_biased - _E8M0_BIAS  # unbiased exponent, [-127, 128]
    s_exp = s_exp.clamp(-_E8M0_BIAS, _E8M0_BIAS + 1)
    s_int = (s_exp + _F32_EXP_BIAS) << _MBITS_F32  # FP32 bits for 2**s_exp
    s_fp32 = s_int.view(torch.float32).repeat_interleave(MX_BLOCK, dim=-1)  # [k, n]
    return (vals * s_fp32).to(torch.bfloat16)


def load_expert_mxfp4(rows: int, cols: int, device: str = "cuda", source_hint: str | None = None):
    """Load one expert weight and pack to MXFP4.

    If ``MOSAICVM_MODEL_ROOT`` is set, load ``layers.0.ffn.experts.{e}`` via
    StreamingLoader. Already-packed FP4 codes are kept verbatim; FineGrainedFP8
    experts are dequant'd to bf16 then packed (a format change, not INT4 requant);
    otherwise random bf16 is packed.  Returns ``(packed, scale, source)``.
    """
    root = os.environ.get("MOSAICVM_MODEL_ROOT")
    if not root:
        w = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
        p, s = pack_mxfp4(w)
        return p, s, "synthetic_mxfp4"

    from mosaicvm.runtime.fp8 import dequant_fp8, fit_linear_weight
    from mosaicvm.runtime.loader import StreamingLoader
    from mosaicvm.runtime.streaming_model import _maybe_dequant

    loader = StreamingLoader(root)
    # Caller passes the expert index via source_hint when used by Floor B;
    # here we only need a single weight, so the caller builds the prefix.
    raise NotImplementedError("use load_expert_mxfp4_prefix for checkpoint experts")


def load_expert_mxfp4_prefix(loader, prefix: str, rows: int, cols: int, device: str = "cuda"):
    """Load ``prefix + 'w1'/'w3'/'w2'``-style weights and pack to MXFP4.

    Returns ``(packed, scale, source)`` where source is one of
    ``"mxfp4_kept"``, ``"fp8_then_mxfp4"``, ``"bf16_then_mxfp4"``.
    """
    from mosaicvm.runtime.fp8 import dequant_fp8, fit_linear_weight
    from mosaicvm.runtime.streaming_model import _maybe_dequant

    t = _maybe_dequant(loader, prefix + ".weight", "cpu")
    # Already-packed FP4? numel * 2 == rows * cols
    if t.numel() * 2 == rows * cols:
        t2 = t.reshape(t.shape[0], -1)
        if t2.shape[0] == rows and t2.shape[1] * 2 == cols:
            scale_name = prefix + ".weight.scale"
            if scale_name in loader.shards.shards:
                scale = loader.load_tensor(scale_name, device="cpu").to(device)
            else:
                scale = torch.ones(rows, cols // MX_BLOCK, dtype=torch.float8_e8m0fnu, device=device)
            return t2.to(device).to(torch.uint8), scale, "mxfp4_kept"
    # FineGrainedFP8? has a companion .scale
    scale_name = prefix + ".weight.scale"
    if scale_name in loader.shards.shards and t.dtype == torch.float8_e4m3fn:
        scale = loader.load_tensor(scale_name, device="cpu")
        w = dequant_fp8(t, scale).to(device)
        p, s = pack_mxfp4(w)
        return p, s, "fp8_then_mxfp4"
    w = fit_linear_weight(t, rows, cols).to(device)
    p, s = pack_mxfp4(w)
    return p, s, "bf16_then_mxfp4"
