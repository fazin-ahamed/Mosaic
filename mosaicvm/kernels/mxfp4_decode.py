"""Register-exact E2M1 / E4M3 decode for native XSIMT/QPA.

Weight: 16 E2M1 codes x raw E8M0 byte -> f16. Race lut_global / lut_reg /
bitwise / bitwise_e8m0; default lut_reg on a tie. Discard bitwise if any of
the 4096 (code x e8m0) pairs fail vs NK dequant.

Act: E4M3 -> f16 (E4M3 is a subset of f16). LUT filled from torch so codes
match ``uint8.view(float8_e4m3fn).float()``.
"""
from __future__ import annotations

import torch

from mosaicvm.kernels.fp4_source import decode_e2m1_nk, e8m0_to_fp32
from mosaicvm.kernels.mxfp4 import E2M1_LUT
from mosaicvm.kernels.ptx_jit import compile_cuda, launch, load_kernel

# f16 bits for E2M1 codebook (sign in bit 15).
E2M1_F16_BITS = (
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
)

DEVICE_HELPERS = r"""
__device__ __forceinline__ unsigned f32_bits(float f) {
  unsigned u;
  asm volatile("mov.b32 %0, %1;" : "=r"(u) : "f"(f));
  return u;
}
__device__ __forceinline__ float bits_f32(unsigned u) {
  float f;
  asm volatile("mov.b32 %0, %1;" : "=f"(f) : "r"(u));
  return f;
}
__device__ __forceinline__ float e8m0_to_f32(unsigned char s) {
  return bits_f32(((unsigned)s) << 23);
}
__device__ __forceinline__ unsigned e2m1_f16_lut_reg(int code) {
  // Pack low-half-first so mag=0 reads bits[15:0].
  unsigned t0 = 0x38000000u;
  unsigned t1 = 0x3E003C00u;
  unsigned t2 = 0x42004000u;
  unsigned t3 = 0x46004400u;
  int mag = code & 7;
  unsigned v;
  if (mag < 2) v = (t0 >> (16 * mag)) & 0xFFFFu;
  else if (mag < 4) v = (t1 >> (16 * (mag - 2))) & 0xFFFFu;
  else if (mag < 6) v = (t2 >> (16 * (mag - 4))) & 0xFFFFu;
  else v = (t3 >> (16 * (mag - 6))) & 0xFFFFu;
  if (code & 8) v |= 0x8000u;
  return v;
}
__device__ __forceinline__ float f16bits_to_f32(unsigned h) {
  unsigned s = (h >> 15) & 1u;
  unsigned e = (h >> 10) & 0x1fu;
  unsigned m = h & 0x3ffu;
  unsigned out;
  if (e == 0) {
    if (m == 0) out = s << 31;
    else {
      int sh = 0;
      unsigned mm = m;
      while ((mm & 0x400u) == 0) { mm <<= 1; sh++; }
      mm &= 0x3ffu;
      unsigned ne = 127 - 15 - sh + 1;
      out = (s << 31) | (ne << 23) | (mm << 13);
    }
  } else if (e == 31) {
    out = (s << 31) | 0x7f800000u | (m << 13);
  } else {
    out = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
  }
  return bits_f32(out);
}
__device__ __forceinline__ unsigned f32_to_bf16_bits(float f) {
  unsigned x = f32_bits(f);
  unsigned lsb = (x >> 16) & 1u;
  unsigned bias = 0x7fffu + lsb;
  return (x + bias) >> 16;
}
__device__ __forceinline__ float e2m1_f32_lut_reg(int code) {
  return f16bits_to_f32(e2m1_f16_lut_reg(code));
}
__device__ __forceinline__ unsigned e4m3_to_f16bits(unsigned char b) {
  unsigned s = (b >> 7) & 1u;
  unsigned e = (b >> 3) & 0xfu;
  unsigned m = b & 7u;
  if (e == 0) {
    if (m == 0) return s << 15;
    float f = (s ? -1.f : 1.f) * ((float)m) * (1.f / 512.f);
    unsigned fb = f32_bits(f);
    unsigned fe = (fb >> 23) & 0xffu;
    unsigned fm = fb & 0x7fffffu;
    if (fe == 0) return s << 15;
    unsigned he = fe - 127 + 15;
    return (s << 15) | (he << 10) | (fm >> 13);
  }
  // E4M3: e=15,m=7 is NaN; e=15,m=0..6 are finite 256..448.
  if (e == 15 && m == 7) return (s << 15) | 0x7e00u;
  return (s << 15) | ((e + 8) << 10) | (m << 7);
}
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
  return f16bits_to_f32(e4m3_to_f16bits(b));
}
__device__ __forceinline__ int e2m1_qint(int code) {
  // qW = 2*W_E2M1 in [-12,12]. mag codes 0..7 -> 0,1,2,3,4,6,8,12
  int mag = code & 7;
  int q = (mag <= 4) ? mag : (mag == 5 ? 6 : (mag == 6 ? 8 : 12));
  return (code & 8) ? -q : q;
}
__device__ __forceinline__ int e4m3_qint(unsigned char b) {
  // qA = 512 * A_e4m3, exact (|qA| <= 229376 < 2^24 as fp32).
  return (int)(e4m3_to_f32(b) * 512.f);
}
__device__ __forceinline__ void split_qA_s8(int q, int* a0, int* a1, int* a2) {
  // Carry-propagating signed 3-limb s8 split. Invariant: q == a0 + 256*a1 + 65536*a2.
  int u0 = q & 0xFF;
  int s0 = (u0 >= 128) ? (u0 - 256) : u0;
  int c1 = (u0 >= 128) ? 1 : 0;
  int t1 = ((q >> 8) & 0xFF) + c1;
  int b1 = t1 & 0xFF;
  int s1 = (b1 >= 128) ? (b1 - 256) : b1;
  int c2 = (t1 >> 8) + ((b1 >= 128) ? 1 : 0);
  int t2 = ((q >> 16) & 0xFF) + c2;
  int b2 = t2 & 0xFF;
  int s2 = (b2 >= 128) ? (b2 - 256) : b2;
  *a0 = s0; *a1 = s1; *a2 = s2;
}
"""

_SRC_LUT_REG = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_e2m1_lut_reg(
    const unsigned char* packed, const unsigned char* scale,
    float* out, int n_rows, int k_half, int n_groups
) {
  int row = blockIdx.x;
  int g = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= n_rows || g >= n_groups) return;
  unsigned char s = scale[row * n_groups + g];
  float sw = e8m0_to_f32(s);
  const unsigned char* p = packed + row * k_half + g * 16;
  float* o = out + row * (n_groups * 32) + g * 32;
  if (lane < 16) {
    unsigned char b = p[lane];
    o[lane * 2]     = e2m1_f32_lut_reg(b & 0x0f) * sw;
    o[lane * 2 + 1] = e2m1_f32_lut_reg(b >> 4) * sw;
  }
}
"""

_SRC_LUT_GLOBAL = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_e2m1_lut_global(
    const unsigned char* packed, const unsigned char* scale,
    const float* lut, float* out, int n_rows, int k_half, int n_groups
) {
  int row = blockIdx.x;
  int g = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= n_rows || g >= n_groups) return;
  float sw = e8m0_to_f32(scale[row * n_groups + g]);
  const unsigned char* p = packed + row * k_half + g * 16;
  float* o = out + row * (n_groups * 32) + g * 32;
  if (lane < 16) {
    unsigned char b = p[lane];
    o[lane * 2]     = lut[b & 0x0f] * sw;
    o[lane * 2 + 1] = lut[b >> 4] * sw;
  }
}
"""

_SRC_BITWISE = DEVICE_HELPERS + r"""
__device__ __forceinline__ float e2m1_bitwise(int code) {
  int mag = code & 7;
  float v;
  if (mag == 0) v = 0.f;
  else if (mag == 1) v = 0.5f;
  else if (mag == 2) v = 1.f;
  else if (mag == 3) v = 1.5f;
  else if (mag == 4) v = 2.f;
  else if (mag == 5) v = 3.f;
  else if (mag == 6) v = 4.f;
  else v = 6.f;
  return (code & 8) ? -v : v;
}
extern "C" __global__ void decode_e2m1_bitwise(
    const unsigned char* packed, const unsigned char* scale,
    float* out, int n_rows, int k_half, int n_groups
) {
  int row = blockIdx.x;
  int g = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= n_rows || g >= n_groups) return;
  float sw = e8m0_to_f32(scale[row * n_groups + g]);
  const unsigned char* p = packed + row * k_half + g * 16;
  float* o = out + row * (n_groups * 32) + g * 32;
  if (lane < 16) {
    unsigned char b = p[lane];
    o[lane * 2]     = e2m1_bitwise(b & 0x0f) * sw;
    o[lane * 2 + 1] = e2m1_bitwise(b >> 4) * sw;
  }
}
"""

_SRC_BITWISE_E8M0 = DEVICE_HELPERS + r"""
__device__ __forceinline__ float e2m1_bitwise(int code) {
  int mag = code & 7;
  float v;
  if (mag == 0) v = 0.f;
  else if (mag == 1) v = 0.5f;
  else if (mag == 2) v = 1.f;
  else if (mag == 3) v = 1.5f;
  else if (mag == 4) v = 2.f;
  else if (mag == 5) v = 3.f;
  else if (mag == 6) v = 4.f;
  else v = 6.f;
  return (code & 8) ? -v : v;
}
__device__ __forceinline__ float e8m0_bitwise(unsigned char s) {
  int exp = (int)s - 127;
  return bits_f32((unsigned)(exp + 127) << 23);
}
extern "C" __global__ void decode_e2m1_bitwise_e8m0(
    const unsigned char* packed, const unsigned char* scale,
    float* out, int n_rows, int k_half, int n_groups
) {
  int row = blockIdx.x;
  int g = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= n_rows || g >= n_groups) return;
  float sw = e8m0_bitwise(scale[row * n_groups + g]);
  const unsigned char* p = packed + row * k_half + g * 16;
  float* o = out + row * (n_groups * 32) + g * 32;
  if (lane < 16) {
    unsigned char b = p[lane];
    o[lane * 2]     = e2m1_bitwise(b & 0x0f) * sw;
    o[lane * 2 + 1] = e2m1_bitwise(b >> 4) * sw;
  }
}
"""

_SRC_E4M3 = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_e4m3_all(
    const unsigned char* bytes, float* out, int n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = e4m3_to_f32(bytes[i]);
}
"""

_SRC_EXHAUSTIVE = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_exhaustive_lut_reg(
    const unsigned char* packed, const unsigned char* scale, float* out
) {
  int code = blockIdx.x;   // 0..15
  int e8 = blockIdx.y;     // 0..255
  if (threadIdx.x != 0) return;
  unsigned char p = (unsigned char)((code & 0x0f) | ((code & 0x0f) << 4));
  float sw = e8m0_to_f32((unsigned char)e8);
  float v = e2m1_f32_lut_reg(code) * sw;
  out[code * 256 + e8] = v;
  (void)p;
}
extern "C" __global__ void decode_exhaustive_bitwise(
    float* out
) {
  int code = blockIdx.x;
  int e8 = blockIdx.y;
  if (threadIdx.x != 0) return;
  int mag = code & 7;
  float v;
  if (mag == 0) v = 0.f;
  else if (mag == 1) v = 0.5f;
  else if (mag == 2) v = 1.f;
  else if (mag == 3) v = 1.5f;
  else if (mag == 4) v = 2.f;
  else if (mag == 5) v = 3.f;
  else if (mag == 6) v = 4.f;
  else v = 6.f;
  if (code & 8) v = -v;
  v *= e8m0_to_f32((unsigned char)e8);
  out[code * 256 + e8] = v;
}
"""

_fns: dict[str, object] = {}
CHOSEN_DECODER = "lut_reg"
BITWISE_OK = True
E4M3_OK = True


def _fn(name: str, src: str, entry: str):
    key = entry
    if key not in _fns:
        ptx = compile_cuda(src, name)
        _fns[key] = load_kernel(ptx, entry)
    return _fns[key]


def e4m3_lut_f32(device="cuda") -> torch.Tensor:
    b = torch.arange(256, dtype=torch.uint8, device=device)
    return b.view(torch.float8_e4m3fn).float()


def e2m1_lut_f32(device="cuda") -> torch.Tensor:
    return torch.tensor(E2M1_LUT, dtype=torch.float32, device=device)


def decode_weight_nk(packed: torch.Tensor, scale: torch.Tensor, decoder: str = "lut_reg") -> torch.Tensor:
    """packed [N,K/2] + e8m0 [N,K/32] -> fp32 [N,K] (E2M1 * scale)."""
    n, k_half = packed.shape
    k = k_half * 2
    n_groups = k // 32
    packed_u = packed.contiguous().view(torch.uint8)
    scale_u = scale.contiguous().view(torch.uint8)
    out = torch.empty(n, k, dtype=torch.float32, device=packed.device)
    grid = (n, n_groups)
    block = 32
    if decoder == "lut_global":
        lut = e2m1_lut_f32(packed.device)
        fn = _fn("decode_e2m1_lut_global", _SRC_LUT_GLOBAL, "decode_e2m1_lut_global")
        launch(fn, grid, block, [packed_u, scale_u, lut, out, n, k_half, n_groups])
    elif decoder == "bitwise":
        fn = _fn("decode_e2m1_bitwise", _SRC_BITWISE, "decode_e2m1_bitwise")
        launch(fn, grid, block, [packed_u, scale_u, out, n, k_half, n_groups])
    elif decoder == "bitwise_e8m0":
        fn = _fn("decode_e2m1_bitwise_e8m0", _SRC_BITWISE_E8M0, "decode_e2m1_bitwise_e8m0")
        launch(fn, grid, block, [packed_u, scale_u, out, n, k_half, n_groups])
    else:
        fn = _fn("decode_e2m1_lut_reg", _SRC_LUT_REG, "decode_e2m1_lut_reg")
        launch(fn, grid, block, [packed_u, scale_u, out, n, k_half, n_groups])
    return out


def decode_e4m3_device(bytes_u8: torch.Tensor) -> torch.Tensor:
    n = int(bytes_u8.numel())
    out = torch.empty(n, dtype=torch.float32, device=bytes_u8.device)
    fn = _fn("decode_e4m3_all", _SRC_E4M3, "decode_e4m3_all")
    block = 256
    grid = (n + block - 1) // block
    launch(fn, grid, block, [bytes_u8.contiguous().view(-1), out, n])
    return out.view(bytes_u8.shape)


def exhaustive_e2m1_table(decoder: str = "lut_reg") -> torch.Tensor:
    """[16, 256] fp32: all code x e8m0, constructed packed directly (not pack_mxfp4)."""
    device = "cuda"
    out = torch.empty(16, 256, dtype=torch.float32, device=device)
    if decoder == "bitwise":
        fn = _fn("decode_exhaustive_bitwise", _SRC_EXHAUSTIVE, "decode_exhaustive_bitwise")
        launch(fn, (16, 256), 1, [out])
        return out
    # Build packed [16*256, 16] with a single K32 of the given code, scale = e8.
    codes = torch.arange(16, device=device, dtype=torch.uint8)
    e8 = torch.arange(256, device=device, dtype=torch.uint8)
    code_g = codes.view(16, 1).expand(16, 256).reshape(-1)
    e8_g = e8.view(1, 256).expand(16, 256).reshape(-1)
    # 16 packed bytes, both nibbles = code
    packed = (code_g | (code_g << 4)).view(-1, 1).expand(-1, 16).contiguous()
    scale = e8_g.view(-1, 1).contiguous()
    dec = decode_weight_nk(packed, scale.view(torch.float8_e8m0fnu), decoder=decoder)
    # all 32 values in the row equal; take first
    return dec[:, 0].view(16, 256)


def reference_exhaustive_nk() -> torch.Tensor:
    """CPU/GPU reference: E2M1_LUT[code] * e8m0_to_fp32(e8) for all 4096 pairs."""
    device = "cuda"
    codes = torch.arange(16, device=device)
    e8 = torch.arange(256, device=device, dtype=torch.uint8)
    packed = (codes.view(16, 1).expand(16, 1).to(torch.uint8) | (codes.view(16, 1).to(torch.uint8) << 4))
    # one byte is enough; dequant path wants K=32
    packed32 = packed.expand(16, 16).contiguous()
    # We want a [16,256] table not 16 rows of K=32. Direct:
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=device)
    sw = e8m0_to_fp32(e8.view(torch.float8_e8m0fnu))
    return lut.view(16, 1) * sw.view(1, 256)


def race_decoders(n: int = 4096, k: int = 4096) -> dict:
    """Time decoder variants on an NK tensor. Default lut_reg on a tie."""
    from mosaicvm.bench.compute_floor import _bench
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk

    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    packed, scale = pack_mxfp4_nk(w)
    ref = decode_e2m1_nk(packed) * e8m0_to_fp32(scale).repeat_interleave(32, dim=-1)
    variants = ("lut_reg", "lut_global", "bitwise", "bitwise_e8m0")
    times = {}
    ok = {}
    for v in variants:
        try:
            out = decode_weight_nk(packed, scale, decoder=v)
            err = (out - ref).abs().max().item()
            ok[v] = err < 1e-5
            times[v] = _bench(lambda v=v: decode_weight_nk(packed, scale, decoder=v), iters=20, warmup=5)
        except Exception as exc:
            ok[v] = False
            times[v] = None
            times[v + "_error"] = str(exc)
    valid = [v for v in variants if ok.get(v) and times.get(v)]
    chosen = "lut_reg"
    if valid:
        best = min(valid, key=lambda v: times[v])
        # default lut_reg if tie (within 5%)
        if best != "lut_reg" and "lut_reg" in valid:
            if times["lut_reg"] <= times[best] * 1.05:
                chosen = "lut_reg"
            else:
                chosen = best
        else:
            chosen = best
    global CHOSEN_DECODER
    CHOSEN_DECODER = chosen
    return {"times": times, "ok": ok, "chosen": chosen}


def verify_exhaustive() -> dict:
    """All 16x256 E2M1 and 256 E4M3. Sets BITWISE_OK / E4M3_OK."""
    global BITWISE_OK, E4M3_OK
    ref = reference_exhaustive_nk()
    lut = exhaustive_e2m1_table("lut_reg")
    # NaN e8m0=255: compare only finite
    finite = torch.isfinite(ref)
    lut_ok = torch.allclose(lut[finite], ref[finite], rtol=0, atol=1e-6)
    bit = exhaustive_e2m1_table("bitwise")
    bit_ok = bool((~finite | torch.isclose(bit, ref, rtol=0, atol=1e-6)).all().item())
    BITWISE_OK = bit_ok
    b = torch.arange(256, dtype=torch.uint8, device="cuda")
    want = b.view(torch.float8_e4m3fn).float()
    got = decode_e4m3_device(b)
    # NaN bytes: both nan
    nan_w = torch.isnan(want)
    e4_ok = bool(torch.equal(got[~nan_w], want[~nan_w]) or torch.allclose(got[~nan_w], want[~nan_w], atol=1e-6))
    e4_ok = e4_ok and bool(torch.isnan(got[nan_w]).all().item() if nan_w.any() else True)
    E4M3_OK = e4_ok
    return {
        "e2m1_lut_reg_ok": bool(lut_ok),
        "e2m1_bitwise_ok": bit_ok,
        "e4m3_ok": e4_ok,
        "n_e2m1": 16 * 256,
        "n_e4m3": 256,
    }
