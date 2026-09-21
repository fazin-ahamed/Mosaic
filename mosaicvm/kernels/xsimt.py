"""Native XSIMT MXFP4 GEMM (M=1-3): one lane per output neuron, K32 tiles.

Prepack tiles 32N x 32K so lane t loads bytes 16*t .. 16*t+15 of a 512-byte
slab (tight warp load). Inverse is bit-exact. No dequant, no tl.dot, no
shared weight tile, no [K,N] materialization.

Phase 2.8: GEMM kernels consume ``const half* x_norm`` (predecoded E4M3).
Ordered K-split variants read the 544-B / 272-B QPA fragment bank.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mosaicvm.kernels.fp4_source import ACT_BLOCK, MX_BLOCK, act_quant_ref
from mosaicvm.kernels.mxfp4_decode import DEVICE_HELPERS, e4m3_lut_f32
from mosaicvm.kernels.ptx_jit import compile_cuda, launch, load_kernel

_LAUNCHES = 0
_PTX_CACHE: dict[str, bytes] = {}
_FN_CACHE: dict[str, object] = {}
_BANK_CACHE: dict[int, tuple] = {}

XSIMT_TILE_N = 32
XSIMT_TILE_K = 32
XSIMT_CODE_BYTES = 512  # 32 rows * 16 packed bytes
XSIMT_SCALE_BYTES = 32


def last_launch_count() -> int:
    return _LAUNCHES


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepack_xsimt(packed: torch.Tensor, scale: torch.Tensor):
    """packed [N,K/2], scale [N,K/32] -> (codes [n_tiles,k_tiles,512], scales [n_tiles,k_tiles,32])."""
    n, k_half = packed.shape
    k = k_half * 2
    n_tiles = _cdiv(n, XSIMT_TILE_N)
    k_tiles = k // XSIMT_TILE_K
    pad_n = n_tiles * XSIMT_TILE_N - n
    if pad_n:
        packed = torch.nn.functional.pad(packed, (0, 0, 0, pad_n))
        scale = torch.nn.functional.pad(scale.view(torch.uint8), (0, 0, 0, pad_n)).view(scale.dtype)
    p = packed.view(n_tiles, XSIMT_TILE_N, k_tiles, 16).permute(0, 2, 1, 3).contiguous()
    codes = p.view(n_tiles, k_tiles, XSIMT_CODE_BYTES)
    s = scale.view(torch.uint8).view(n_tiles, XSIMT_TILE_N, k_tiles).permute(0, 2, 1).contiguous()
    scales = s.view(n_tiles, k_tiles, XSIMT_SCALE_BYTES)
    return codes, scales


def inverse_prepack_xsimt(codes: torch.Tensor, scales: torch.Tensor, n: int, k: int):
    """Bit-exact inverse of prepack_xsimt. Returns packed [N,K/2], scale uint8 [N,K/32]."""
    n_tiles, k_tiles, _ = codes.shape
    p = codes.view(n_tiles, k_tiles, XSIMT_TILE_N, 16).permute(0, 2, 1, 3).contiguous()
    packed = p.view(n_tiles * XSIMT_TILE_N, k_tiles * 16)[:n, : k // 2]
    s = scales.view(n_tiles, k_tiles, XSIMT_TILE_N).permute(0, 2, 1).contiguous()
    scale = s.view(n_tiles * XSIMT_TILE_N, k_tiles)[:n, : k // 32]
    return packed.contiguous(), scale.contiguous()


def _invqpa_c() -> str:
    """Byte offset of packed[row r][j] inside a 272-B QPA fragment (codes 0..255)."""
    vals = []
    for r in range(16):
        for j in range(16):
            g = r & 7
            t = j & 3
            lane = g * 4 + t
            jhi = (j >> 2) & 1
            blk = j >> 3
            a = ((r >> 3) & 1) + 2 * jhi
            pos = blk * 128 + lane * 4 + a
            vals.append(str(pos))
    return ",".join(vals)


_INVQPA = _invqpa_c()


_DECODE_SRC = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_activation_exact(
    const unsigned char* x_fp8, unsigned short* out, int n
) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = (unsigned short)e4m3_to_f16bits(x_fp8[i]);
}
"""


def decode_activation_exact(pair) -> torch.Tensor:
    """(x_fp8 e4m3fn [M,K], sA unused) -> x_half [M,K] f16 via exact e4m3_to_f16bits."""
    if isinstance(pair, tuple):
        x_fp8 = pair[0]
    else:
        x_fp8 = pair
    if x_fp8.dtype == torch.float8_e4m3fn:
        x_u8 = x_fp8.view(torch.uint8)
    else:
        x_u8 = x_fp8.contiguous().view(torch.uint8)
    out = torch.empty(x_u8.shape, dtype=torch.float16, device=x_u8.device)
    n = x_u8.numel()
    tpb = 256
    grid = (_cdiv(n, tpb),)
    fn = _get_fn("decode_activation_exact", _DECODE_SRC, "decode_activation_exact")
    launch(fn, grid, tpb, [x_u8, out.view(torch.uint16), n])
    return out


_XSIMT_SRC = DEVICE_HELPERS + r"""
extern "C" __global__ void xsimt_gemm_pre(
    const unsigned short* x_norm,
    const float* sA,
    const unsigned char* packed_pre,
    const unsigned char* scale_pre,
    unsigned short* out,
    int M, int N, int K, int n_tiles, int n_groups
) {
  int m = blockIdx.x;
  int nt = blockIdx.y;
  int lane = threadIdx.x;
  int n = nt * 32 + lane;
  if (m >= M || nt >= n_tiles) return;
  int valid = n < N;
  float acc = 0.f;
  const unsigned short* xrow = x_norm + (size_t)m * (size_t)K;
  const float* sa_row = sA + (size_t)m * (size_t)(K / 128);
  for (int g = 0; g < n_groups; g++) {
    const unsigned char* slab = packed_pre
      + (((size_t)nt * (size_t)n_groups + (size_t)g) * 512ull)
      + (size_t)lane * 16ull;
    unsigned char swb = scale_pre[((size_t)nt * (size_t)n_groups + (size_t)g) * 32ull + (size_t)lane];
    float sw = e8m0_to_f32(swb);
    float sa = sa_row[g >> 2];
    const unsigned short* a = xrow + g * 32;
    float dot = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      unsigned char b = slab[i];
      float w0 = e2m1_f32_lut_reg(b & 0x0f);
      float w1 = e2m1_f32_lut_reg(b >> 4);
      float a0 = f16bits_to_f32(a[i * 2]);
      float a1 = f16bits_to_f32(a[i * 2 + 1]);
      dot += a0 * w0 + a1 * w1;
    }
    acc += dot * sw * sa;
  }
  if (valid) {
    out[(size_t)m * (size_t)N + (size_t)n] = (unsigned short)f32_to_bf16_bits(acc);
  }
}
"""


def _ksplit_xsimt_src(kind: str, warps: int, ordered: bool) -> str:
    """kind in {'gu','down'}. ordered=False is the race-only contiguous-slice assignment."""
    name = f"xsimt_{kind}_ksplit_{'ordered' if ordered else 'reordered'}_w{warps}"
    assign = "g = r * WARPS_K + w" if ordered else "g = w * n_rounds + r"
    if kind == "gu":
        frag_off = """
    int j = blockIdx.y;
    int half = lane >> 4;
    int rrow = lane & 15;
    const unsigned char* fbase = frag
      + (size_t)e * (size_t)frag_stride_e
      + (((size_t)j * (size_t)k_tiles + (size_t)g) * 544ull)
      + (size_t)half * 272ull;
    int col = (half == 0) ? (j * 16 + rrow) : (inter + j * 16 + rrow);
"""
        n_tiles_y = "n_mac"
    else:
        frag_off = """
    int td = blockIdx.y;
    int half = lane >> 4;
    int rrow = lane & 15;
    int nt = td * 2 + half;
    const unsigned char* fbase = frag
      + (size_t)e * (size_t)frag_stride_e
      + (((size_t)nt * (size_t)k_tiles + (size_t)g) * 272ull);
    int col = td * 32 + lane;
"""
        n_tiles_y = "n_mega"
    return DEVICE_HELPERS + f"""
__device__ __constant__ unsigned short INVQPA[256] = {{{_INVQPA}}};
extern "C" __global__ void {name}(
    const unsigned short* x_norm,
    const float* sA,
    const int* row_token,
    const unsigned char* frag,
    const int* offsets,
    const int* expert_index,
    const int* m_start,
    const int* tile_rows,
    const int* n_tiles_ptr,
    unsigned short* out,
    int A, int N, int inter, int K, int k_tiles, int n_mac, int n_mega, long long frag_stride_e
) {{
  enum {{ WARPS_K = {warps} }};
  int pid_t = blockIdx.x;
  int n_tiles = n_tiles_ptr[0];
  if (pid_t >= n_tiles) return;
  int lane = threadIdx.x & 31;
  int w = threadIdx.x >> 5;
  int e = expert_index[pid_t];
  int m0 = m_start[pid_t];
  int m_tile = tile_rows[pid_t];
  if (m_tile <= 0) return;
  int row0 = offsets[e] + m0;
  int src = row_token ? row_token[row0] : row0;
  __shared__ float red[WARPS_K][32];
  float acc = 0.f;
  int sag = K / 128;
  int n_rounds = (k_tiles + WARPS_K - 1) / WARPS_K;
  for (int r = 0; r < n_rounds; r++) {{
    int g;
    {assign};
    float partial = 0.f;
    if (g < k_tiles) {{
      {frag_off}
      const unsigned short* a = x_norm + (size_t)src * (size_t)K + (size_t)g * 32ull;
      float sw = e8m0_to_f32(fbase[256 + rrow]);
      float sa = sA[(size_t)src * (size_t)sag + (size_t)(g >> 2)];
      float dot = 0.f;
      #pragma unroll
      for (int i = 0; i < 16; i++) {{
        unsigned char b = fbase[INVQPA[rrow * 16 + i]];
        float w0 = e2m1_f32_lut_reg(b & 0x0f);
        float w1 = e2m1_f32_lut_reg(b >> 4);
        float a0 = f16bits_to_f32(a[i * 2]);
        float a1 = f16bits_to_f32(a[i * 2 + 1]);
        dot += a0 * w0 + a1 * w1;
      }}
      partial = dot * sw * sa;
    }}
    red[w][lane] = partial;
    __syncthreads();
    if (w == 0) {{
      float s = 0.f;
      #pragma unroll
      for (int ww = 0; ww < WARPS_K; ww++) s += red[ww][lane];
      acc += s;
    }}
    __syncthreads();
  }}
  if (w == 0) {{
    {frag_off.replace("g < k_tiles", "1").split("const unsigned char")[0] if False else ""}
    int j = blockIdx.y;
    int td = blockIdx.y;
    int half = lane >> 4;
    int rrow = lane & 15;
    int col;
    if ({1 if kind == "gu" else 0}) {{
      col = (half == 0) ? (j * 16 + rrow) : (inter + j * 16 + rrow);
    }} else {{
      col = td * 32 + lane;
    }}
    if (row0 < A && col < N) {{
      out[(size_t)row0 * (size_t)N + (size_t)col] = (unsigned short)f32_to_bf16_bits(acc);
    }}
  }}
}}
"""


def _get_fn(name: str, src: str, entry: str):
    if entry not in _FN_CACHE:
        ptx = compile_cuda(src, name)
        _PTX_CACHE[entry] = ptx
        _FN_CACHE[entry] = load_kernel(ptx, entry)
    return _FN_CACHE[entry]


def _as_fp8_uint8(x_fp8: torch.Tensor) -> torch.Tensor:
    if x_fp8.dtype == torch.float8_e4m3fn:
        return x_fp8.view(torch.uint8)
    return x_fp8.contiguous().view(torch.uint8)


def _prequant(x, prequant):
    if prequant is not None:
        x_fp8, sA = prequant
        return _as_fp8_uint8(x_fp8), sA.contiguous().to(torch.float32)
    x_fp8, sA = act_quant_ref(x)
    return _as_fp8_uint8(x_fp8), sA


def xsimt_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale_w: torch.Tensor,
    n_out: int,
    backend: str = "native",
    prequant=None,
    prepacked=None,
) -> torch.Tensor:
    """x [M,K] bf16, packed [N,K/2], scale [N,K/32] -> [M,N] bf16."""
    global _LAUNCHES
    m, k = x.shape
    n = packed.shape[0]
    if n != n_out:
        raise ValueError(f"n_out={n_out} != packed N={n}")
    x_fp8, sA = _prequant(x, prequant)
    if backend == "triton":
        return _xsimt_triton(x_fp8, sA, packed, scale_w, n)
    x_norm = decode_activation_exact((x_fp8, sA))
    if prepacked is None:
        codes, scales = prepack_xsimt(packed, scale_w)
    else:
        codes, scales = prepacked
    n_tiles = codes.shape[0]
    n_groups = codes.shape[1]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    fn = _get_fn("xsimt_gemm_pre", _XSIMT_SRC, "xsimt_gemm_pre")
    _LAUNCHES += 1
    launch(
        fn, (m, n_tiles), 32,
        [x_norm.view(torch.uint16), sA, codes.view(torch.uint8), scales.view(torch.uint8),
         out.view(torch.uint16), m, n, k, n_tiles, n_groups],
    )
    return out


def xsimt_ksplit_grouped(
    kind: str,
    x_norm: torch.Tensor,
    sA: torch.Tensor,
    row_token: torch.Tensor | None,
    frag: torch.Tensor,
    offsets: torch.Tensor,
    expert_index: torch.Tensor,
    m_start: torch.Tensor,
    n_tiles: torch.Tensor,
    tile_rows: torch.Tensor,
    n_out: int,
    inter: int,
    k: int,
    warps: int = 4,
    ordered: bool = True,
    cap: int | None = None,
    A: int | None = None,
) -> torch.Tensor:
    """Grouped XSIMT K-split on a QPA fragment bank. kind in {'gu','down'}."""
    global _LAUNCHES
    if A is None:
        A = row_token.numel() if row_token is not None else x_norm.shape[0]
    out = torch.zeros(A, n_out, dtype=torch.bfloat16, device=x_norm.device)
    t_cap = cap if cap is not None else expert_index.numel()
    n_mac = frag.shape[1] if kind == "gu" else 0
    n_mega = (frag.shape[1] // 2) if kind == "down" else 0
    k_tiles = frag.shape[2]
    if kind == "gu":
        n_tiles_y = n_mac
        frag_stride_e = n_mac * k_tiles * 544
    else:
        n_tiles_y = n_mega
        frag_stride_e = frag.shape[1] * k_tiles * 272
    entry = f"xsimt_{kind}_ksplit_{'ordered' if ordered else 'reordered'}_w{warps}"
    src = _ksplit_xsimt_src(kind, warps, ordered)
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (t_cap, n_tiles_y), 32 * warps,
        [x_norm.view(torch.uint16), sA.contiguous().to(torch.float32),
         row_token,
         frag.view(torch.uint8),
         offsets.to(torch.int32), expert_index.to(torch.int32),
         m_start.to(torch.int32), tile_rows.to(torch.int32), n_tiles.to(torch.int32),
         out.view(torch.uint16),
         A, n_out, inter, k, k_tiles, n_mac, n_mega, frag_stride_e],
        shared=warps * 32 * 4,
    )
    return out


def prepack_expert_bank(packed_e: torch.Tensor, scale_e: torch.Tensor):
    """packed [E,N,K/2], scale [E,N,K/32] -> prepacked banks."""
    key = id(packed_e)
    hit = _BANK_CACHE.get(key)
    if hit is not None:
        return hit
    e, n, k_half = packed_e.shape
    codes_l, scales_l = [], []
    for i in range(e):
        c, s = prepack_xsimt(packed_e[i], scale_e[i])
        codes_l.append(c)
        scales_l.append(s)
    out = (torch.stack(codes_l, 0), torch.stack(scales_l, 0))
    _BANK_CACHE[key] = out
    return out


@triton.jit
def _xsimt_triton_kernel(
    X, SA, PACKED, SCALE, LUT4, LUT2, OUT,
    M, N, K,
    stride_xm, stride_xk,
    stride_sam, stride_sag,
    stride_pn, stride_pk,
    stride_sn, stride_sg,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    n_act_groups = K // 128
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(X + pid_m * stride_xm + offs_k * stride_xk, mask=mask_k, other=0).to(tl.int32)
        a = tl.load(LUT4 + x, mask=mask_k, other=0.0)
        packed_k = offs_k // 2
        p = tl.load(
            PACKED + offs_n[:, None] * stride_pn + packed_k[None, :] * stride_pk,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        )
        nibble = tl.where((offs_k[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
        w = tl.load(LUT2 + nibble.to(tl.int32))
        g = offs_k // 32
        s = tl.load(
            SCALE + offs_n[:, None] * stride_sn + g[None, :] * stride_sg,
            mask=mask_n[:, None] & mask_k[None, :], other=1.0,
        )
        ag = offs_k // 128
        sa = tl.load(SA + pid_m * stride_sam + ag * stride_sag, mask=offs_k < K, other=1.0)
        acc += tl.sum(a[None, :] * w * s * sa[None, :], axis=1)
    tl.store(OUT + pid_m * stride_om + offs_n * stride_on, acc.to(tl.bfloat16), mask=mask_n)


def _xsimt_triton(x_fp8, sA, packed, scale_w, n):
    global _LAUNCHES
    m, k = x_fp8.shape
    from mosaicvm.kernels.fp4_source import e8m0_to_fp32
    scale_fp = e8m0_to_fp32(scale_w)
    lut4 = e4m3_lut_f32(x_fp8.device)
    lut2 = torch.tensor(
        __import__("mosaicvm.kernels.mxfp4", fromlist=["E2M1_LUT"]).E2M1_LUT,
        dtype=torch.float32, device=x_fp8.device,
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_fp8.device)
    _LAUNCHES += 1
    grid = (m, triton.cdiv(n, 16))
    _xsimt_triton_kernel[grid](
        x_fp8.view(torch.uint8), sA, packed, scale_fp, lut4, lut2, out,
        m, n, k,
        x_fp8.stride(0), x_fp8.stride(1),
        sA.stride(0), sA.stride(1),
        packed.stride(0), packed.stride(1),
        scale_fp.stride(0), scale_fp.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=16, BLOCK_K=128,
    )
    return out
