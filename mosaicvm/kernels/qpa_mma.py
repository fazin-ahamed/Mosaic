"""Native QPA (warp MMA) MXFP4 GEMM — K32 scale-aligned fragment.

Tile: 16N x 32K.  Two m16n8k16 MMAs cover K32, then one vector of 16 weight
scales and act scales multiply the K32 partial before fp32 accumulate.

m16n8k16.row.col.f32.f16.f16.f32 fragment layouts (Ampere):
  A (weights [16 row-major x 16 K], f16), thread lane=(group:3,tidg:2):
    a0 = W[g  ][k=2t]  , a0.hi = W[g  ][2t+1]       (b0 carries k=2t,2t+1)
    a1 = W[g+8][2t]    ,                              (b1 k=2t,   2t+1)
    a2 = W[g  ][8+2t]  ,                              (b2 k=8+2t, 8+2t+1)
    a3 = W[g+8][8+2t]  ,                              (b3)
  Each .b32 packs two adjacent K f16 (k, k+1).
  B (act [K x token=N], col-major), thread lane:
    b0 = X[g row-token][k=2t] , b0.hi = [2t+1]
    b1 = X[g][k=8+2t], [8+2t+1]
  C/D (16x8 f32):
    c0,c1 = row g,   token 2t..2t+1 ; c2,c3 = row g+8, token 2t..2t+1

Prepack tile [272 B] per (nt,kt): K16#1 128B, K16#2 128B, scale 16B.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from mosaicvm.kernels.fp4_source import ACT_BLOCK, act_quant_ref
from mosaicvm.kernels.mxfp4 import E2M1_LUT
from mosaicvm.kernels.mxfp4_decode import DEVICE_HELPERS, e4m3_lut_f32
from mosaicvm.kernels.ptx_jit import compile_cuda, launch, load_kernel, ptx_has_m16n8k16
from mosaicvm.kernels.xsimt import decode_activation_exact

MMA = 16
TOKENS_PER_MMA = 8
QPA_TILE_N = 16
QPA_TILE_K = 32
PREPACK_BYTES = 272  # 128 + 128 + 16

_LAUNCHES = 0
_PTX_CACHE: dict[str, bytes] = {}
_FN_CACHE: dict[str, object] = {}
_BANK_CACHE: dict[int, object] = {}
PTX_HAS_M16N8K16 = False
CHOSEN_PREPACK = "qpa_k32_fragment"
CHOSEN_QPA_K = 8
HYBRID = "qpa_xsimt"


@dataclass
class QPAExpertBank:
    frag_gu: torch.Tensor   # [E, inter//16, K//32, 544]
    frag_d: torch.Tensor    # [E, hidden//16, inter//32, 272]
    n_out_gu: int
    n_out_d: int


def last_launch_count() -> int:
    return _LAUNCHES


def _cdiv(a, b):
    return (a + b - 1) // b


def _gather(pk, row_idxs, jidx):
    """pk [n_tiles,16,k_tiles,16] -> values [n_tiles,k_tiles,32] per lane.

    row_idxs, jidx are [32] integer lane maps; returns [nt,kt,lane].
    """
    n_tiles, _, k_tiles, _ = pk.shape
    nn = torch.arange(n_tiles, device=pk.device)[:, None, None]
    kk = torch.arange(k_tiles, device=pk.device)[None, :, None]
    ll = torch.arange(32, device=pk.device)
    vals = pk[nn, row_idxs[None, None, :], kk, jidx[None, None, :]]  # [nt,kt,32]
    return vals


def _scatter(pk, row_idxs, jidx, val):
    """Inverse of _gather: val [nt,kt,32] -> scatter into pk [nt,16,kt,16]."""
    n_tiles, _, k_tiles, _ = pk.shape
    nn = torch.arange(n_tiles, device=pk.device)[:, None, None].expand(-1, -1, 32)
    kk = torch.arange(k_tiles, device=pk.device)[None, :, None].expand(-1, -1, 32)
    ll = torch.arange(32, device=pk.device)
    pk[nn, row_idxs[None, None, :].expand(n_tiles, k_tiles, -1),
       kk, jidx[None, None, :].expand(n_tiles, k_tiles, -1)] = val
    return pk


def prepack_qpa(packed: torch.Tensor, scale: torch.Tensor, stage: str = "qpa_k32_fragment"):
    """packed [N,K/2], scale [N,K/32] -> A-fragment tile [n_tiles,k_tiles,272].

    stage: "identity" | "xsimt_tile" | "qpa_k32_fragment".
    """
    if stage == "identity":
        return packed.view(torch.uint8), scale.view(torch.uint8)
    n, k_half = packed.shape
    k = k_half * 2
    n_tiles = _cdiv(n, QPA_TILE_N)
    k_tiles = k // QPA_TILE_K
    pad_n = n_tiles * QPA_TILE_N - n
    if pad_n:
        packed = torch.nn.functional.pad(packed, (0, 0, 0, pad_n)).contiguous()
        scale = torch.nn.functional.pad(scale.view(torch.uint8), (0, 0, 0, pad_n)).view(scale.dtype)
    p = packed.view(torch.uint8).view(n_tiles, QPA_TILE_N, k_tiles, 16)
    s = scale.view(torch.uint8).view(n_tiles, QPA_TILE_N, k_tiles)

    lane = torch.arange(32, device=packed.device)
    g = lane // 4
    t = lane % 4
    gr = g
    gr8 = g + 8

    # K16#1: a0(row=g  ,j=t), a1(g+8,j=t), a2(g,j=4+t), a3(+8,4+t)
    h1 = torch.stack([
        _gather(p, gr, t), _gather(p, gr8, t),
        _gather(p, gr, 4 + t), _gather(p, gr8, 4 + t),
    ], dim=-1).reshape(n_tiles, k_tiles, 128)
    # K16#2: j = 8+t, 8+t, 12+t, 12+t
    h2 = torch.stack([
        _gather(p, gr, 8 + t), _gather(p, gr8, 8 + t),
        _gather(p, gr, 12 + t), _gather(p, gr8, 12 + t),
    ], dim=-1).reshape(n_tiles, k_tiles, 128)
    scales = s.permute(0, 2, 1).contiguous()  # [nt,kt,16]
    tiles = torch.cat([h1, h2, scales], dim=-1).contiguous()
    return tiles, None


def inverse_prepack_qpa(tiles: torch.Tensor, n: int, k: int):
    """Bit-exact inverse: tiles -> packed [N,K/2], scale_u8 [N,K/32]."""
    n_tiles, k_tiles, _ = tiles.shape
    p = torch.zeros(n_tiles, QPA_TILE_N, k_tiles, 16, dtype=torch.uint8, device=tiles.device)
    lane = torch.arange(32, device=tiles.device)
    g = lane // 4
    gr = g
    gr8 = g + 8
    t = lane % 4
    h1 = tiles[:, :, :128].reshape(n_tiles, k_tiles, 32, 4)
    h2 = tiles[:, :, 128:256].reshape(n_tiles, k_tiles, 32, 4)
    _scatter(p, gr, t, h1[:, :, :, 0])
    _scatter(p, gr8, t, h1[:, :, :, 1])
    _scatter(p, gr, 4 + t, h1[:, :, :, 2])
    _scatter(p, gr8, 4 + t, h1[:, :, :, 3])
    _scatter(p, gr, 8 + t, h2[:, :, :, 0])
    _scatter(p, gr8, 8 + t, h2[:, :, :, 1])
    _scatter(p, gr, 12 + t, h2[:, :, :, 2])
    _scatter(p, gr8, 12 + t, h2[:, :, :, 3])
    packed = p.reshape(n_tiles * QPA_TILE_N, k_tiles * 16)[:n, : k // 2]
    scales = tiles[:, :, 256:].permute(0, 2, 1).reshape(n_tiles * QPA_TILE_N, k_tiles)[:n, : k // 32]
    return packed.contiguous(), scales.contiguous()


def _as_u8(t):
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    return t.contiguous().view(torch.uint8)


def _prep_act(x, prequant):
    if prequant is not None:
        x_fp8, sA = prequant
        return _as_u8(x_fp8), sA.contiguous().to(torch.float32)
    x_fp8, sA = act_quant_ref(x)
    return _as_u8(x_fp8), sA


_QPA_SRC = DEVICE_HELPERS + r"""
__device__ __forceinline__ unsigned e2m1_f16_pair(unsigned char b) {
  unsigned lo = e2m1_f16_lut_reg(b & 0x0f);
  unsigned hi = e2m1_f16_lut_reg(b >> 4);
  return lo | (hi << 16);
}
__device__ __forceinline__ unsigned load_act_b32(
    const unsigned short* x_norm, int token, int k, int M, int K, int kt) {
  if (token < 0 || token >= M) return 0;
  unsigned lo = x_norm[(size_t)token * (size_t)K + (size_t)(kt * 32 + k)];
  unsigned hi = x_norm[(size_t)token * (size_t)K + (size_t)(kt * 32 + k + 1)];
  return lo | (hi << 16);
}
extern "C" __global__ void qpa_gemm_pre(
    const unsigned short* x_norm,
    const float* sA,
    const unsigned char* frag,
    unsigned short* out,
    int M, int N, int K, int n_tiles, int k_tiles
) {
  int lane = threadIdx.x;
  int nt = blockIdx.x;
  if (nt >= n_tiles) return;
  unsigned g = lane >> 2;
  unsigned t = lane & 3;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  int t0 = 2 * (int)t, t1 = 2 * (int)t + 1;
  int sag = K / 128;

  for (int kt = 0; kt < k_tiles; kt++) {
    const unsigned char* tb = frag + (((size_t)nt * (size_t)k_tiles + (size_t)kt) * 272ull);
    unsigned a0 = e2m1_f16_pair(tb[4 * lane + 0]);
    unsigned a1 = e2m1_f16_pair(tb[4 * lane + 1]);
    unsigned a2 = e2m1_f16_pair(tb[4 * lane + 2]);
    unsigned a3 = e2m1_f16_pair(tb[4 * lane + 3]);
    unsigned a4 = e2m1_f16_pair(tb[128 + 4 * lane + 0]);
    unsigned a5 = e2m1_f16_pair(tb[128 + 4 * lane + 1]);
    unsigned a6 = e2m1_f16_pair(tb[128 + 4 * lane + 2]);
    unsigned a7 = e2m1_f16_pair(tb[128 + 4 * lane + 3]);
    float sw_g = e8m0_to_f32(tb[256 + g]);
    float sw_g8 = e8m0_to_f32(tb[256 + g + 8]);
    unsigned b0 = load_act_b32(x_norm, (int)g, 2 * (int)t, M, K, kt);
    unsigned b1 = load_act_b32(x_norm, (int)g, 8 + 2 * (int)t, M, K, kt);
    unsigned b2 = load_act_b32(x_norm, (int)g, 16 + 2 * (int)t, M, K, kt);
    unsigned b3 = load_act_b32(x_norm, (int)g, 24 + 2 * (int)t, M, K, kt);
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
      : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b2), "r"(b3));
    float sa0 = (t0 < M) ? sA[(size_t)t0 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
    float sa1 = (t1 < M) ? sA[(size_t)t1 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
    d0 += p0 * sw_g * sa0;
    d1 += p1 * sw_g * sa1;
    d2 += p2 * sw_g8 * sa0;
    d3 += p3 * sw_g8 * sa1;
  }
  int r0 = nt * 16 + (int)g, r1 = nt * 16 + (int)g + 8;
  if (t0 < M && r0 < N) out[(size_t)t0 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(d0);
  if (t1 < M && r0 < N) out[(size_t)t1 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(d1);
  if (t0 < M && r1 < N) out[(size_t)t0 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(d2);
  if (t1 < M && r1 < N) out[(size_t)t1 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(d3);
}
"""


def _ksplit_direct_src(warps: int, ordered: bool) -> str:
    """Single-GEMM ordered (or reordered) K-split: 16N tile, WARPS_K warps."""
    name = f"qpa_ksplit_direct_{'ordered' if ordered else 'reordered'}_w{warps}"
    if not ordered:
        name = f"qpa_ksplit_direct_reordered_w{warps}"
    else:
        name = f"qpa_ksplit_direct_w{warps}"
    assign = "kt = r * WARPS_K + w" if ordered else "kt = w * n_rounds + r"
    return DEVICE_HELPERS + f"""
__device__ __forceinline__ unsigned e2m1_f16_pair(unsigned char b) {{
  unsigned lo = e2m1_f16_lut_reg(b & 0x0f);
  unsigned hi = e2m1_f16_lut_reg(b >> 4);
  return lo | (hi << 16);
}}
__device__ __forceinline__ unsigned load_act_b32(
    const unsigned short* x_norm, int token, int k, int M, int K, int kt) {{
  if (token < 0 || token >= M) return 0;
  unsigned lo = x_norm[(size_t)token * (size_t)K + (size_t)(kt * 32 + k)];
  unsigned hi = x_norm[(size_t)token * (size_t)K + (size_t)(kt * 32 + k + 1)];
  return lo | (hi << 16);
}}
extern "C" __global__ void {name}(
    const unsigned short* x_norm,
    const float* sA,
    const unsigned char* frag,
    unsigned short* out,
    int M, int N, int K, int n_tiles, int k_tiles
) {{
  enum {{ WARPS_K = {warps} }};
  int nt = blockIdx.x;
  if (nt >= n_tiles) return;
  int lane = threadIdx.x & 31;
  int w = threadIdx.x >> 5;
  unsigned g = lane >> 2;
  unsigned t = lane & 3;
  int t0 = 2 * (int)t, t1 = 2 * (int)t + 1;
  int sag = K / 128;
  __shared__ float red[WARPS_K][32][4];
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  int n_rounds = (k_tiles + WARPS_K - 1) / WARPS_K;
  for (int r = 0; r < n_rounds; r++) {{
    int kt;
    {assign};
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    if (kt < k_tiles) {{
      const unsigned char* tb = frag + (((size_t)nt * (size_t)k_tiles + (size_t)kt) * 272ull);
      unsigned a0 = e2m1_f16_pair(tb[4 * lane + 0]);
      unsigned a1 = e2m1_f16_pair(tb[4 * lane + 1]);
      unsigned a2 = e2m1_f16_pair(tb[4 * lane + 2]);
      unsigned a3 = e2m1_f16_pair(tb[4 * lane + 3]);
      unsigned a4 = e2m1_f16_pair(tb[128 + 4 * lane + 0]);
      unsigned a5 = e2m1_f16_pair(tb[128 + 4 * lane + 1]);
      unsigned a6 = e2m1_f16_pair(tb[128 + 4 * lane + 2]);
      unsigned a7 = e2m1_f16_pair(tb[128 + 4 * lane + 3]);
      float sw_g = e8m0_to_f32(tb[256 + g]);
      float sw_g8 = e8m0_to_f32(tb[256 + g + 8]);
      unsigned b0 = load_act_b32(x_norm, (int)g, 2 * (int)t, M, K, kt);
      unsigned b1 = load_act_b32(x_norm, (int)g, 8 + 2 * (int)t, M, K, kt);
      unsigned b2 = load_act_b32(x_norm, (int)g, 16 + 2 * (int)t, M, K, kt);
      unsigned b3 = load_act_b32(x_norm, (int)g, 24 + 2 * (int)t, M, K, kt);
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
        : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
        : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
        : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b2), "r"(b3));
      float sa0 = (t0 < M) ? sA[(size_t)t0 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
      float sa1 = (t1 < M) ? sA[(size_t)t1 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
      p0 *= sw_g * sa0; p1 *= sw_g * sa1; p2 *= sw_g8 * sa0; p3 *= sw_g8 * sa1;
    }}
    red[w][lane][0] = p0; red[w][lane][1] = p1; red[w][lane][2] = p2; red[w][lane][3] = p3;
    __syncthreads();
    if (w == 0) {{
      #pragma unroll
      for (int ww = 0; ww < WARPS_K; ww++) {{
        acc0 += red[ww][lane][0];
        acc1 += red[ww][lane][1];
        acc2 += red[ww][lane][2];
        acc3 += red[ww][lane][3];
      }}
    }}
    __syncthreads();
  }}
  if (w == 0) {{
    int r0 = nt * 16 + (int)g, r1 = nt * 16 + (int)g + 8;
    if (t0 < M && r0 < N) out[(size_t)t0 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(acc0);
    if (t1 < M && r0 < N) out[(size_t)t1 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(acc1);
    if (t0 < M && r1 < N) out[(size_t)t0 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(acc2);
    if (t1 < M && r1 < N) out[(size_t)t1 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(acc3);
  }}
}}
"""


def _ksplit_grouped_src(kind: str, warps: int, ordered: bool) -> str:
    """kind in {{'gu','down'}}. GU reads 544-B macro-tiles; Down reads two 272-B fragments."""
    tag = "ordered" if ordered else "reordered"
    name = f"qpa_{kind}_ksplit_{tag}_w{warps}"
    assign = "kt = r * WARPS_K + w" if ordered else "kt = w * n_rounds + r"
    if kind == "gu":
        n_y = "n_mac"
        half_loop = """
    int j = blockIdx.y;
    int n_out_half = inter;
"""
        frag_ptr = """
      const unsigned char* tb = frag
        + (size_t)e * (size_t)frag_stride_e
        + (((size_t)j * (size_t)k_tiles + (size_t)kt) * 544ull)
        + (size_t)half * 272ull;
"""
        store = """
    int col0 = (half == 0) ? (j * 16 + (int)g) : (inter + j * 16 + (int)g);
    int col1 = col0 + 8;
"""
    else:
        n_y = "n_mega"
        half_loop = """
    int td = blockIdx.y;
    int n_out_half = N;
"""
        frag_ptr = """
      int nt = td * 2 + half;
      const unsigned char* tb = frag
        + (size_t)e * (size_t)frag_stride_e
        + (((size_t)nt * (size_t)k_tiles + (size_t)kt) * 272ull);
"""
        store = """
    int col0 = td * 32 + half * 16 + (int)g;
    int col1 = col0 + 8;
"""
    return DEVICE_HELPERS + f"""
__device__ __forceinline__ unsigned e2m1_f16_pair(unsigned char b) {{
  unsigned lo = e2m1_f16_lut_reg(b & 0x0f);
  unsigned hi = e2m1_f16_lut_reg(b >> 4);
  return lo | (hi << 16);
}}
__device__ __forceinline__ unsigned load_act_b32_tok(
    const unsigned short* x_norm, const int* row_token, int row, int tok, int k, int m_tile, int K, int kt) {{
  if (tok < 0 || tok >= m_tile) return 0;
  int src = row_token ? row_token[row + tok] : (row + tok);
  unsigned lo = x_norm[(size_t)src * (size_t)K + (size_t)(kt * 32 + k)];
  unsigned hi = x_norm[(size_t)src * (size_t)K + (size_t)(kt * 32 + k + 1)];
  return lo | (hi << 16);
}}
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
  int row = offsets[e] + m0;
  unsigned g = lane >> 2;
  unsigned t = lane & 3;
  int t0 = 2 * (int)t, t1 = 2 * (int)t + 1;
  int sag = K / 128;
  {half_loop}
  __shared__ float red[WARPS_K][32][4];
  int n_rounds = (k_tiles + WARPS_K - 1) / WARPS_K;
  for (int half = 0; half < 2; half++) {{
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int r = 0; r < n_rounds; r++) {{
      int kt;
      {assign};
      float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
      if (kt < k_tiles) {{
        {frag_ptr}
        unsigned a0 = e2m1_f16_pair(tb[4 * lane + 0]);
        unsigned a1 = e2m1_f16_pair(tb[4 * lane + 1]);
        unsigned a2 = e2m1_f16_pair(tb[4 * lane + 2]);
        unsigned a3 = e2m1_f16_pair(tb[4 * lane + 3]);
        unsigned a4 = e2m1_f16_pair(tb[128 + 4 * lane + 0]);
        unsigned a5 = e2m1_f16_pair(tb[128 + 4 * lane + 1]);
        unsigned a6 = e2m1_f16_pair(tb[128 + 4 * lane + 2]);
        unsigned a7 = e2m1_f16_pair(tb[128 + 4 * lane + 3]);
        float sw_g = e8m0_to_f32(tb[256 + g]);
        float sw_g8 = e8m0_to_f32(tb[256 + g + 8]);
        unsigned b0 = load_act_b32_tok(x_norm, row_token, row, (int)g, 2 * (int)t, m_tile, K, kt);
        unsigned b1 = load_act_b32_tok(x_norm, row_token, row, (int)g, 8 + 2 * (int)t, m_tile, K, kt);
        unsigned b2 = load_act_b32_tok(x_norm, row_token, row, (int)g, 16 + 2 * (int)t, m_tile, K, kt);
        unsigned b3 = load_act_b32_tok(x_norm, row_token, row, (int)g, 24 + 2 * (int)t, m_tile, K, kt);
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
          : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
          : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
          : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b2), "r"(b3));
        float sa0 = 0.f, sa1 = 0.f;
        if (t0 < m_tile) {{
          int src0 = row_token ? row_token[row + t0] : (row + t0);
          sa0 = sA[(size_t)src0 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        if (t1 < m_tile) {{
          int src1 = row_token ? row_token[row + t1] : (row + t1);
          sa1 = sA[(size_t)src1 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        p0 *= sw_g * sa0; p1 *= sw_g * sa1; p2 *= sw_g8 * sa0; p3 *= sw_g8 * sa1;
      }}
      red[w][lane][0] = p0; red[w][lane][1] = p1; red[w][lane][2] = p2; red[w][lane][3] = p3;
      __syncthreads();
      if (w == 0) {{
        #pragma unroll
        for (int ww = 0; ww < WARPS_K; ww++) {{
          acc0 += red[ww][lane][0];
          acc1 += red[ww][lane][1];
          acc2 += red[ww][lane][2];
          acc3 += red[ww][lane][3];
        }}
      }}
      __syncthreads();
    }}
    if (w == 0) {{
      {store}
      if (t0 < m_tile && col0 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc0);
      if (t1 < m_tile && col0 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc1);
      if (t0 < m_tile && col1 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc2);
      if (t1 < m_tile && col1 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc3);
    }}
  }}
}}
"""



def _get_fn(name: str, src: str, entry: str):
    global PTX_HAS_M16N8K16
    if entry not in _FN_CACHE:
        ptx = compile_cuda(src, name)
        _PTX_CACHE[entry] = ptx
        PTX_HAS_M16N8K16 = PTX_HAS_M16N8K16 or ptx_has_m16n8k16(ptx)
        _FN_CACHE[entry] = load_kernel(ptx, entry)
    return _FN_CACHE[entry]


def qpa_mma_gemm(
    x, packed, scale_w, n_out,
    backend="native", qpa_k=1, prequant=None, prepacked=None,
    prepack_stage="qpa_k32_fragment", x_norm=None, sA=None, ordered=True,
) -> torch.Tensor:
    """x [M,K] bf16, packed [N,K/2], scale [N,K/32] -> [M,N] bf16.

    If ``x_norm`` (f16) and ``sA`` are provided, skip act_quant/predecode
    (kernel-only roofline). ``ordered=False`` compiles the race-only reordered variant.
    """
    global _LAUNCHES, CHOSEN_PREPACK, CHOSEN_QPA_K
    m, k = x.shape
    n = packed.shape[0]
    if x_norm is None:
        x_fp8, sA = _prep_act(x, prequant)
        if backend == "triton":
            return _qpa_triton(x_fp8, sA, packed, scale_w, n)
        x_norm = decode_activation_exact((x_fp8, sA))
    elif backend == "triton":
        x_fp8, sA2 = _prep_act(x, prequant)
        return _qpa_triton(x_fp8, sA2, packed, scale_w, n)
    if prepacked is None:
        frag, _ = prepack_qpa(packed, scale_w, "qpa_k32_fragment")
        CHOSEN_PREPACK = prepack_stage
    else:
        frag = prepacked
    n_tiles = frag.shape[0]
    k_tiles = frag.shape[1]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    _LAUNCHES += 1
    CHOSEN_QPA_K = qpa_k
    if qpa_k == 1:
        fn = _get_fn("qpa_gemm_pre", _QPA_SRC, "qpa_gemm_pre")
        launch(fn, (n_tiles,), 32,
               [x_norm.view(torch.uint16), sA, frag.view(torch.uint8), out.view(torch.uint16),
                m, n, k, n_tiles, k_tiles])
    else:
        entry = f"qpa_ksplit_direct_{'ordered' if ordered else 'reordered'}_w{qpa_k}" if not ordered else f"qpa_ksplit_direct_w{qpa_k}"
        fn = _get_fn(entry, _ksplit_direct_src(qpa_k, ordered), entry)
        launch(fn, (n_tiles,), 32 * qpa_k,
               [x_norm.view(torch.uint16), sA, frag.view(torch.uint8), out.view(torch.uint16),
                m, n, k, n_tiles, k_tiles],
               shared=qpa_k * 32 * 4 * 4)
    return out




def _prepack_nk_to_frag(packed_e, scale_e):
    """NK bank [E,N,K/2] -> [E,n_tiles,k_tiles,272]."""
    e, n, k_half = packed_e.shape
    k = k_half * 2
    n_tiles = _cdiv(n, QPA_TILE_N)
    k_tiles = k // QPA_TILE_K
    pad_n = n_tiles * QPA_TILE_N - n
    p0 = packed_e.view(torch.uint8)
    s0 = scale_e.view(torch.uint8)
    if pad_n:
        p0 = torch.nn.functional.pad(p0, (0, 0, 0, pad_n))
        s0 = torch.nn.functional.pad(s0, (0, 0, 0, pad_n))
    p = p0.view(e, n_tiles, QPA_TILE_N, k_tiles, 16)
    s = s0.view(e, n_tiles, QPA_TILE_N, k_tiles)
    lane = torch.arange(32, device=packed_e.device)
    g = lane // 4
    t = lane % 4
    ee = torch.arange(e, device=packed_e.device)[:, None, None, None]
    nt = torch.arange(n_tiles, device=packed_e.device)[None, :, None, None]
    kt = torch.arange(k_tiles, device=packed_e.device)[None, None, :, None]

    def take(row, j):
        return p[ee, nt, row[None, None, None, :], kt, j[None, None, None, :]]

    h1 = torch.stack([take(g, t), take(g + 8, t), take(g, 4 + t), take(g + 8, 4 + t)], dim=-1)
    h2 = torch.stack([take(g, 8 + t), take(g + 8, 8 + t), take(g, 12 + t), take(g + 8, 12 + t)], dim=-1)
    codes = torch.cat([h1.reshape(e, n_tiles, k_tiles, 128), h2.reshape(e, n_tiles, k_tiles, 128)], dim=-1)
    scales = s.permute(0, 1, 3, 2).contiguous()
    return torch.cat([codes, scales], dim=-1).contiguous()


def prepack_expert_bank_qpa(packed_g, scale_g, packed_u=None, scale_u=None, packed_d=None, scale_d=None, chunk: int = 32):
    """Build a QPAExpertBank from NK packed tensors.

    Accepts either:
      - (packed_g, scale_g, packed_u, scale_u, packed_d, scale_d) with
        packed_g/u [E, inter, K/2] and packed_d [E, hidden, inter/2], or
      - concat packed_g [E, 2*inter, K/2] (legacy) + packed_d as packed_u.
    Chunked along E so NK + fragment banks are never all resident.
    """
    # concat form: packed_g is [E, 2*inter, K/2] (gate|up along N)
    if packed_u is None:
        e, n2, k_half = packed_g.shape
        inter = n2 // 2
        packed_u = packed_g[:, inter:]
        scale_u = scale_g[:, inter:]
        packed_g = packed_g[:, :inter]
        scale_g = scale_g[:, :inter]
    e = packed_g.shape[0]
    inter = packed_g.shape[1]
    hidden = packed_d.shape[1]
    k_g = packed_g.shape[2] * 2
    k_d = packed_d.shape[2] * 2
    n_mac = _cdiv(inter, QPA_TILE_N)
    k_tiles_g = k_g // QPA_TILE_K
    n_tiles_d = _cdiv(hidden, QPA_TILE_N)
    k_tiles_d = k_d // QPA_TILE_K
    frag_gu = torch.empty(e, n_mac, k_tiles_g, 544, dtype=torch.uint8, device=packed_g.device)
    frag_d = torch.empty(e, n_tiles_d, k_tiles_d, 272, dtype=torch.uint8, device=packed_d.device)
    for i0 in range(0, e, chunk):
        i1 = min(e, i0 + chunk)
        gfrag = _prepack_nk_to_frag(packed_g[i0:i1], scale_g[i0:i1])  # [c, n_mac, kt, 272]
        ufrag = _prepack_nk_to_frag(packed_u[i0:i1], scale_u[i0:i1])
        frag_gu[i0:i1] = torch.cat([gfrag, ufrag], dim=-1)
        del gfrag, ufrag
        frag_d[i0:i1] = _prepack_nk_to_frag(packed_d[i0:i1], scale_d[i0:i1])
    return QPAExpertBank(frag_gu=frag_gu, frag_d=frag_d, n_out_gu=2 * inter, n_out_d=hidden)


@triton.jit
def _qpa_triton_kernel(
    X, SA, PACKED, SCALE, LUT4, LUT2, OUT,
    M, N, K,
    stride_xm, stride_xk,
    stride_sam, stride_sag,
    stride_pn, stride_pk,
    stride_sn, stride_sg,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0).to(tl.int32)
        a = tl.load(LUT4 + x)
        packed_k = offs_k // 2
        p = tl.load(PACKED + offs_n[:, None] * stride_pn + packed_k[None, :] * stride_pk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0)
        nibble = tl.where((offs_k[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
        w = tl.load(LUT2 + nibble.to(tl.int32))
        grp = offs_k // 32
        s = tl.load(SCALE + offs_n[:, None] * stride_sn + grp[None, :] * stride_sg,
                    mask=mask_n[:, None] & mask_k[None, :], other=1.0)
        ag = offs_k // 128
        sa = tl.load(SA + offs_m[:, None] * stride_sam + ag[None, :] * stride_sag,
                     mask=mask_m[:, None] & mask_k[None, :], other=1.0)
        acc += tl.sum(a[:, None, :] * w[None, :, :] * s[None, :, :] * sa[:, None, :], axis=2)
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _qpa_triton(x_fp8, sA, packed, scale_w, n):
    from mosaicvm.kernels.fp4_source import e8m0_to_fp32
    global _LAUNCHES
    m, k = x_fp8.shape
    scale_fp = e8m0_to_fp32(scale_w)
    lut4 = e4m3_lut_f32(x_fp8.device)
    lut2 = torch.tensor(E2M1_LUT, dtype=torch.float32, device=x_fp8.device)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_fp8.device)
    _LAUNCHES += 1
    grid = (_cdiv(m, 8), _cdiv(n, 16))
    _qpa_triton_kernel[grid](
        _as_u8(x_fp8), sA, packed, scale_fp, lut4, lut2, out,
        m, n, k, x_fp8.stride(0), x_fp8.stride(1),
        sA.stride(0), sA.stride(1),
        packed.stride(0), packed.stride(1),
        scale_fp.stride(0), scale_fp.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=16, BLOCK_K=128, BLOCK_M=8,
    )
    return out


# ---------------------------------------------------------------------------
# 2-queue grouped MoE
# ---------------------------------------------------------------------------

def fused_moe_qpa(
    x, expert_ids, expert_weights,
    packed_gate_up=None, scale_gate_up=None,
    packed_down=None, scale_down=None,
    bank: QPAExpertBank | None = None,
    crossover: int = 3,
    qpa_k: int = 8,
    xsimt_k: int = 4,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Hybrid ordered-KSplit MoE: one physical QPAExpertBank, sync-free launches.

    Token activations are quantized+predecoded once. Router weight is applied
    to the FP32 SwiGLU result before the down projection. Routed sum is
    sequential in ascending expert-id order.
    """
    global _LAUNCHES, CHOSEN_QPA_K, HYBRID
    _LAUNCHES = 0
    CHOSEN_QPA_K = qpa_k
    from mosaicvm.kernels.moe_layout import layout_assignments
    from mosaicvm.kernels.w4a16_grouped import emit_tiles
    from mosaicvm.kernels.xsimt import xsimt_ksplit_grouped

    if bank is None:
        if packed_gate_up is None or packed_down is None:
            raise TypeError("fused_moe_qpa requires bank= or packed_gate_up/packed_down")
        bank = prepack_expert_bank_qpa(packed_gate_up, scale_gate_up, packed_d=packed_down, scale_d=scale_down)

    h = x.shape[1]
    inter = bank.n_out_gu // 2
    if expert_ids.ndim == 2:
        k_tok = x.shape[0]
        top_k = expert_ids.shape[1]
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k_tok, device=x.device).repeat_interleave(top_k)
        slot_of = torch.arange(top_k, device=x.device).repeat(k_tok)
        weights = torch.ones(k_tok, top_k, device=x.device, dtype=torch.float32) \
            if expert_weights is None else expert_weights.to(dtype=torch.float32)
    else:
        k_tok = x.shape[0]
        top_k = 1
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k_tok, device=x.device)
        slot_of = torch.zeros(k_tok, device=x.device, dtype=torch.int64)
        weights = torch.ones(k_tok, 1, device=x.device, dtype=torch.float32) \
            if expert_weights is None else expert_weights.reshape(k_tok, 1).to(dtype=torch.float32)

    n_experts = bank.frag_gu.shape[0]
    _sorted, perm, offsets, counts = layout_assignments(flat_ids, n_experts)
    del _sorted
    a = token_of.numel()
    zeros = torch.zeros_like(counts)
    counts_x = torch.where(counts <= crossover, counts, zeros)
    counts_q = torch.where(counts > crossover, counts, zeros)
    cap = max(n_experts, a)
    ex, mx, nx, rx = emit_tiles(counts_x, block_m=1, max_tiles=cap)
    eq, mq, nq, rq = emit_tiles(counts_q, block_m=8, max_tiles=cap)

    # quantize+predecode token activations once
    xf8_tok, sA_tok = act_quant_ref(x, ACT_BLOCK)
    x_norm_tok = decode_activation_exact((xf8_tok, sA_tok))
    row_token = token_of[perm].to(torch.int32)
    w_a = weights.reshape(-1)[perm]

    n_gu = bank.n_out_gu
    k_h = x.shape[1]
    k_i = inter
    try:
        HYBRID = "qpa_xsimt"
        gu = _run_two_queues(
            "gu", x_norm_tok, sA_tok, row_token, bank.frag_gu,
            offsets, eq, mq, nq, rq, ex, mx, nx, rx, n_gu, inter, k_h, a, cap, qpa_k, xsimt_k,
        )
    except Exception:
        HYBRID = "qpa_all"
        gu = _run_two_queues(
            "gu", x_norm_tok, sA_tok, row_token, bank.frag_gu,
            offsets, eq, mq, nq, rq, ex, mx, nx, rx, n_gu, inter, k_h, a, cap, qpa_k, xsimt_k,
            force_qpa=True,
        )

    gate = gu[:, :inter].float()
    up = gu[:, inter:].float()
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    z = torch.nn.functional.silu(gate) * up
    z = z * w_a[:, None]
    zd = z.to(x.dtype)
    xf8_d, sA_d = act_quant_ref(zd, ACT_BLOCK)
    x_norm_d = decode_activation_exact((xf8_d, sA_d))
    yd = _run_two_queues(
        "down", x_norm_d, sA_d, None, bank.frag_d,
        offsets, eq, mq, nq, rq, ex, mx, nx, rx, h, inter, k_i, a, cap, qpa_k, xsimt_k,
        force_qpa=(HYBRID == "qpa_all"),
    )
    partials = torch.zeros(k_tok, top_k, h, dtype=torch.float32, device=x.device)
    partials[token_of[perm], slot_of[perm]] = yd.float()
    order = expert_ids.long().argsort(dim=1) if expert_ids.ndim == 2 else torch.zeros(k_tok, 1, dtype=torch.int64, device=x.device)
    out = torch.zeros(k_tok, h, dtype=torch.float32, device=x.device)
    for s in range(top_k):
        slot = order[:, s]
        out = out + partials.gather(1, slot.view(k_tok, 1, 1).expand(k_tok, 1, h)).squeeze(1)
    return out.to(x.dtype)


def _run_two_queues(kind, x_norm, sA, row_token, frag, offsets,
                    eq, mq, nq, rq, ex, mx, nx, rx,
                    n_out, inter, k, a, cap, qpa_k, xsimt_k, force_qpa: bool = False):
    from mosaicvm.kernels.xsimt import xsimt_ksplit_grouped
    out = torch.zeros(a, n_out, dtype=torch.bfloat16, device=x_norm.device)
    n_mac = frag.shape[1] if kind == "gu" else 0
    n_mega = (frag.shape[1] // 2) if kind == "down" else 0
    k_tiles = frag.shape[2]
    if kind == "gu":
        n_tiles_y = n_mac
        frag_stride_e = n_mac * k_tiles * 544
    else:
        n_tiles_y = n_mega
        frag_stride_e = frag.shape[1] * k_tiles * 272
    rt = row_token

    def _qpa(ei, ms, nt, tr, warps):
        entry = f"qpa_{kind}_ksplit_ordered_w{warps}"
        src = _ksplit_grouped_src(kind, warps, True)
        fn = _get_fn(entry, src, entry)
        launch(
            fn, (cap, n_tiles_y), 32 * warps,
            [x_norm.view(torch.uint16), sA.contiguous().to(torch.float32), rt,
             frag.view(torch.uint8),
             offsets.to(torch.int32), ei.to(torch.int32), ms.to(torch.int32),
             tr.to(torch.int32), nt.to(torch.int32),
             out.view(torch.uint16),
             a, n_out, inter, k, k_tiles, n_mac, n_mega, frag_stride_e],
            shared=warps * 32 * 4 * 4,
        )

    global _LAUNCHES
    _LAUNCHES += 1
    _qpa(eq, mq, nq, rq, qpa_k)
    if force_qpa:
        _LAUNCHES += 1
        _qpa(ex, mx, nx, rx, qpa_k)
    else:
        xout = xsimt_ksplit_grouped(
            kind, x_norm, sA, row_token, frag, offsets, ex, mx, nx, rx,
            n_out, inter, k, warps=xsimt_k, ordered=True, cap=cap, A=a,
        )
        out = out + xout
    return out


def launch_census(fn) -> dict:
    """CUDA launch census via torch.profiler. Falls back to custom_only."""
    try:
        activities = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            fn()
            torch.cuda.synchronize()
        events = list(prof.events())
        names = [e.name for e in events]
        total = sum(1 for n in names if n)
        custom_qpa = sum(1 for n in names if "qpa_" in n.lower())
        custom_xsimt = sum(1 for n in names if "xsimt_" in n.lower())
        custom_mxint = sum(1 for n in names if "mxint_" in n.lower())
        layout = sum(1 for n in names if "emit" in n.lower() or "bincount" in n.lower() or "sort" in n.lower())
        return {
            "total_cuda_launches": total,
            "custom_qpa_launches": custom_qpa,
            "custom_xsimt_launches": custom_xsimt,
            "custom_mxint_launches": custom_mxint,
            "layout_launches": layout,
            "epilogue_launches": max(0, total - custom_qpa - custom_xsimt - custom_mxint - layout),
        }
    except Exception:
        return {"launch_census": "custom_only"}