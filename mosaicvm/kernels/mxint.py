"""Phase 2.9 MXINT: integer E2M1xE4M3 on dp4a.s32.s32 / mma.m16n8k32.s8.

Exact arithmetic: qW = 2*W_E2M1 (s8 in [-12,12]), qA = 512*A_E4M3 (int32,
|qA|<=229376) split into three carry-propagating s8 limbs (a0+256*a1+65536*a2).
AW = (qA*qW)/1024 per element; per-K32 |sum qA*qW| <= 51.38e6 < 2^31 so int32
accumulation is exact, then one fp32 multiply by sw*sa/1024.

Hot8 bank: plain s8 row-major [N,K] (this IS the IMMA A-fragment), no QPA
272-B fragment. QPA in qpa_mma.py stays the frozen Exact reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mosaicvm.kernels.fp4_source import ACT_BLOCK, act_quant_ref
from mosaicvm.kernels.mxfp4 import E2M1_LUT
from mosaicvm.kernels.mxfp4_decode import DEVICE_HELPERS
from mosaicvm.kernels.ptx_jit import compile_cuda, launch, load_kernel

_LUT = None
_LAUNCHES = 0
_HEXA_LAUNCHES = 0
_FN_CACHE: dict[str, object] = {}
PTX_HAS_M16N8K32 = False
PTX_HAS_DP4A = False
CHOSEN_MXINT_K = 8
HYBRID = "dp4a_imma"
LUT_STATUS = "not_built"


def last_launch_count() -> int:
    return _LAUNCHES


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _qw_lut(device) -> torch.Tensor:
    global _LUT
    if _LUT is None or _LUT.device != torch.device(device):
        _LUT = torch.tensor(
            [int(round(2 * v)) for v in E2M1_LUT], dtype=torch.int8, device=device
        )
    return _LUT


def _as_u8(t):
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    return t.contiguous().view(torch.uint8)


# ---------------------------------------------------------------------------
# Host encode
# ---------------------------------------------------------------------------

def qint_encode_w(packed: torch.Tensor) -> torch.Tensor:
    """packed uint8 [...,K/2] -> qW int8 [...,K] = 2*E2M1_LUT[code]."""
    lut = _qw_lut(packed.device)
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return lut[codes].contiguous()


def _split_host(q: torch.Tensor):
    """q int32 [...,K] -> (a0,a1,a2) int8 planes with q == a0+256*a1+65536*a2."""
    q = q.to(torch.int64)
    u0 = q & 0xFF
    a0 = torch.where(u0 >= 128, u0 - 256, u0)
    c1 = (u0 >= 128).to(torch.int64)
    t1 = ((q >> 8) & 0xFF) + c1
    b1 = t1 & 0xFF
    a1 = torch.where(b1 >= 128, b1 - 256, b1)
    c2 = (t1 >> 8) + (b1 >= 128).to(torch.int64)
    t2 = ((q >> 16) & 0xFF) + c2
    b2 = t2 & 0xFF
    a2 = torch.where(b2 >= 128, b2 - 256, b2)
    return a0.to(torch.int8).contiguous(), a1.to(torch.int8).contiguous(), a2.to(torch.int8).contiguous()


def qint_encode_a(x_fp8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x_fp8 e4m3fn [M,K] -> (a0,a1,a2) int8 [M,K] planes of qA = 512*A."""
    from mosaicvm.kernels.fp4_source import decode_e4m3

    x_u8 = _as_u8(x_fp8)
    q = (512.0 * decode_e4m3(x_u8)).round().to(torch.int64)
    return _split_host(q)


def _split_host_qa(q: torch.Tensor):
    """int32/int64 tensor of qA values -> limb planes (helper for tests)."""
    return _split_host(q)


# ---------------------------------------------------------------------------
# Device decode of activations
# ---------------------------------------------------------------------------

_DECODE_INT_SRC = DEVICE_HELPERS + r"""
extern "C" __global__ void decode_activation_int(
    const unsigned char* x_fp8, char* a0, char* a1, char* a2, int n
) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  int q = e4m3_qint(x_fp8[i]);
  int b0, b1, b2;
  split_qA_s8(q, &b0, &b1, &b2);
  a0[i] = (char)b0; a1[i] = (char)b1; a2[i] = (char)b2;
}
extern "C" __global__ void split_qA_probe(
    const int* q, char* a0, char* a1, char* a2, int n
) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  int b0, b1, b2;
  split_qA_s8(q[i], &b0, &b1, &b2);
  a0[i] = (char)b0; a1[i] = (char)b1; a2[i] = (char)b2;
}
"""


def decode_activation_int(pair) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_fp8 e4m3fn [M,K], sA) -> (a0,a1,a2) int8 [M,K] planes."""
    x_fp8 = pair[0] if isinstance(pair, tuple) else pair
    x_u8 = _as_u8(x_fp8)
    n = x_u8.numel()
    a0 = torch.empty(x_u8.shape, dtype=torch.int8, device=x_u8.device)
    a1 = torch.empty_like(a0)
    a2 = torch.empty_like(a0)
    fn = _get_fn("decode_activation_int", _DECODE_INT_SRC, "decode_activation_int")
    launch(fn, (_cdiv(n, 256),), 256, [x_u8, a0, a1, a2, n])
    return a0, a1, a2


def split_qA_device(q_host: torch.Tensor):
    """q int32/int64 [..] on CUDA -> (a0,a1,a2) int8 planes via device split_qA_s8."""
    q = q_host.to(torch.int32).contiguous().view(-1)
    n = q.numel()
    a0 = torch.empty(n, dtype=torch.int8, device=q.device)
    a1 = torch.empty_like(a0)
    a2 = torch.empty_like(a0)
    fn = _get_fn("split_qA_probe", _DECODE_INT_SRC, "split_qA_probe")
    launch(fn, (_cdiv(n, 256),), 256, [q, a0, a1, a2, n])
    return a0.view(q_host.shape), a1.view(q_host.shape), a2.view(q_host.shape)


# ---------------------------------------------------------------------------
# Expert bank (Hot8)
# ---------------------------------------------------------------------------

@dataclass
class MXINTExpertBank:
    w_gu: torch.Tensor   # int8 [E, 2*inter, K]
    s_gu: torch.Tensor   # uint8 [E, 2*inter, K/32]
    w_d: torch.Tensor    # int8 [E, hidden, inter]
    s_d: torch.Tensor    # uint8 [E, hidden, inter/32]
    n_out_gu: int
    n_out_d: int


def prepack_expert_bank_mxint(
    packed_g, scale_g, packed_u=None, scale_u=None, packed_d=None, scale_d=None,
    chunk: int = 32,
) -> MXINTExpertBank:
    """Build an MXINTExpertBank from NK packed tensors (concat or split gate/up)."""
    from mosaicvm.runtime.memory_budget import budget

    if packed_u is None:
        e, n2, _ = packed_g.shape
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
    w_gu = torch.empty(e, 2 * inter, k_g, dtype=torch.int8, device=packed_g.device)
    s_gu = torch.empty(e, 2 * inter, k_g // 32, dtype=torch.uint8, device=packed_g.device)
    w_d = torch.empty(e, hidden, k_d, dtype=torch.int8, device=packed_d.device)
    s_d = torch.empty(e, hidden, k_d // 32, dtype=torch.uint8, device=packed_d.device)
    b = budget()
    for i0 in range(0, e, chunk):
        i1 = min(e, i0 + chunk)
        pg, sg = packed_g[i0:i1], scale_g[i0:i1].view(torch.uint8)
        pu, su = packed_u[i0:i1], scale_u[i0:i1].view(torch.uint8)
        pd, sd = packed_d[i0:i1], scale_d[i0:i1].view(torch.uint8)
        w_gu[i0:i1, :inter] = qint_encode_w(pg)
        w_gu[i0:i1, inter:] = qint_encode_w(pu)
        s_gu[i0:i1, :inter] = sg
        s_gu[i0:i1, inter:] = su
        w_d[i0:i1] = qint_encode_w(pd)
        s_d[i0:i1] = sd
        del pg, sg, pu, su, pd, sd
    b.charge_device(w_gu, "mxint.w_gu")
    b.charge_device(s_gu, "mxint.s_gu")
    b.charge_device(w_d, "mxint.w_d")
    b.charge_device(s_d, "mxint.s_d")
    return MXINTExpertBank(w_gu=w_gu, s_gu=s_gu, w_d=w_d, s_d=s_d,
                           n_out_gu=2 * inter, n_out_d=hidden)

@dataclass
class MXINTPacked4Bank:
    p_gu: torch.Tensor  # uint8 [E, 2*inter, K/2]
    s_gu: torch.Tensor  # uint8 [E, 2*inter, K/32]
    p_d:  torch.Tensor  # uint8 [E, hidden, inter/2]
    s_d:  torch.Tensor  # uint8 [E, hidden, inter/32]
    n_out_gu: int
    n_out_d: int


@dataclass
class MXINTHybridBank:
    hot8: MXINTExpertBank
    packed4: MXINTPacked4Bank
    hot_mask: torch.Tensor | None  # uint8 [E], 1 = Hot8, 0 = packed4


@dataclass
class MXINTMaskedBank:
    """Full packed4 + compact hot8 banks with per-expert/per-matrix hot maps.

    gu_hot_ids[e] = compact row in h8_gu_w for expert e, or -1 (packed4).
    VRAM = full packed4 + only the hot experts' hot8 bytes (§17 masks).
    """
    p4: MXINTPacked4Bank
    h8_gu_w: torch.Tensor          # int8 [Eh_gu, 2*inter, K]
    h8_gu_s: torch.Tensor          # uint8 [Eh_gu, 2*inter, K/32]
    h8_down_w: torch.Tensor        # int8 [Eh_dn, H, inter]
    h8_down_s: torch.Tensor        # uint8 [Eh_dn, H, inter/32]
    gu_hot_ids: torch.Tensor       # int32 [E], -1 = cold
    down_hot_ids: torch.Tensor     # int32 [E], -1 = cold


def prepack_expert_bank_mxint_packed4(
    packed_g, scale_g, packed_u=None, scale_u=None, packed_d=None, scale_d=None,
    chunk: int = 32,
) -> MXINTPacked4Bank:
    """Build an MXINTPacked4Bank from NK packed tensors (no qint_encode_w)."""
    from mosaicvm.runtime.memory_budget import budget

    if packed_u is None:
        e, n2, _ = packed_g.shape
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
    p_gu = torch.empty(e, 2 * inter, k_g // 2, dtype=torch.uint8, device=packed_g.device)
    s_gu = torch.empty(e, 2 * inter, k_g // 32, dtype=torch.uint8, device=packed_g.device)
    p_d = torch.empty(e, hidden, k_d // 2, dtype=torch.uint8, device=packed_d.device)
    s_d = torch.empty(e, hidden, k_d // 32, dtype=torch.uint8, device=packed_d.device)
    b = budget()
    for i0 in range(0, e, chunk):
        i1 = min(e, i0 + chunk)
        pg = packed_g[i0:i1].view(torch.uint8) if packed_g.dtype != torch.uint8 else packed_g[i0:i1]
        sg = scale_g[i0:i1].view(torch.uint8)
        pu = packed_u[i0:i1].view(torch.uint8) if packed_u.dtype != torch.uint8 else packed_u[i0:i1]
        su = scale_u[i0:i1].view(torch.uint8)
        pd = packed_d[i0:i1].view(torch.uint8) if packed_d.dtype != torch.uint8 else packed_d[i0:i1]
        sd = scale_d[i0:i1].view(torch.uint8)
        p_gu[i0:i1, :inter] = pg
        p_gu[i0:i1, inter:] = pu
        s_gu[i0:i1, :inter] = sg
        s_gu[i0:i1, inter:] = su
        p_d[i0:i1] = pd
        s_d[i0:i1] = sd
        del pg, sg, pu, su, pd, sd
    b.charge_device(p_gu, "mxint.p_gu")
    b.charge_device(s_gu, "mxint.s_gu")
    b.charge_device(p_d, "mxint.p_d")
    b.charge_device(s_d, "mxint.s_d")
    return MXINTPacked4Bank(p_gu=p_gu, s_gu=s_gu, p_d=p_d, s_d=s_d,
                            n_out_gu=2 * inter, n_out_d=hidden)


# ---------------------------------------------------------------------------
# DP4A GEMM (M=1..3): one lane per output neuron
# ---------------------------------------------------------------------------

_PACKED4_HELP = r"""
__device__ __forceinline__ int e2m1_pair_b32(unsigned char b0, unsigned char b1) {
  // two packed MXFP4 bytes -> one b32 of four qW s8 (k, k+1, k+2, k+3)
  int wv;
  char* p = (char*)&wv;
  p[0] = (char)e2m1_qint(b0 & 15);
  p[1] = (char)e2m1_qint(b0 >> 4);
  p[2] = (char)e2m1_qint(b1 & 15);
  p[3] = (char)e2m1_qint(b1 >> 4);
  return wv;
}
"""


def _dp4a_src(packed: bool) -> str:
    if packed:
        name = "mxint_dp4a_packed4_gemm_pre"
        wdecl = "    const unsigned char* w,"
        wb = "  const unsigned char* rw = w + (size_t)n * (size_t)(K / 2);"
        wload = "      int wb = e2m1_pair_b32(rw[g * 16 + i * 2], rw[g * 16 + i * 2 + 1]);"
        pre = _PACKED4_HELP
    else:
        name = "mxint_dp4a_gemm_pre"
        wdecl = "    const char* w,"
        wb = "  const char* rw = w + (size_t)n * (size_t)K;"
        wload = "      int wb = *(const int*)(rw + g * 32 + i * 4);"
        pre = ""
    return DEVICE_HELPERS + pre + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
{wdecl}
    const unsigned char* scale,
    unsigned short* out,
    int M, int N, int K, int n_groups
) {{
  int m = blockIdx.x;
  int nt = blockIdx.y;
  int lane = threadIdx.x;
  int n = nt * 32 + lane;
  if (m >= M || nt >= (N + 31) / 32 || n >= N) return;
  float acc = 0.f;
  const char* r0 = a0 + (size_t)m * (size_t)K;
  const char* r1 = a1 + (size_t)m * (size_t)K;
  const char* r2 = a2 + (size_t)m * (size_t)K;
{wb}
  const unsigned char* sc = scale + (size_t)n * (size_t)n_groups;
  int sag = K / 128;
  const float* sa_row = sA + (size_t)m * (size_t)sag;
  for (int g = 0; g < n_groups; g++) {{
    int D0 = 0, D1 = 0, D2 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {{
      int x0 = *(const int*)(r0 + g * 32 + i * 4);
      int x1 = *(const int*)(r1 + g * 32 + i * 4);
      int x2 = *(const int*)(r2 + g * 32 + i * 4);
{wload}
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D0) : "r"(x0), "r"(wb));
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D1) : "r"(x1), "r"(wb));
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D2) : "r"(x2), "r"(wb));
    }}
    float sw = e8m0_to_f32(sc[g]);
    float sa = sa_row[g >> 2];
    acc += ((float)(D0 + 256 * D1 + 65536 * D2)) * (1.f / 1024.f) * sw * sa;
  }}
  out[(size_t)m * (size_t)N + (size_t)n] = (unsigned short)f32_to_bf16_bits(acc);
}}
"""


def mxint_dp4a_gemm(x, packed, scale_w, n_out, prequant=None, xint=None, prepacked=None,
                    weight_repr: str = "hot8") -> torch.Tensor:
    """x [M,K] bf16, packed [N,K/2], scale [N,K/32] -> [M,N] bf16. M<=3.

    ``prepacked`` is the Hot8 s8 [N,K] bank (hot8) or the packed uint8
    [N,K/2] tensor itself (packed4). Kernel-only roofline.
    """
    global _LAUNCHES
    m, k = x.shape
    n = packed.shape[0]
    if xint is None:
        if prequant is None:
            x_fp8, sA = act_quant_ref(x, ACT_BLOCK)
        else:
            x_fp8, sA = prequant
            sA = sA.contiguous().to(torch.float32)
        a0, a1, a2 = decode_activation_int((x_fp8, sA))
    else:
        a0, a1, a2, sA = xint
    if weight_repr == "hot8":
        wq = prepacked if prepacked is not None else qint_encode_w(packed)
        entry = "mxint_dp4a_gemm_pre"
        src = _dp4a_src(False)
    elif weight_repr == "packed4":
        wq = prepacked if prepacked is not None else packed
        entry = "mxint_dp4a_packed4_gemm_pre"
        src = _dp4a_src(True)
    else:
        raise ValueError(f"weight_repr must be hot8|packed4, got {weight_repr}")
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    n_groups = k // 32
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (m, _cdiv(n, 32)), 32,
        [a0, a1, a2, sA.contiguous().to(torch.float32), wq,
         scale_w.view(torch.uint8) if scale_w.dtype != torch.uint8 else scale_w,
         out.view(torch.uint16), m, n, k, n_groups],
    )
    return out


def _dp4a_grouped_src(packed: bool) -> str:
    if packed:
        name = "mxint_dp4a_grouped_packed4"
        wdecl = "    const unsigned char* w,"
        wb = "  const unsigned char* wb = w + (size_t)e * (size_t)w_stride_e + (size_t)n * (size_t)(K / 2);"
        wload = "        int wv = e2m1_pair_b32(wb[g * 16 + i * 2], wb[g * 16 + i * 2 + 1]);"
        pre = _PACKED4_HELP
    else:
        name = "mxint_dp4a_grouped"
        wdecl = "    const char* w,"
        wb = "  const char* wb = w + (size_t)e * (size_t)w_stride_e + (size_t)n * (size_t)K;"
        wload = "        int wv = *(const int*)(wb + g * 32 + i * 4);"
        pre = ""
    return DEVICE_HELPERS + pre + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
    const int* row_token,
{wdecl}
    const unsigned char* scale,
    const int* offsets,
    const int* expert_index,
    const int* m_start,
    const int* tile_rows,
    const int* n_tiles_ptr,
    unsigned short* out,
    int A, int N, int K, int n_groups, int e_stride_n, long long w_stride_e, long long s_stride_e
) {{
  int pid_t = blockIdx.x;
  int n_tiles = n_tiles_ptr[0];
  if (pid_t >= n_tiles) return;
  int nt = blockIdx.y;
  int lane = threadIdx.x;
  int n = nt * 32 + lane;
  if (n >= N) return;
  int e = expert_index[pid_t];
  int m0 = m_start[pid_t];
  int m_tile = tile_rows[pid_t];
  if (m_tile <= 0) return;
  int row0 = offsets[e] + m0;
{wb}
  const unsigned char* sc = scale + (size_t)e * (size_t)s_stride_e + (size_t)n * (size_t)n_groups;
  int sag = K / 128;
  float acc = 0.f;
  #pragma unroll 4
  for (int tok = 0; tok < 8; tok++) {{
    if (tok >= m_tile) break;
    int row = row0 + tok;
    int src = row_token ? row_token[row] : row;
    const char* r0 = a0 + (size_t)src * (size_t)K;
    const char* r1 = a1 + (size_t)src * (size_t)K;
    const char* r2 = a2 + (size_t)src * (size_t)K;
    float tacc = 0.f;
    #pragma unroll
    for (int g = 0; g < n_groups; g++) {{
      int D0 = 0, D1 = 0, D2 = 0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {{
        int x0 = *(const int*)(r0 + g * 32 + i * 4);
        int x1 = *(const int*)(r1 + g * 32 + i * 4);
        int x2 = *(const int*)(r2 + g * 32 + i * 4);
{wload}
        asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D0) : "r"(x0), "r"(wv));
        asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D1) : "r"(x1), "r"(wv));
        asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D2) : "r"(x2), "r"(wv));
      }}
      float sw = e8m0_to_f32(sc[g]);
      float sa = sA[(size_t)src * (size_t)sag + (size_t)(g >> 2)];
      tacc += ((float)(D0 + 256 * D1 + 65536 * D2)) * (1.f / 1024.f) * sw * sa;
    }}
    acc = tacc;
    if (row < A) out[(size_t)row * (size_t)N + (size_t)n] = (unsigned short)f32_to_bf16_bits(acc);
  }}
}}
"""


def mxint_dp4a_grouped(
    a0, a1, a2, sA, row_token, w, scale, offsets, expert_index, m_start,
    n_tiles, tile_rows, n_out, k, cap=None, A=None, weight_repr: str = "hot8",
) -> torch.Tensor:
    """Grouped DP4A on an s8 (hot8) or packed uint8 (packed4) bank."""
    global _LAUNCHES
    if A is None:
        A = row_token.numel() if row_token is not None else a0.shape[0]
    out = torch.zeros(A, n_out, dtype=torch.bfloat16, device=a0.device)
    t_cap = cap if cap is not None else expert_index.numel()
    n_groups = k // 32
    e_stride_n = w.shape[1] if w.dim() == 3 else n_out
    w_stride_e = w.shape[1] * w.shape[2] if w.dim() == 3 else 0
    s_stride_e = scale.shape[1] * scale.shape[2] if scale.dim() == 3 else 0
    wv = w if w.dim() == 3 else w.unsqueeze(0)
    sv = scale if scale.dim() == 3 else scale.unsqueeze(0)
    if weight_repr == "hot8":
        entry = "mxint_dp4a_grouped"
        src = _dp4a_grouped_src(False)
    elif weight_repr == "packed4":
        entry = "mxint_dp4a_grouped_packed4"
        src = _dp4a_grouped_src(True)
    else:
        raise ValueError(f"weight_repr must be hot8|packed4, got {weight_repr}")
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (t_cap, _cdiv(n_out, 32)), 32,
        [a0, a1, a2, sA.contiguous().to(torch.float32), row_token,
         wv, sv,
         offsets.to(torch.int32), expert_index.to(torch.int32),
         m_start.to(torch.int32), tile_rows.to(torch.int32), n_tiles.to(torch.int32),
         out.view(torch.uint16),
         A, n_out, k, n_groups, e_stride_n, w_stride_e, s_stride_e],
    )
    return out


# ---------------------------------------------------------------------------
# IMMA ordered K-split (M>=4)
# ---------------------------------------------------------------------------

_MMA_S8 = r"""
__device__ __forceinline__ void mma_s8k32(
    int* d0, int* d1, int* d2, int* d3,
    int a0, int a1, int a2, int a3, int b0, int b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(*d0), "+r"(*d1), "+r"(*d2), "+r"(*d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int load_act_s8b32(
    const char* plane, int token, int koff, int M, int K, int kt) {
  if (token < 0 || token >= M) return 0;
  return *(const int*)(plane + (size_t)token * (size_t)K + (size_t)(kt * 32 + koff));
}
__device__ __forceinline__ int load_act_s8b32_tok(
    const char* plane, const int* row_token, int row, int tok, int koff, int m_tile, int K, int kt) {
  if (tok < 0 || tok >= m_tile) return 0;
  int src = row_token ? row_token[row + tok] : (row + tok);
  return *(const int*)(plane + (size_t)src * (size_t)K + (size_t)(kt * 32 + koff));
}
"""


def _imma_direct_src(warps: int, ordered: bool, packed: bool = False) -> str:
    """Single-GEMM ordered K-split IMMA: 16N tile, WARPS_K warps, 3 limbs."""
    tag = "ordered" if ordered else "reordered"
    if packed:
        name = f"mxint_imma_ksplit_{tag}_packed4_w{warps}"
        wdecl = "    const unsigned char* w,"
        frag = f"""
      const unsigned char* pkb = w + (size_t)(nt * 16) * (size_t)(K / 2) + (size_t)kt * 16ull;
      int aw0 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
      int aw1 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
      int aw2 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
      int aw3 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
"""
        pre = _PACKED4_HELP
    else:
        name = f"mxint_imma_ksplit_{tag}_w{warps}"
        wdecl = "    const char* w,"
        frag = f"""
      const char* wb = w + (size_t)(nt * 16) * (size_t)K + (size_t)kt * 32ull;
      int aw0 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t));
      int aw1 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t));
      int aw2 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t + 16));
      int aw3 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t + 16));
"""
        pre = ""
    assign = "kt = r * WARPS_K + wid" if ordered else "kt = wid * n_rounds + r"
    return DEVICE_HELPERS + _MMA_S8 + pre + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
{wdecl}
    const unsigned char* scale,
    unsigned short* out,
    int M, int N, int K, int n_tiles, int k_tiles
) {{
  enum {{ WARPS_K = {warps} }};
  int nt = blockIdx.x;
  if (nt >= n_tiles) return;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
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
{frag}
      int x00 = load_act_s8b32(a0, (int)g, 4 * (int)t, M, K, kt);
      int x01 = load_act_s8b32(a0, (int)g, 4 * (int)t + 16, M, K, kt);
      int x10 = load_act_s8b32(a1, (int)g, 4 * (int)t, M, K, kt);
      int x11 = load_act_s8b32(a1, (int)g, 4 * (int)t + 16, M, K, kt);
      int x20 = load_act_s8b32(a2, (int)g, 4 * (int)t, M, K, kt);
      int x21 = load_act_s8b32(a2, (int)g, 4 * (int)t + 16, M, K, kt);
      int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
      int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
      int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
      mma_s8k32(&d0, &d1, &d2, &d3, aw0, aw1, aw2, aw3, x00, x01);
      mma_s8k32(&e0, &e1, &e2, &e3, aw0, aw1, aw2, aw3, x10, x11);
      mma_s8k32(&f0, &f1, &f2, &f3, aw0, aw1, aw2, aw3, x20, x21);
      int D0 = d0 + 256 * e0 + 65536 * f0;
      int D1 = d1 + 256 * e1 + 65536 * f1;
      int D2 = d2 + 256 * e2 + 65536 * f2;
      int D3 = d3 + 256 * e3 + 65536 * f3;
      float sw_g = e8m0_to_f32(scale[(size_t)(nt * 16 + g) * (size_t)(K / 32) + (size_t)kt]);
      float sw_g8 = e8m0_to_f32(scale[(size_t)(nt * 16 + g + 8) * (size_t)(K / 32) + (size_t)kt]);
      float sa0 = (t0 < M) ? sA[(size_t)t0 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
      float sa1 = (t1 < M) ? sA[(size_t)t1 * (size_t)sag + (size_t)(kt >> 2)] : 0.f;
      p0 = ((float)D0) * (1.f / 1024.f) * sw_g * sa0;
      p1 = ((float)D1) * (1.f / 1024.f) * sw_g * sa1;
      p2 = ((float)D2) * (1.f / 1024.f) * sw_g8 * sa0;
      p3 = ((float)D3) * (1.f / 1024.f) * sw_g8 * sa1;
    }}
    red[wid][lane][0] = p0; red[wid][lane][1] = p1; red[wid][lane][2] = p2; red[wid][lane][3] = p3;
    __syncthreads();
    if (wid == 0) {{
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
  if (wid == 0) {{
    int r0 = nt * 16 + (int)g, r1 = r0 + 8;
    if (t0 < M && r0 < N) out[(size_t)t0 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(acc0);
    if (t1 < M && r0 < N) out[(size_t)t1 * (size_t)N + (size_t)r0] = (unsigned short)f32_to_bf16_bits(acc1);
    if (t0 < M && r1 < N) out[(size_t)t0 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(acc2);
    if (t1 < M && r1 < N) out[(size_t)t1 * (size_t)N + (size_t)r1] = (unsigned short)f32_to_bf16_bits(acc3);
  }}
}}
"""


def _imma_grouped_src(kind: str, warps: int, ordered: bool, packed: bool = False) -> str:
    """kind in {{'gu','down'}}. Ordered K-split IMMA on the s8 Hot8 / packed4 bank."""
    tag = "ordered" if ordered else "reordered"
    if packed:
        name = f"mxint_{kind}_ksplit_{tag}_packed4_w{warps}"
        wdecl = "    const unsigned char* w,"
    else:
        name = f"mxint_{kind}_ksplit_{tag}_w{warps}"
        wdecl = "    const char* w,"
    assign = "kt = r * WARPS_K + wid" if ordered else "kt = wid * n_rounds + r"
    if kind == "gu":
        head = """
    int j = blockIdx.y;
"""
        nbase = "j * 16 + half * inter"
        store = """
    int col0 = j * 16 + half * inter + (int)g;
    int col1 = col0 + 8;
"""
    else:
        head = """
    int td = blockIdx.y;
"""
        nbase = "td * 32 + half * 16"
        store = """
    int col0 = td * 32 + half * 16 + (int)g;
    int col1 = col0 + 8;
"""
    if packed:
        frag = f"""
      int nb = {nbase};
      const unsigned char* pkb = w
        + (size_t)e * (size_t)w_stride_e
        + (size_t)nb * (size_t)(K / 2)
        + (size_t)kt * 16ull;
      const unsigned char* sb = scale
        + (size_t)e * (size_t)s_stride_e
        + (size_t)kt;
      int aw0 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
      int aw1 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
      int aw2 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
      int aw3 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
"""
        pre = _PACKED4_HELP
    else:
        frag = f"""
      int nb = {nbase};
      const char* wb = w
        + (size_t)e * (size_t)w_stride_e
        + (size_t)nb * (size_t)K
        + (size_t)kt * 32ull;
      const unsigned char* sb = scale
        + (size_t)e * (size_t)s_stride_e
        + (size_t)kt;
      // NOTE: rows are addressed by full row offset at the load site;
      // do NOT fold nb*(K/32) into this base (that was the
      // GROUPED_IMMA_TILE_BUG: sb[g] advanced the kt column instead of
      // selecting row nb+g).
      int aw0 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t));
      int aw1 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t));
      int aw2 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t + 16));
      int aw3 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t + 16));
"""
        pre = ""

    return DEVICE_HELPERS + _MMA_S8 + pre + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
    const int* row_token,
{wdecl}
    const unsigned char* scale,
    const int* offsets,
    const int* expert_index,
    const int* m_start,
    const int* tile_rows,
    const int* n_tiles_ptr,
    unsigned short* out,
    int A, int N, int inter, int K, int k_tiles, int n_mac, int n_mega,
    long long w_stride_e, long long s_stride_e
) {{
  enum {{ WARPS_K = {warps} }};
  int pid_t = blockIdx.x;
  int n_tiles = n_tiles_ptr[0];
  if (pid_t >= n_tiles) return;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  int e = expert_index[pid_t];
  int m0 = m_start[pid_t];
  int m_tile = tile_rows[pid_t];
  if (m_tile <= 0) return;
  int row = offsets[e] + m0;
  unsigned g = lane >> 2;
  unsigned t = lane & 3;
  int t0 = 2 * (int)t, t1 = 2 * (int)t + 1;
  int sag = K / 128;
  {head}
  __shared__ float red[WARPS_K][32][4];
  int n_rounds = (k_tiles + WARPS_K - 1) / WARPS_K;
  for (int half = 0; half < 2; half++) {{
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int r = 0; r < n_rounds; r++) {{
      int kt;
      {assign};
      float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
      if (kt < k_tiles) {{
{frag}
        int x00 = load_act_s8b32_tok(a0, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x01 = load_act_s8b32_tok(a0, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int x10 = load_act_s8b32_tok(a1, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x11 = load_act_s8b32_tok(a1, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int x20 = load_act_s8b32_tok(a2, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x21 = load_act_s8b32_tok(a2, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
        int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
        mma_s8k32(&d0, &d1, &d2, &d3, aw0, aw1, aw2, aw3, x00, x01);
        mma_s8k32(&e0, &e1, &e2, &e3, aw0, aw1, aw2, aw3, x10, x11);
        mma_s8k32(&f0, &f1, &f2, &f3, aw0, aw1, aw2, aw3, x20, x21);
        int D0 = d0 + 256 * e0 + 65536 * f0;
        int D1 = d1 + 256 * e1 + 65536 * f1;
        int D2 = d2 + 256 * e2 + 65536 * f2;
        int D3 = d3 + 256 * e3 + 65536 * f3;
        float sw_g = e8m0_to_f32(sb[(size_t)(nb + (int)g) * (size_t)(K / 32)]);
        float sw_g8 = e8m0_to_f32(sb[(size_t)(nb + (int)g + 8) * (size_t)(K / 32)]);
        float sa0 = 0.f, sa1 = 0.f;
        if (t0 < m_tile) {{
          int src0 = row_token ? row_token[row + t0] : (row + t0);
          sa0 = sA[(size_t)src0 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        if (t1 < m_tile) {{
          int src1 = row_token ? row_token[row + t1] : (row + t1);
          sa1 = sA[(size_t)src1 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        p0 = ((float)D0) * (1.f / 1024.f) * sw_g * sa0;
        p1 = ((float)D1) * (1.f / 1024.f) * sw_g * sa1;
        p2 = ((float)D2) * (1.f / 1024.f) * sw_g8 * sa0;
        p3 = ((float)D3) * (1.f / 1024.f) * sw_g8 * sa1;
      }}
      red[wid][lane][0] = p0; red[wid][lane][1] = p1; red[wid][lane][2] = p2; red[wid][lane][3] = p3;
      __syncthreads();
      if (wid == 0) {{
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
    if (wid == 0) {{
      {store}
      if (t0 < m_tile && col0 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc0);
      if (t1 < m_tile && col0 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc1);
      if (t0 < m_tile && col1 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc2);
      if (t1 < m_tile && col1 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc3);
    }}
    __syncthreads();
  }}
}}
"""

def _imma_grouped_masked_src(kind: str, warps: int) -> str:
    """Masked grouped IMMA: packed4 full bank + compact hot8 bank + hot_ids[E].

    Per expert tile, hot_ids[e] >= 0 -> read hot8 weights (compact row),
    else packed4. Single launch; the branch is uniform per block. Scales:
    each bank carries its own (identical values when derived from one source).
    """
    if kind == "gu":
        head = """
    int j = blockIdx.y;
"""
        nbase = "j * 16 + half * inter"
        store = """
    int col0 = j * 16 + half * inter + (int)g;
    int col1 = col0 + 8;
"""
    else:
        head = """
    int td = blockIdx.y;
"""
        nbase = "td * 32 + half * 16"
        store = """
    int col0 = td * 32 + half * 16 + (int)g;
    int col1 = col0 + 8;
"""
    name = f"mxint_{kind}_masked_w{warps}"
    return DEVICE_HELPERS + _MMA_S8 + _PACKED4_HELP + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
    const int* row_token,
    const unsigned char* w_p4, const unsigned char* s_p4,
    const char* w_h8, const unsigned char* s_h8,
    const int* hot_ids,
    const int* offsets,
    const int* expert_index,
    const int* m_start,
    const int* tile_rows,
    const int* n_tiles_ptr,
    unsigned short* out,
    int A, int N, int inter, int K, int k_tiles,
    long long w_p4_stride_e, long long w_h8_stride_e, long long s_stride_e
) {{
  enum {{ WARPS_K = {warps} }};
  int pid_t = blockIdx.x;
  int n_tiles = n_tiles_ptr[0];
  if (pid_t >= n_tiles) return;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  int e = expert_index[pid_t];
  int m0 = m_start[pid_t];
  int m_tile = tile_rows[pid_t];
  if (m_tile <= 0) return;
  int row = offsets[e] + m0;
  unsigned g = lane >> 2;
  unsigned t = lane & 3;
  int t0 = 2 * (int)t, t1 = 2 * (int)t + 1;
  int sag = K / 128;
  int hrow = hot_ids[e];
{head}
  __shared__ float red[WARPS_K][32][4];
  int n_rounds = (k_tiles + WARPS_K - 1) / WARPS_K;
  for (int half = 0; half < 2; half++) {{
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int r = 0; r < n_rounds; r++) {{
      int kt;
      kt = r * WARPS_K + wid;
      float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
      if (kt < k_tiles) {{
      int nb = {nbase};
      int aw0, aw1, aw2, aw3;
      float sw_g, sw_g8;
      if (hrow >= 0) {{
        const char* wb = w_h8
          + (size_t)hrow * (size_t)w_h8_stride_e
          + (size_t)nb * (size_t)K
          + (size_t)kt * 32ull;
        const unsigned char* sb = s_h8
          + (size_t)hrow * (size_t)s_stride_e
          + (size_t)kt;
        aw0 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t));
        aw1 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t));
        aw2 = *(const int*)(wb + (size_t)g * (size_t)K + (size_t)(4 * t + 16));
        aw3 = *(const int*)(wb + (size_t)(g + 8) * (size_t)K + (size_t)(4 * t + 16));
        sw_g = e8m0_to_f32(sb[(size_t)(nb + (int)g) * (size_t)(K / 32)]);
        sw_g8 = e8m0_to_f32(sb[(size_t)(nb + (int)g + 8) * (size_t)(K / 32)]);
      }} else {{
        const unsigned char* pkb = w_p4
          + (size_t)e * (size_t)w_p4_stride_e
          + (size_t)nb * (size_t)(K / 2)
          + (size_t)kt * 16ull;
        const unsigned char* sb = s_p4
          + (size_t)e * (size_t)s_stride_e
          + (size_t)kt;
        aw0 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
        aw1 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 1)]);
        aw2 = e2m1_pair_b32(pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)g * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
        aw3 = e2m1_pair_b32(pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 8)], pkb[(size_t)(g + 8) * (size_t)(K / 2) + (size_t)(2 * t + 9)]);
        sw_g = e8m0_to_f32(sb[(size_t)(nb + (int)g) * (size_t)(K / 32)]);
        sw_g8 = e8m0_to_f32(sb[(size_t)(nb + (int)g + 8) * (size_t)(K / 32)]);
      }}
        int x00 = load_act_s8b32_tok(a0, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x01 = load_act_s8b32_tok(a0, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int x10 = load_act_s8b32_tok(a1, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x11 = load_act_s8b32_tok(a1, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int x20 = load_act_s8b32_tok(a2, row_token, row, (int)g, 4 * (int)t, m_tile, K, kt);
        int x21 = load_act_s8b32_tok(a2, row_token, row, (int)g, 4 * (int)t + 16, m_tile, K, kt);
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
        int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
        mma_s8k32(&d0, &d1, &d2, &d3, aw0, aw1, aw2, aw3, x00, x01);
        mma_s8k32(&e0, &e1, &e2, &e3, aw0, aw1, aw2, aw3, x10, x11);
        mma_s8k32(&f0, &f1, &f2, &f3, aw0, aw1, aw2, aw3, x20, x21);
        int D0 = d0 + 256 * e0 + 65536 * f0;
        int D1 = d1 + 256 * e1 + 65536 * f1;
        int D2 = d2 + 256 * e2 + 65536 * f2;
        int D3 = d3 + 256 * e3 + 65536 * f3;
        float sa0 = 0.f, sa1 = 0.f;
        if (t0 < m_tile) {{
          int src0 = row_token ? row_token[row + t0] : (row + t0);
          sa0 = sA[(size_t)src0 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        if (t1 < m_tile) {{
          int src1 = row_token ? row_token[row + t1] : (row + t1);
          sa1 = sA[(size_t)src1 * (size_t)sag + (size_t)(kt >> 2)];
        }}
        p0 = ((float)D0) * (1.f / 1024.f) * sw_g * sa0;
        p1 = ((float)D1) * (1.f / 1024.f) * sw_g * sa1;
        p2 = ((float)D2) * (1.f / 1024.f) * sw_g8 * sa0;
        p3 = ((float)D3) * (1.f / 1024.f) * sw_g8 * sa1;
      }}
      red[wid][lane][0] = p0; red[wid][lane][1] = p1; red[wid][lane][2] = p2; red[wid][lane][3] = p3;
      __syncthreads();
      if (wid == 0) {{
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
    if (wid == 0) {{
      {store}
      if (t0 < m_tile && col0 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc0);
      if (t1 < m_tile && col0 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col0] = (unsigned short)f32_to_bf16_bits(acc1);
      if (t0 < m_tile && col1 < N) out[(size_t)(row + t0) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc2);
      if (t1 < m_tile && col1 < N) out[(size_t)(row + t1) * (size_t)N + (size_t)col1] = (unsigned short)f32_to_bf16_bits(acc3);
    }}
    __syncthreads();
  }}
}}
"""


def mxint_ksplit_grouped_masked(
    kind: str,
    a0, a1, a2, sA, row_token,
    w_p4, s_p4,
    w_h8, s_h8, hot_ids,
    offsets, expert_index, m_start, n_tiles, tile_rows,
    n_out, inter, k, warps=8, cap=None, A=None,
):
    """Masked grouped IMMA (hot8 subset + packed4 fallback), single launch."""
    global _LAUNCHES
    if A is None:
        A = row_token.numel() if row_token is not None else a0.shape[0]
    out = torch.zeros(A, n_out, dtype=torch.bfloat16, device=a0.device)
    t_cap = cap if cap is not None else expert_index.numel()
    k_tiles = k // 32
    if kind == "gu":
        n_tiles_y = _cdiv(inter, 16)
    else:
        n_tiles_y = _cdiv(n_out, 32)
    e_p4 = w_p4.shape[0]
    w_p4_stride = w_p4.shape[1] * w_p4.shape[2]
    w_h8_stride = w_h8.shape[1] * w_h8.shape[2]
    s_stride = s_p4.shape[1] * s_p4.shape[2]
    entry = f"mxint_{kind}_masked_w{warps}"
    fn = _get_fn(entry, _imma_grouped_masked_src(kind, warps), entry)
    _LAUNCHES += 1
    launch(
        fn, (t_cap, n_tiles_y), 32 * warps,
        [a0, a1, a2, sA.contiguous().to(torch.float32), row_token,
         w_p4.view(torch.uint8) if w_p4.dtype != torch.uint8 else w_p4,
         s_p4.view(torch.uint8) if s_p4.dtype != torch.uint8 else s_p4,
         w_h8, s_h8.view(torch.uint8) if s_h8.dtype != torch.uint8 else s_h8,
         hot_ids.to(torch.int32),
         offsets.to(torch.int32), expert_index.to(torch.int32),
         m_start.to(torch.int32), tile_rows.to(torch.int32),
         n_tiles.to(torch.int32),
         out.view(torch.uint16),
         A, n_out, inter, k, k_tiles,
         w_p4_stride, w_h8_stride, s_stride],
        shared=warps * 32 * 4 * 4,
    )
    return out


def mxint_imma_gemm(x, packed, scale_w, n_out, prequant=None, xint=None,
                    qpa_k=8, ordered=True, prepacked=None,
                    weight_repr: str = "hot8") -> torch.Tensor:
    """x [M,K] bf16, packed [N,K/2], scale [N,K/32] -> [M,N] bf16. M>=4."""
    global _LAUNCHES, CHOSEN_MXINT_K
    m, k = x.shape
    n = packed.shape[0]
    if xint is None:
        if prequant is None:
            x_fp8, sA = act_quant_ref(x, ACT_BLOCK)
        else:
            x_fp8, sA = prequant
            sA = sA.contiguous().to(torch.float32)
        a0, a1, a2 = decode_activation_int((x_fp8, sA))
    else:
        a0, a1, a2, sA = xint
    tag = "ordered" if ordered else "reordered"
    if weight_repr == "hot8":
        wq = prepacked if prepacked is not None else qint_encode_w(packed)
        entry = f"mxint_imma_ksplit_{tag}_w{qpa_k}"
        src = _imma_direct_src(qpa_k, ordered, packed=False)
    elif weight_repr == "packed4":
        wq = prepacked if prepacked is not None else packed
        entry = f"mxint_imma_ksplit_{tag}_packed4_w{qpa_k}"
        src = _imma_direct_src(qpa_k, ordered, packed=True)
    else:
        raise ValueError(f"weight_repr must be hot8|packed4, got {weight_repr}")
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    n_tiles = _cdiv(n, 16)
    k_tiles = k // 32
    CHOSEN_MXINT_K = qpa_k
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (n_tiles,), 32 * qpa_k,
        [a0, a1, a2, sA.contiguous().to(torch.float32), wq,
         scale_w.view(torch.uint8) if scale_w.dtype != torch.uint8 else scale_w,
         out.view(torch.uint16), m, n, k, n_tiles, k_tiles],
        shared=qpa_k * 32 * 4 * 4,
    )
    return out


def mxint_ksplit_grouped(
    kind: str,
    a0, a1, a2, sA, row_token,
    w, scale,
    offsets, expert_index, m_start, n_tiles, tile_rows,
    n_out, inter, k, warps=8, ordered=True, cap=None, A=None,
    weight_repr: str = "hot8",
) -> torch.Tensor:
    """Grouped IMMA ordered-KSplit on the s8 (hot8) or packed uint8 (packed4) bank."""
    global _LAUNCHES
    if A is None:
        A = row_token.numel() if row_token is not None else a0.shape[0]
    out = torch.zeros(A, n_out, dtype=torch.bfloat16, device=a0.device)
    t_cap = cap if cap is not None else expert_index.numel()
    k_tiles = k // 32
    if kind == "gu":
        n_tiles_y = _cdiv(inter, 16)
    else:
        n_tiles_y = _cdiv(n_out, 32)
    w_stride_e = w.shape[1] * w.shape[2] if w.dim() == 3 else 0
    s_stride_e = scale.shape[1] * scale.shape[2] if scale.dim() == 3 else 0
    wv = w if w.dim() == 3 else w.unsqueeze(0)
    sv = scale if scale.dim() == 3 else scale.unsqueeze(0)
    tag = "ordered" if ordered else "reordered"
    if weight_repr == "hot8":
        entry = f"mxint_{kind}_ksplit_{tag}_w{warps}"
        src = _imma_grouped_src(kind, warps, ordered, packed=False)
    elif weight_repr == "packed4":
        entry = f"mxint_{kind}_ksplit_{tag}_packed4_w{warps}"
        src = _imma_grouped_src(kind, warps, ordered, packed=True)
    else:
        raise ValueError(f"weight_repr must be hot8|packed4, got {weight_repr}")
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (t_cap, n_tiles_y), 32 * warps,
        [a0, a1, a2, sA.contiguous().to(torch.float32), row_token,
         wv, sv,
         offsets.to(torch.int32), expert_index.to(torch.int32),
         m_start.to(torch.int32), tile_rows.to(torch.int32), n_tiles.to(torch.int32),
         out.view(torch.uint16),
         A, n_out, inter, k, k_tiles, n_tiles_y, n_tiles_y,
         w_stride_e, s_stride_e],
        shared=warps * 32 * 4 * 4,
    )
    return out


# ---------------------------------------------------------------------------
# HexaExpert (K1, one token, 6 warps = 6 assignments)
# ---------------------------------------------------------------------------

def _hexa_src(packed: bool) -> str:
    if packed:
        name = "mxint_hexa_dp4a_packed4"
        wdecl = "    const unsigned char* w,"
        wb = "  const unsigned char* wb = w + (size_t)e * (size_t)w_stride_e + (size_t)n * (size_t)(K / 2);"
        wload = "      int wv = e2m1_pair_b32(wb[g * 16 + i * 2], wb[g * 16 + i * 2 + 1]);"
        pre = _PACKED4_HELP
    else:
        name = "mxint_hexa_dp4a"
        wdecl = "    const char* w,"
        wb = "  const char* wb = w + (size_t)e * (size_t)w_stride_e + (size_t)n * (size_t)K;"
        wload = "      int wv = *(const int*)(wb + g * 32 + i * 4);"
        pre = ""
    return DEVICE_HELPERS + pre + f"""
extern "C" __global__ void {name}(
    const char* a0, const char* a1, const char* a2,
    const float* sA,
    const int* expert_ids,
{wdecl}
    const unsigned char* scale,
    unsigned short* out,
    int N, int K, int n_groups, int act_row_mode,
    long long w_stride_e, long long s_stride_e
) {{
  int nt = blockIdx.x;
  int wslot = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int n = nt * 32 + lane;
  if (n >= N) return;
  int e = expert_ids[wslot];
{wb}
  const unsigned char* sc = scale + (size_t)e * (size_t)s_stride_e + (size_t)n * (size_t)n_groups;
  int src = act_row_mode ? wslot : 0;
  const char* r0 = a0 + (size_t)src * (size_t)K;
  const char* r1 = a1 + (size_t)src * (size_t)K;
  const char* r2 = a2 + (size_t)src * (size_t)K;
  const float* sa_row = sA + (size_t)src * (size_t)(K / 128);
  float acc = 0.f;
  for (int g = 0; g < n_groups; g++) {{
    int D0 = 0, D1 = 0, D2 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {{
      int x0 = *(const int*)(r0 + g * 32 + i * 4);
      int x1 = *(const int*)(r1 + g * 32 + i * 4);
      int x2 = *(const int*)(r2 + g * 32 + i * 4);
{wload}
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D0) : "r"(x0), "r"(wv));
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D1) : "r"(x1), "r"(wv));
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(D2) : "r"(x2), "r"(wv));
    }}
    float sw = e8m0_to_f32(sc[g]);
    float sa = sa_row[g >> 2];
    acc += ((float)(D0 + 256 * D1 + 65536 * D2)) * (1.f / 1024.f) * sw * sa;
  }}
  out[(size_t)wslot * (size_t)N + (size_t)n] = (unsigned short)f32_to_bf16_bits(acc);
}}
"""


def mxint_hexa_dp4a(a0, a1, a2, sA, expert_ids, w, scale, n_out, k,
                    act_row_mode: int = 0, weight_repr: str = "hot8") -> torch.Tensor:
    """6 assignment slots of one token -> [6, N] bf16. Bank w [E,N,K] s8 or [E,N,K/2] u8.

    act_row_mode=0: all warps read activation row 0 (GU of a single token).
    act_row_mode=1: warp w reads activation row w (packed Down).
    """
    global _LAUNCHES
    out = torch.empty(6, n_out, dtype=torch.bfloat16, device=a0.device)
    n_groups = k // 32
    w_stride_e = w.shape[1] * w.shape[2] if w.dim() == 3 else 0
    s_stride_e = scale.shape[1] * scale.shape[2] if scale.dim() == 3 else 0
    wv = w if w.dim() == 3 else w.unsqueeze(0)
    sv = scale if scale.dim() == 3 else scale.unsqueeze(0)
    if weight_repr == "hot8":
        entry = "mxint_hexa_dp4a"
        src = _hexa_src(False)
    elif weight_repr == "packed4":
        entry = "mxint_hexa_dp4a_packed4"
        src = _hexa_src(True)
    else:
        raise ValueError(f"weight_repr must be hot8|packed4, got {weight_repr}")
    fn = _get_fn(entry, src, entry)
    _LAUNCHES += 1
    launch(
        fn, (_cdiv(n_out, 32),), 6 * 32,
        [a0, a1, a2, sA.contiguous().to(torch.float32),
         expert_ids.to(torch.int32).contiguous(), wv, sv,
         out.view(torch.uint16), n_out, k, n_groups, int(act_row_mode),
         w_stride_e, s_stride_e],
    )
    return out

_LUT32_SRC = DEVICE_HELPERS + r"""
extern "C" __global__ void mxint_lut32_gemm(
    const float* x_f32,
    const unsigned char* packed,
    const unsigned char* scale,
    unsigned short* out,
    int M, int N, int K, int n_groups
) {
  // One CTA per (m, N-tile of 32). Rebuild 32x16 LUT per K32 group; reuse across the 32 N.
  __shared__ float lut[32][16];
  int m = blockIdx.x;
  int nt = blockIdx.y;
  int lane = threadIdx.x;
  int n = nt * 32 + lane;
  float acc = 0.f;
  int valid = n < N;
  for (int g = 0; g < n_groups; g++) {
    for (int i = lane; i < 512; i += 32) {
      int kk = i >> 4, c = i & 15;
      lut[kk][c] = x_f32[(size_t)m * (size_t)K + (size_t)(g * 32 + kk)] * e2m1_f32_lut_reg(c);
    }
    __syncthreads();
    if (valid) {
      float sw = e8m0_to_f32(scale[(size_t)n * (size_t)n_groups + (size_t)g]);
      float dot = 0.f;
      #pragma unroll
      for (int kk = 0; kk < 32; kk++) {
        unsigned char b = packed[(size_t)n * (size_t)(K / 2) + (size_t)(g * 16 + (kk >> 1))];
        int code = (kk & 1) ? (b >> 4) : (b & 0x0f);
        dot += lut[kk][code];
      }
      acc += dot * sw;
    }
    __syncthreads();
  }
  if (valid) out[(size_t)m * (size_t)N + (size_t)n] = (unsigned short)f32_to_bf16_bits(acc);
}
"""


def _get_fn(name: str, src: str, entry: str):
    global PTX_HAS_M16N8K32, PTX_HAS_DP4A
    if entry not in _FN_CACHE:
        ptx = compile_cuda(src, name)
        PTX_HAS_M16N8K32 = PTX_HAS_M16N8K32 or ("m16n8k32" in ptx.decode("utf-8", "replace"))
        PTX_HAS_DP4A = PTX_HAS_DP4A or ("dp4a.s32.s32" in ptx.decode("utf-8", "replace"))
        _FN_CACHE[entry] = load_kernel(ptx, entry)
    return _FN_CACHE[entry]


# ---------------------------------------------------------------------------
# 2.9L Track B: exact epilogue fusion (SwiGLU+requant+qA, routed reduction)
# ---------------------------------------------------------------------------

_EPILOGUE_SRC = DEVICE_HELPERS + r"""
__device__ __forceinline__ float bf16bits_to_f32(unsigned short h) {
  return bits_f32(((unsigned)h) << 16);
}
__device__ __forceinline__ unsigned char f32_to_e4m3fn_bits(float x) {
  // Round-to-nearest-even fp32 -> e4m3fn byte (finite format, NaN-only oob).
  unsigned ux = f32_bits(x);
  unsigned sign = (ux >> 31) & 1u;
  unsigned mag = ux & 0x7FFFFFFFu;
  if (mag > 0x7F800000u) return (unsigned char)((sign << 7) | 0x7Fu); // NaN
  if (mag >= 0x43F00000u) return (unsigned char)((sign << 7) | 0x7Fu); // >=480
  unsigned char result;
  if (mag < (121u << 23)) {
    // |x| < 2^-6: subnormal lattice 2^-9; RNE to lattice multiples.
    float a = bits_f32(mag);
    float y = a * 512.0f;
    float fl = floorf(y);
    float r = y - fl;
    unsigned m = (unsigned)fl;
    if (r > 0.5f || (r == 0.5f && (m & 1u))) m += 1u;
    result = (unsigned char)m; // 0..8; 8 -> 0x08 = 2^-6 normal transition
  } else {
    unsigned E = (mag >> 23) - 127u + 7u;
    unsigned frac = mag & 0x7FFFFFu;
    unsigned m3 = frac >> 20;
    unsigned rem = frac & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) {
      m3 += 1u;
      if (m3 == 8u) { m3 = 0u; E += 1u; }
    }
    if (E >= 16u) return (unsigned char)((sign << 7) | 0x7Fu);
    result = (unsigned char)((E << 3) | m3);
  }
  return (unsigned char)((sign << 7) | result);
}
extern "C" __global__ void mxint_probe_e4m3(const float* x, unsigned char* out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = f32_to_e4m3fn_bits(x[i]);
}
extern "C" __global__ void mxint_probe_silu(const float* x, float* out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = x[i] / (1.f + expf(-x[i]));
}
// Track B kernel 1: GU -> clamp -> SiLU -> route weight -> source BF16 cast ->
// down act_quant (E8M0 per 128) -> e4m3 -> qA 3-limb split. One pass.
// Requires inter % 128 == 0 (wrapper enforces).
extern "C" __global__ void mxint_swiglu_quant_down(
    const unsigned short* gu,   // [A, 2*inter] bf16: gate | up
    const float* w_a,           // [A] router weights
    char* a0, char* a1, char* a2, // [A, inter] qA limb planes
    float* sA_out,              // [A, inter/128]
    int A, int inter, float limit
) {
  int row = blockIdx.x;
  int blk = blockIdx.y;
  int tid = threadIdx.x;
  int j = blk * 128 + tid;   // inter % 128 == 0 -> always valid
  int nb = inter / 128;
  float gate = bf16bits_to_f32(gu[(size_t)row * (size_t)(2 * inter) + (size_t)j]);
  float up = bf16bits_to_f32(gu[(size_t)row * (size_t)(2 * inter) + (size_t)(inter + j)]);
  if (limit > 0.f) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
  }
  float z = (gate / (1.f + expf(-gate))) * up;
  z = z * w_a[row];
  unsigned zd_b = f32_to_bf16_bits(z);          // source-visible BF16 boundary
  float zd = bf16bits_to_f32((unsigned short)zd_b);
  float v = fabsf(zd);
  __shared__ float wmax[4];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  if ((tid & 31) == 0) wmax[tid >> 5] = v;
  __syncthreads();
  float amax = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
  // exact act_quant_ref scale math
  float a = fmaxf(amax, 1e-4f);
  unsigned vb = f32_bits(a * (1.0f / 448.0f));
  int pow_exp = (int)((vb >> 23) & 0xFFu) - 127 + (int)((vb & 0x7FFFFFu) != 0u);
  float sA = bits_f32((unsigned)(pow_exp + 127) << 23);
  float y = zd / sA;
  y = fmaxf(fminf(y, 448.0f), -448.0f);
  unsigned char b = f32_to_e4m3fn_bits(y);
  int q = e4m3_qint(b);
  int p0, p1, p2;
  split_qA_s8(q, &p0, &p1, &p2);
  a0[(size_t)row * (size_t)inter + (size_t)j] = (char)p0;
  a1[(size_t)row * (size_t)inter + (size_t)j] = (char)p1;
  a2[(size_t)row * (size_t)inter + (size_t)j] = (char)p2;
  if (tid == 0) sA_out[(size_t)row * (size_t)nb + (size_t)blk] = sA;
}
// Track B kernel 2: six routed partials -> ascending expert-id order ->
// FP32 sequential accumulate -> source output cast.
extern "C" __global__ void mxint_routed_reduce(
    const unsigned short* yd,   // [A, H] bf16 packed-order partials
    const int* slot_rows,       // [k_tok, top_k] yd row per (token, slot)
    const int* order,           // [k_tok, top_k] ascending-expert-id slot seq
    unsigned short* out,        // [k_tok, H] bf16
    int k_tok, int top_k, int H
) {
  int t = blockIdx.x;
  int h = (int)(blockIdx.y * blockDim.x + threadIdx.x);
  if (t >= k_tok || h >= H) return;
  float acc = 0.f;
  for (int s = 0; s < top_k; s++) {
    int slot = order[t * top_k + s];
    int row = slot_rows[t * top_k + slot];
    acc += bf16bits_to_f32(yd[(size_t)row * (size_t)H + (size_t)h]);
  }
  out[(size_t)t * (size_t)H + (size_t)h] = (unsigned short)f32_to_bf16_bits(acc);
}
"""


def mxint_probe_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Probe: run the device e4m3fn encoder over fp32 x -> uint8."""
    xf = x.contiguous().view(-1)
    out = torch.empty(xf.numel(), dtype=torch.uint8, device=xf.device)
    fn = _get_fn("mxint_probe_e4m3", _EPILOGUE_SRC, "mxint_probe_e4m3")
    launch(fn, (_cdiv(xf.numel(), 256),), 256, [xf, out, xf.numel()])
    return out.view(x.shape)


def mxint_probe_silu(x: torch.Tensor) -> torch.Tensor:
    """Probe: device silu formulation x/(1+expf(-x)) over fp32 x."""
    xf = x.contiguous().view(-1)
    out = torch.empty(xf.numel(), dtype=torch.float32, device=xf.device)
    fn = _get_fn("mxint_probe_silu", _EPILOGUE_SRC, "mxint_probe_silu")
    launch(fn, (_cdiv(xf.numel(), 256),), 256, [xf, out, xf.numel()])
    return out.view(x.shape)


def mxint_swiglu_quant_down(gu, w_a, inter: int, swiglu_limit: float):
    """GU bf16 [A, 2*inter] + router weights [A] -> (a0,a1,a2,sA) down inputs.

    One fused source-exact pass: FP32 gate/up clamps, SiLU, route weight,
    source BF16 cast, per-128-block E8M0 act_quant, e4m3 quantize, qA split.
    """
    global _LAUNCHES
    A = gu.shape[0]
    if gu.shape[1] != 2 * inter or inter % 128 != 0 or swiglu_limit <= 0:
        raise ValueError("mxint_swiglu_quant_down requires [A,2*inter], inter%128==0, limit>0")
    a0 = torch.empty(A, inter, dtype=torch.int8, device=gu.device)
    a1 = torch.empty_like(a0)
    a2 = torch.empty_like(a0)
    sA = torch.empty(A, inter // 128, dtype=torch.float32, device=gu.device)
    fn = _get_fn("mxint_swiglu_quant_down", _EPILOGUE_SRC, "mxint_swiglu_quant_down")
    _LAUNCHES += 1
    launch(
        fn, (A, inter // 128), 128,
        [gu.contiguous().view(torch.uint16), w_a.contiguous().to(torch.float32),
         a0, a1, a2, sA, A, inter, float(swiglu_limit)],
    )
    return a0, a1, a2, sA


def mxint_routed_reduce(yd, slot_rows, order, h: int):
    """Packed-order DOWN partials bf16 [A,H] -> final bf16 [k_tok,H].

    Sums each token's top_k partials in ascending expert-id order (FP32
    sequential, matching the torch reference) and applies the source cast.
    """
    global _LAUNCHES
    k_tok, top_k = slot_rows.shape
    out = torch.empty(k_tok, h, dtype=torch.bfloat16, device=yd.device)
    fn = _get_fn("mxint_routed_reduce", _EPILOGUE_SRC, "mxint_routed_reduce")
    _LAUNCHES += 1
    launch(
        fn, (k_tok, _cdiv(h, 256)), 256,
        [yd.contiguous().view(torch.uint16),
         slot_rows.contiguous().to(torch.int32),
         order.contiguous().to(torch.int32),
         out.view(torch.uint16), k_tok, top_k, h],
    )
    return out


# ---------------------------------------------------------------------------
# Fused MoE (Step 5)
# ---------------------------------------------------------------------------
def fused_moe_mxint(
    x, expert_ids, expert_weights,
    packed_gate_up=None, scale_gate_up=None,
    packed_down=None, scale_down=None,
    bank=None,
    crossover: int = 0,   # all-IMMA: 2.5-2.7x faster at K>=32 M~4+ and
                           # bitwise-exact vs QPA (GROUPED_IMMA_TILE_BUG fix,
                           # tests/test_grouped_imma_parity.py). c3 kept for
                           # small-K/low-M regimes (K8-mid validation point).
    imma_k: int = 8,
    swiglu_limit: float = 10.0,
    stage_times: dict | None = None,
    weight_repr: str = "hot8",
    epilogue: str = "torch",
    gu_bank=None, down_bank=None, masked_bank: "MXINTMaskedBank | None" = None,
) -> torch.Tensor:
    """MXINT fused MoE: DP4A (M<=crossover) + IMMA (M>crossover) + Hexa (K1).

    gu_bank/down_bank: matrix-selective mixed mode — the gu and down stages
    are already separate kernel launches, so reading them from different
    banks adds ZERO launches (unlike the rejected two-call expert split).
    Each partial bank supplies only its stage's matrices; dispatch derives
    hot8/packed4 per stage from the bank type.
    """
    global _LAUNCHES, HYBRID
    _LAUNCHES = 0
    HYBRID = "dp4a_imma"
    from mosaicvm.kernels.moe_layout import layout_assignments
    from mosaicvm.kernels.w4a16_grouped import emit_tiles

    if weight_repr not in ("hot8", "packed4", "hybrid"):
        raise ValueError(f"weight_repr must be hot8|packed4|hybrid, got {weight_repr}")
    if bank is None and gu_bank is None and down_bank is None and masked_bank is None:
        if packed_gate_up is None or packed_down is None:
            raise TypeError("fused_moe_mxint requires bank= or packed_gate_up/packed_down")
        if weight_repr == "hot8":
            bank = prepack_expert_bank_mxint(
                packed_gate_up, scale_gate_up, packed_d=packed_down, scale_d=scale_down)
        elif weight_repr == "packed4":
            bank = prepack_expert_bank_mxint_packed4(
                packed_gate_up, scale_gate_up, packed_d=packed_down, scale_d=scale_down)
        else:
            bank = MXINTHybridBank(
                hot8=prepack_expert_bank_mxint(
                    packed_gate_up, scale_gate_up, packed_d=packed_down, scale_d=scale_down),
                packed4=prepack_expert_bank_mxint_packed4(
                    packed_gate_up, scale_gate_up, packed_d=packed_down, scale_d=scale_down),
                hot_mask=None,
            )
    masked_mode = masked_bank is not None
    mixed_mode = gu_bank is not None or down_bank is not None
    if not mixed_mode and not masked_mode:
        if weight_repr == "hot8" and not isinstance(bank, MXINTExpertBank):
            raise TypeError("weight_repr='hot8' requires an MXINTExpertBank")
        if weight_repr == "packed4" and not isinstance(bank, MXINTPacked4Bank):
            raise TypeError("weight_repr='packed4' requires an MXINTPacked4Bank")
        if weight_repr == "hybrid" and not isinstance(bank, MXINTHybridBank):
            raise TypeError("weight_repr='hybrid' requires an MXINTHybridBank")

    evs: dict[str, torch.cuda.Event] = {}

    def _mark(name: str) -> None:
        if stage_times is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            evs[name] = ev

    _mark("start")

    imma_repr_by: dict[str, str] = {}
    dp_repr_by: dict[str, str] = {}
    if gu_bank is not None or down_bank is not None:
        # matrix-selective mixed mode (per-stage bank; zero extra launches)
        g_b = gu_bank if gu_bank is not None else bank
        d_b = down_bank if down_bank is not None else bank
        if isinstance(g_b, MXINTPacked4Bank):
            imma_w_gu, imma_s_gu = g_b.p_gu, g_b.s_gu
            imma_repr_by["gu"] = "packed4"
        else:
            imma_w_gu, imma_s_gu = g_b.w_gu, g_b.s_gu
            imma_repr_by["gu"] = "hot8"
        if isinstance(d_b, MXINTPacked4Bank):
            imma_w_d, imma_s_d = d_b.p_d, d_b.s_d
            imma_repr_by["down"] = "packed4"
        else:
            imma_w_d, imma_s_d = d_b.w_d, d_b.s_d
            imma_repr_by["down"] = "hot8"
        dp_w_gu, dp_s_gu = imma_w_gu, imma_s_gu
        dp_w_d, dp_s_d = imma_w_d, imma_s_d
        dp_repr_by = dict(imma_repr_by)
        weight_repr = "mixed"
        n_experts = imma_w_gu.shape[0]
    elif masked_mode:
        n_experts = masked_bank.p4.p_gu.shape[0]
        imma_w_gu, imma_s_gu = masked_bank.p4.p_gu, masked_bank.p4.s_gu
        dp_w_gu, dp_s_gu = imma_w_gu, imma_s_gu
        imma_w_d, imma_s_d = masked_bank.p4.p_d, masked_bank.p4.s_d
        dp_w_d, dp_s_d = imma_w_d, imma_s_d
        imma_repr_by = {"gu": "packed4", "down": "packed4"}
        dp_repr_by = dict(imma_repr_by)
    elif weight_repr == "hot8":
        imma_w_gu, imma_s_gu, imma_w_d, imma_s_d = bank.w_gu, bank.s_gu, bank.w_d, bank.s_d
        dp_w_gu, dp_s_gu, dp_w_d, dp_s_d = bank.w_gu, bank.s_gu, bank.w_d, bank.s_d
        imma_repr_by = {"gu": "hot8", "down": "hot8"}
        dp_repr_by = dict(imma_repr_by)
        n_experts = bank.w_gu.shape[0]
    elif weight_repr == "packed4":
        imma_w_gu, imma_s_gu, imma_w_d, imma_s_d = bank.p_gu, bank.s_gu, bank.p_d, bank.s_d
        dp_w_gu, dp_s_gu, dp_w_d, dp_s_d = bank.p_gu, bank.s_gu, bank.p_d, bank.s_d
        imma_repr_by = {"gu": "packed4", "down": "packed4"}
        dp_repr_by = dict(imma_repr_by)
        n_experts = bank.p_gu.shape[0]
    else:  # hybrid
        imma_w_gu, imma_s_gu, imma_w_d, imma_s_d = (
            bank.hot8.w_gu, bank.hot8.s_gu, bank.hot8.w_d, bank.hot8.s_d)
        dp_w_gu, dp_s_gu, dp_w_d, dp_s_d = (
            bank.packed4.p_gu, bank.packed4.s_gu, bank.packed4.p_d, bank.packed4.s_d)
        imma_repr_by = {"gu": "hot8", "down": "hot8"}
        dp_repr_by = {"gu": "packed4", "down": "packed4"}
        n_experts = bank.hot8.w_gu.shape[0]

    h = x.shape[1]
    if weight_repr == "hybrid":
        inter = bank.hot8.n_out_gu // 2
    elif weight_repr == "mixed":
        inter = (gu_bank if gu_bank is not None else bank).n_out_gu // 2
    elif masked_mode:
        inter = masked_bank.p4.n_out_gu // 2
    else:
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

    _sorted, perm, offsets, counts = layout_assignments(flat_ids, n_experts)
    del _sorted
    if weight_repr == "hybrid" and hasattr(bank, "hot_mask"):
        bank.hot_mask = (counts > crossover).to(torch.uint8)
    a = token_of.numel()
    zeros = torch.zeros_like(counts)
    counts_x = torch.where(counts <= crossover, counts, zeros)
    counts_q = torch.where(counts > crossover, counts, zeros)
    cap = max(n_experts, a)
    ex, mx, nx, rx = emit_tiles(counts_x, block_m=1, max_tiles=cap)
    eq, mq, nq, rq = emit_tiles(counts_q, block_m=8, max_tiles=cap)
    _mark("layout")

    # quantize + int-split token activations once
    xf8_tok, sA_tok = act_quant_ref(x, ACT_BLOCK)
    _mark("act_quant")
    a0t, a1t, a2t = decode_activation_int((xf8_tok, sA_tok))
    _mark("decode_int")
    row_token = token_of[perm].to(torch.int32)
    w_a = weights.reshape(-1)[perm]

    hexa_used = (k_tok == 1 and top_k == 6 and weight_repr != "mixed"
                and not masked_mode)
    ids_packed = flat_ids[perm].to(torch.int32)

    def _two_queues(kind, a0, a1, a2, sA, row_tok,
                    imma_w, imma_sc, dp_w, dp_sc, n_out, k_dim, inter_dim):
        global HYBRID, _HEXA_LAUNCHES
        out = torch.zeros(a, n_out, dtype=torch.bfloat16, device=x.device)
        if hexa_used:
            mode = 0 if kind == "gu" else 1
            out6 = mxint_hexa_dp4a(a0, a1, a2, sA, ids_packed, dp_w, dp_sc,
                                    n_out, k_dim, act_row_mode=mode,
                                    weight_repr=dp_repr_by[kind])
            _HEXA_LAUNCHES += 1
            return out6.view(a, n_out)
        if masked_mode:
            mb = masked_bank
            if kind == "gu":
                w_h8, s_h8, hids = mb.h8_gu_w, mb.h8_gu_s, mb.gu_hot_ids
                w_p4, s_p4 = mb.p4.p_gu, mb.p4.s_gu
            else:
                w_h8, s_h8, hids = mb.h8_down_w, mb.h8_down_s, mb.down_hot_ids
                w_p4, s_p4 = mb.p4.p_d, mb.p4.s_d
            return mxint_ksplit_grouped_masked(
                kind, a0, a1, a2, sA, row_tok,
                w_p4, s_p4, w_h8, s_h8, hids,
                offsets, eq, mq, nq, rq, n_out, inter_dim, k_dim,
                warps=imma_k, cap=cap, A=a)
        try:
            oq = mxint_ksplit_grouped(
                kind, a0, a1, a2, sA, row_tok, imma_w, imma_sc,
                offsets, eq, mq, nq, rq, n_out, inter_dim, k_dim,
                warps=imma_k, ordered=True, cap=cap, A=a,
                weight_repr=imma_repr_by[kind])
        except Exception:
            HYBRID = "dp4a_all"
            oq = torch.zeros(a, n_out, dtype=torch.bfloat16, device=x.device)
        out = out + oq
        if HYBRID == "dp4a_all":
            ox = mxint_dp4a_grouped(
                a0, a1, a2, sA, row_tok, dp_w, dp_sc,
                offsets, eq, mq, nq, rq, n_out, k_dim, cap=cap, A=a,
                weight_repr=dp_repr_by[kind])
            out = out + ox
        else:
            ox = mxint_dp4a_grouped(
                a0, a1, a2, sA, row_tok, dp_w, dp_sc,
                offsets, ex, mx, nx, rx, n_out, k_dim, cap=cap, A=a,
                weight_repr=dp_repr_by[kind])
            out = out + ox
        return out

    if weight_repr == "hybrid":
        n_gu = bank.hot8.n_out_gu
    elif weight_repr == "mixed":
        n_gu = (gu_bank if gu_bank is not None else bank).n_out_gu
    elif masked_mode:
        n_gu = masked_bank.p4.n_out_gu
    else:
        n_gu = bank.n_out_gu
    k_h = x.shape[1]
    k_i = inter
    gu = _two_queues("gu", a0t, a1t, a2t, sA_tok, row_token,
                     imma_w_gu, imma_s_gu, dp_w_gu, dp_s_gu, n_gu, k_h, inter)
    _mark("gu")

    use_fused_epi = (epilogue == "fused" and swiglu_limit > 0
                     and x.dtype == torch.bfloat16 and inter % 128 == 0)
    if use_fused_epi:
        # swiglu_sec ~ 0 (fused into one kernel, timed as down_quant_sec)
        _mark("swiglu")
        a0d, a1d, a2d, sA_d = mxint_swiglu_quant_down(gu, w_a, inter, swiglu_limit)
        _mark("down_quant")
    else:
        gate = gu[:, :inter].float()
        up = gu[:, inter:].float()
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        z = torch.nn.functional.silu(gate) * up
        z = z * w_a[:, None]
        zd = z.to(x.dtype)
        _mark("swiglu")
        xf8_d, sA_d = act_quant_ref(zd, ACT_BLOCK)
        a0d, a1d, a2d = decode_activation_int((xf8_d, sA_d))
        _mark("down_quant")
    yd = _two_queues("down", a0d, a1d, a2d, sA_d, None,
                     imma_w_d, imma_s_d, dp_w_d, dp_s_d, h, k_i, inter)
    _mark("down")

    if use_fused_epi:
        order_t = (expert_ids.long().argsort(dim=1) if expert_ids.ndim == 2
                   else torch.zeros(k_tok, 1, dtype=torch.int64, device=x.device))
        row_of_flat = torch.empty(a, dtype=torch.int64, device=x.device)
        row_of_flat[perm] = torch.arange(a, device=x.device)
        slot_rows = row_of_flat.view(k_tok, top_k)
        out = mxint_routed_reduce(yd, slot_rows, order_t, h)
        _mark("reduction")
        if stage_times is not None:
            torch.cuda.synchronize()
            stage_times["layout_sec"] = evs["start"].elapsed_time(evs["layout"]) / 1e3
            stage_times["act_quant_sec"] = evs["layout"].elapsed_time(evs["act_quant"]) / 1e3
            stage_times["decode_int_sec"] = evs["act_quant"].elapsed_time(evs["decode_int"]) / 1e3
            stage_times["gu_sec"] = evs["decode_int"].elapsed_time(evs["gu"]) / 1e3
            stage_times["swiglu_sec"] = evs["gu"].elapsed_time(evs["swiglu"]) / 1e3
            stage_times["down_quant_sec"] = evs["swiglu"].elapsed_time(evs["down_quant"]) / 1e3
            stage_times["down_sec"] = evs["down_quant"].elapsed_time(evs["down"]) / 1e3
            stage_times["reduction_sec"] = evs["down"].elapsed_time(evs["reduction"]) / 1e3
            stage_times["total_sec"] = sum(
                stage_times[k] for k in (
                    "layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
                    "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec"))
        return out
    partials = torch.zeros(k_tok, top_k, h, dtype=torch.float32, device=x.device)
    partials[token_of[perm], slot_of[perm]] = yd.float()
    order = expert_ids.long().argsort(dim=1) if expert_ids.ndim == 2 else torch.zeros(k_tok, 1, dtype=torch.int64, device=x.device)
    out = torch.zeros(k_tok, h, dtype=torch.float32, device=x.device)
    for s in range(top_k):
        slot = order[:, s]
        out = out + partials.gather(1, slot.view(k_tok, 1, 1).expand(k_tok, 1, h)).squeeze(1)
    _mark("reduction")

    if stage_times is not None:
        torch.cuda.synchronize()
        stage_times["layout_sec"] = evs["start"].elapsed_time(evs["layout"]) / 1e3
        stage_times["act_quant_sec"] = evs["layout"].elapsed_time(evs["act_quant"]) / 1e3
        stage_times["decode_int_sec"] = evs["act_quant"].elapsed_time(evs["decode_int"]) / 1e3
        stage_times["gu_sec"] = evs["decode_int"].elapsed_time(evs["gu"]) / 1e3
        stage_times["swiglu_sec"] = evs["gu"].elapsed_time(evs["swiglu"]) / 1e3
        stage_times["down_quant_sec"] = evs["swiglu"].elapsed_time(evs["down_quant"]) / 1e3
        stage_times["down_sec"] = evs["down_quant"].elapsed_time(evs["down"]) / 1e3
        stage_times["reduction_sec"] = evs["down"].elapsed_time(evs["reduction"]) / 1e3
        stage_times["total_sec"] = sum(
            stage_times[k] for k in (
                "layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
                "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec"))
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# MX-LUT32 GEMM (built only when gated in, Step 6)
# ---------------------------------------------------------------------------

def mxint_lut32_gemm(x, packed, scale_w, n_out, prequant=None) -> torch.Tensor:
    """Gated LUT32 backend. Informational only; not an Exact default."""
    global LUT_STATUS, _LAUNCHES
    m, k = x.shape
    n = packed.shape[0]
    if prequant is None:
        x_fp8, sA = act_quant_ref(x, ACT_BLOCK)
    else:
        x_fp8, sA = prequant
    from mosaicvm.kernels.fp4_source import decode_e4m3
    x_u8 = _as_u8(x_fp8)
    xf = decode_e4m3(x_u8) * sA.repeat_interleave(ACT_BLOCK, dim=-1)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    n_groups = k // 32
    fn = _get_fn("mxint_lut32_gemm", _LUT32_SRC, "mxint_lut32_gemm")
    _LAUNCHES += 1
    launch(
        fn, (m, _cdiv(n, 32)), 32,
        [xf,
         packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed,
         scale_w.view(torch.uint8) if scale_w.dtype != torch.uint8 else scale_w,
         out.view(torch.uint16), m, n, k, n_groups],
    )
    LUT_STATUS = "built"
    return out
