"""Phase 2.11C2: source-FP8 core execution + T_core(K) + VRAM census.

Production cannot keep the BF16-expanded core resident (15.57 GB > 12 GB).
This benchmark keeps the FP8 E4M3 source codes + E8M0 block scales resident
(~7.79 GB) and decodes each layer's weights to a BF16 workspace immediately
before use.

Decode exactness: E4M3 values are exactly representable in BF16 (3 mantissa
bits < 7), and E8M0 scales are powers of two, so the decode
    fp8 -> f32 (exact) -> * scale (exact, pow2) -> bf16 (single rounding)
is bitwise-identical to the reference dequant path in core_timing._load_layer.

Variants measured:
  fp8_sync : decode layer L, then compute layer L (serialized).
  fp8_async: decode layer L+1 on a side stream while layer L computes
             (double-buffered BF16 workspaces).

Gate (K56 STEADY): <=115 EXCELLENT / <=130 PASS / 130-160 VIABLE /
160-200 WEAK / >200 FP8_CORE_EXECUTION_MISS. Reference BF16 ceiling
(one-layer-resident, pre-decoded): 102.4 ms.

The compute op sequence mirrors core_timing.py exactly (same ops, same
shapes, attention approximated by softmax over a 1024-token KV cache) so the
FP8-resident numbers are directly comparable to the BF16 ceiling.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import safe_open

from mosaicvm.bench.thermal import snapshot, classify

CORE_PATH = Path(".mosaicvm/core/core.safetensors")
ARTIFACT = Path("artifacts/fp8_core.json")

K_VALUES = (32, 40, 48, 56)
N_LAYERS = 43
HIDDEN = 4096
N_HEADS = 64
HEAD_DIM = 512
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
VOCAB = 129280
CTX = 1024  # short-context KV (matches core_timing)

FP8_TENSORS = (
    "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
    "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
)


# ---------------------------------------------------------------- triton ---
@triton.jit
def _dequant_block(W, S, O, N, K, SK,
                   BN: tl.constexpr, BK: tl.constexpr):
    """One program = one 128x128 quant block -> one scalar scale.

    W: uint8 view of fp8e4m3 codes [N, K]; S: f32 [N//BN, K//BK]; O: bf16.
    SM86 has no fp8 triton pointers, so E4M3 is decoded by exact bit
    assembly (all-integer ops -> bitwise-identical to torch's w.float()).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BN
    k0 = pid_k * BK
    offs_n = n0 + tl.arange(0, BN)
    offs_k = k0 + tl.arange(0, BK)
    ptr = offs_n[:, None] * K + offs_k[None, :]
    b = tl.load(W + ptr).to(tl.int32)

    sign = (b >> 7) << 31
    e = (b >> 3) & 0xF
    m = b & 0x7
    # normal: (1 + m/8) * 2^(e-7)  ->  f32 bits s | (e+120)<<23 | m<<20
    norm_bits = sign | ((e + 120) << 23) | (m << 20)
    # subnormal: m * 2^-9; normalize: p = msb index of m (0..2)
    p = tl.where(m >= 4, 2, tl.where(m >= 2, 1, 0))
    sub_bits = sign | ((118 + p) << 23) | ((m - (1 << p)) << (23 - p))
    bits = tl.where(e == 0, tl.where(m == 0, sign, sub_bits), norm_bits)
    wf = tl.cast(bits, tl.float32, bitcast=True)

    s = tl.load(S + (n0 // BN) * SK + (k0 // BK))
    o = (wf * s.to(tl.float32)).to(tl.bfloat16)
    tl.store(O + ptr, o)


def _e8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    from mosaicvm.kernels.fp4_source import e8m0_to_fp32
    return e8m0_to_fp32(s)


def dequant_kernel(w_fp8: torch.Tensor, s: torch.Tensor, out: torch.Tensor):
    n, k = w_fp8.shape
    s_f32 = _e8m0_to_f32(s)
    if not s_f32.is_cuda:
        s_f32 = s_f32.to(w_fp8.device)
    _dequant_block[(n // 128, k // 128)](w_fp8.view(torch.uint8), s_f32, out,
                                         n, k, s_f32.shape[1],
                                         BN=128, BK=128, num_warps=4)


def decode_layer(fp8: dict, scales: dict, ws: dict):
    for name in FP8_TENSORS:
        dequant_kernel(fp8[name], scales[name], ws[name])


# ------------------------------------------------------------------ load ---
def load_core_resident(f) -> dict:
    """All 43 layers FP8-resident + BF16 small tensors + LM head."""
    core = {"layers": []}
    core["head_w"] = f.get_tensor("head.weight").to(torch.bfloat16).to("cuda")
    core["norm_w"] = f.get_tensor("norm.weight").to("cuda")
    core["hc_head_scale"] = f.get_tensor("hc_head_scale").to("cuda")

    for layer in range(N_LAYERS):
        pre = f"layers.{layer}."
        lp = {"fp8": {}, "scales": {}, "small": {}}
        for name in FP8_TENSORS:
            lp["fp8"][name] = f.get_tensor(pre + name + ".weight").to("cuda")
            lp["scales"][name] = f.get_tensor(pre + name + ".scale").to("cuda")
        lp["small"]["gate_w"] = f.get_tensor(pre + "ffn.gate.weight").to(torch.bfloat16).to("cuda")
        bias_name = pre + "ffn.gate.bias"
        gb = f.get_tensor(bias_name) if bias_name in f.keys() else None
        lp["small"]["gate_bias"] = (gb.to("cuda") if gb is not None
                                    else torch.zeros(256, device="cuda"))
        for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            lp["small"][nm] = f.get_tensor(pre + nm).to("cuda")
        for src, dst in (("attn_norm.weight", "attn_norm"),
                         ("ffn_norm.weight", "ffn_norm"),
                         ("attn.q_norm.weight", "q_norm"),
                         ("attn.kv_norm.weight", "kv_norm"),
                         ("attn.attn_sink", "attn_sink")):
            lp["small"][dst] = f.get_tensor(pre + src).to("cuda")
        core["layers"].append(lp)
    return core


def alloc_workspace(device="cuda") -> dict:
    """BF16 decode workspace for one layer (double-buffer = two of these)."""
    shapes = {
        "attn.wq_a": (1024, 4096), "attn.wq_b": (32768, 1024),
        "attn.wkv": (512, 4096), "attn.wo_a": (8192, 4096),
        "attn.wo_b": (4096, 8192),
        "ffn.shared_experts.w1": (2048, 4096),
        "ffn.shared_experts.w2": (4096, 2048),
        "ffn.shared_experts.w3": (2048, 4096),
    }
    return {n: torch.empty(s, dtype=torch.bfloat16, device=device)
            for n, s in shapes.items()}

# ------------------------------------------------------------ fused GEMM ---
@triton.jit
def _fused_fp8_linear(X, W, S, O, M, N, K, SK,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = x @ W^T with W decoded from fp8 codes in-register.

    Per 128x128 quant block the E8M0 scale is a power of two, so it factors
    exactly out of the k-sum: acc += dot(x, decode(w)) * s is bitwise
    equivalent to accumulating pre-scaled weights. W tile never materializes
    in DRAM as bf16: 1 byte/elem weight traffic (half of BF16-resident).
    BN | 128 and BK = 128 keep each tile inside a single quant block.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x_t = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                      mask=mmask[:, None], other=0.0)
        b = tl.load(W + offs_k[:, None] + offs_n[None, :] * K).to(tl.int32)
        # scale-free E4M3 bit-assembly decode (same as _dequant_block)
        sign = (b >> 7) << 31
        e = (b >> 3) & 0xF
        m = b & 0x7
        norm_bits = sign | ((e + 120) << 23) | (m << 20)
        p = tl.where(m >= 4, 2, tl.where(m >= 2, 1, 0))
        sub_bits = sign | ((118 + p) << 23) | ((m - (1 << p)) << (23 - p))
        bits = tl.where(e == 0, tl.where(m == 0, sign, sub_bits), norm_bits)
        wf = tl.cast(bits, tl.float32, bitcast=True).to(tl.bfloat16)
        s = tl.load(S + (pid_n * BN // 128) * SK + (k0 // 128))
        acc += tl.dot(x_t, wf) * s
    tl.store(O + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mmask[:, None])


_DEC_WS: torch.Tensor | None = None  # shared BF16 decode scratch (m>80 path)
def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, s: torch.Tensor,
               out: torch.Tensor = None) -> torch.Tensor:
    """F.linear(x, decode(w, s)) — GEMM-constant race winner (§57-1).

    Width dispatch (artifacts/gemm_race.json):
      m <= 80 : fused in-GEMM decode, BM=32/BN=64/BK=128/w4/s3
                (per-shape winner at M=32/56/80)
      m > 80  : dequant to a shared BF16 scratch + cuBLAS
                (wins every shape at M=112: 3.42 vs 3.96 ms/layer)
    v5 split-K measured honestly (M-tiling fixed): 3.66-4.75 ms/layer at
    M=80/112 — loses to both; closed (§30 confirmed).
    """
    m, k = x.shape
    n = w_fp8.shape[0]
    s_f32 = _e8m0_to_f32(s)
    if not s_f32.is_cuda:
        s_f32 = s_f32.to(x.device)
    xc = x if x.is_contiguous() else x.contiguous()
    if m > 80:
        global _DEC_WS
        need = n * k
        if _DEC_WS is None or _DEC_WS.numel() < need:
            _DEC_WS = torch.empty(need, dtype=torch.bfloat16, device=x.device)
        ws = _DEC_WS[:need].view(n, k)
        dequant_kernel(w_fp8, s, ws)
        return torch.nn.functional.linear(xc, ws)
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    BN = 64
    grid = ((m + 31) // 32, (n + BN - 1) // BN)
    _fused_fp8_linear[grid](xc, w_fp8.view(torch.uint8), s_f32, out,
                            m, n, k, s_f32.shape[1],
                            BM=32, BN=BN, BK=128, num_warps=4, num_stages=3)
    return out


def compute_layer_fused(x, fp8, scales, small, k, kv_cache):
    """compute_layer with all 8 core GEMMs fused-decoded from FP8."""
    q_lora = fp8_linear(x, fp8["attn.wq_a"], scales["attn.wq_a"])
    q_lora = torch.nn.functional.rms_norm(q_lora, (Q_LORA,), small["q_norm"])
    q = fp8_linear(q_lora, fp8["attn.wq_b"], scales["attn.wq_b"])
    kv = fp8_linear(x, fp8["attn.wkv"], scales["attn.wkv"])
    kv = torch.nn.functional.rms_norm(kv, (HEAD_DIM,), small["kv_norm"])

    q3 = q.view(k, N_HEADS, HEAD_DIM)
    k_exp = kv_cache.expand(kv_cache.shape[0], N_HEADS, HEAD_DIM)
    scores = torch.einsum("khd,chd->kch", q3, k_exp) / (HEAD_DIM ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("kch,chd->khd", attn, k_exp)

    attn_grouped = attn_out.reshape(k, O_GROUPS, 4096).transpose(0, 1).contiguous()
    wo_a_out = torch.empty(O_GROUPS, k, O_LORA, dtype=torch.bfloat16, device=x.device)
    w_u8 = fp8["attn.wo_a"].view(torch.uint8).view(O_GROUPS, O_LORA, 4096)
    s_5 = scales["attn.wo_a"]
    nb = s_5.shape[0] // O_GROUPS
    for g in range(O_GROUPS):
        fp8_linear(attn_grouped[g],
                   fp8["attn.wo_a"].view(O_GROUPS, O_LORA, 4096)[g],
                   s_5.view(O_GROUPS, nb, s_5.shape[1])[g],
                   out=wo_a_out[g])
    wo_a_out = wo_a_out.transpose(0, 1).reshape(k, O_GROUPS * O_LORA).contiguous()
    y = fp8_linear(wo_a_out, fp8["attn.wo_b"], scales["attn.wo_b"])

    g1 = fp8_linear(x, fp8["ffn.shared_experts.w1"],
                    scales["ffn.shared_experts.w1"]).float().clamp(max=10.0)
    u1 = fp8_linear(x, fp8["ffn.shared_experts.w3"],
                    scales["ffn.shared_experts.w3"]).float().clamp(-10.0, 10.0)
    z = (torch.nn.functional.silu(g1) * u1).to(torch.bfloat16)
    y = y + fp8_linear(z, fp8["ffn.shared_experts.w2"],
                       scales["ffn.shared_experts.w2"])

    logits = torch.nn.functional.linear(x.float(), small["gate_w"].float())
    sr = torch.nn.functional.softplus(logits).sqrt() + small["gate_bias"].float()
    torch.topk(sr, 6, dim=-1)

    hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32, device=x.device)
    linf = torch.nn.functional.linear
    mixes = linf(hc_in, small["hc_attn_fn"].float())
    torch.sigmoid(mixes * small["hc_attn_scale"].float().mean()
                  + small["hc_attn_base"].float()) + 1e-6
    mixes = linf(hc_in, small["hc_ffn_fn"].float())
    torch.sigmoid(mixes * small["hc_ffn_scale"].float().mean()
                  + small["hc_ffn_base"].float()) + 1e-6

    torch.nn.functional.rms_norm(x, (HIDDEN,), small["attn_norm"])
    torch.nn.functional.rms_norm(x, (HIDDEN,), small["ffn_norm"])
    return y


# --------------------------------------------------------------- compute ---
def compute_layer(x, ws, small, k, kv_cache):
    """Identical op sequence to core_timing.py per-layer components."""
    lin = torch.nn.functional.linear
    q_lora = lin(x, ws["attn.wq_a"])
    q_lora = torch.nn.functional.rms_norm(q_lora, (Q_LORA,), small["q_norm"])
    q = lin(q_lora, ws["attn.wq_b"])                       # [k, 32768]
    kv = lin(x, ws["attn.wkv"])
    kv = torch.nn.functional.rms_norm(kv, (HEAD_DIM,), small["kv_norm"])

    # attention (DSA approximated: softmax over CTX KV, timing-only)
    q3 = q.view(k, N_HEADS, HEAD_DIM)
    k_exp = kv_cache.expand(kv_cache.shape[0], N_HEADS, HEAD_DIM)
    scores = torch.einsum("khd,chd->kch", q3, k_exp) / (HEAD_DIM ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("kch,chd->khd", attn, k_exp)   # [k, 64, 512]

    wo_a_w = ws["attn.wo_a"].view(O_GROUPS, O_LORA, 4096)
    attn_grouped = attn_out.reshape(k, O_GROUPS, 4096)
    wo_a_out = torch.einsum("kgd,grd->kgr", attn_grouped, wo_a_w)
    wo_a_out = wo_a_out.reshape(k, O_GROUPS * O_LORA)
    y = lin(wo_a_out, ws["attn.wo_b"])                     # [k, 4096]

    # shared expert SwiGLU (source clamp)
    g = lin(x, ws["ffn.shared_experts.w1"]).float().clamp(max=10.0)
    u = lin(x, ws["ffn.shared_experts.w3"]).float().clamp(-10.0, 10.0)
    z = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
    y = y + lin(z, ws["ffn.shared_experts.w2"])

    # router (F32 exact semantics)
    logits = lin(x.float(), small["gate_w"].float())
    s = torch.nn.functional.softplus(logits).sqrt() + small["gate_bias"].float()
    torch.topk(s, 6, dim=-1)

    # HyperConnections mixing (F32)
    hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32, device=x.device)
    mixes = lin(hc_in, small["hc_attn_fn"].float())
    torch.sigmoid(mixes * small["hc_attn_scale"].float().mean()
                  + small["hc_attn_base"].float()) + 1e-6
    mixes = lin(hc_in, small["hc_ffn_fn"].float())
    torch.sigmoid(mixes * small["hc_ffn_scale"].float().mean()
                  + small["hc_ffn_base"].float()) + 1e-6

    # norms
    torch.nn.functional.rms_norm(x, (HIDDEN,), small["attn_norm"])
    torch.nn.functional.rms_norm(x, (HIDDEN,), small["ffn_norm"])
    return y


def run_variant(core, k, mode: str, n_iters: int = 10, warmup: int = 3) -> float:
    """One full 43-layer pass per iteration. Returns median seconds."""
    device = "cuda"
    x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(CTX, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    ws = [alloc_workspace(device), alloc_workspace(device)] if mode != "fused" else None

    dec_stream = torch.cuda.Stream() if mode == "async" else None
    ready = [torch.cuda.Event(), torch.cuda.Event()]
    freed = [torch.cuda.Event(), torch.cuda.Event()]

    def pass_once() -> None:
        if mode == "fused":
            for L in range(N_LAYERS):
                lp = core["layers"][L]
                compute_layer_fused(x, lp["fp8"], lp["scales"], lp["small"],
                                    k, kv_cache)
            torch.nn.functional.linear(x, core["head_w"])
        elif mode == "sync":
            for L in range(N_LAYERS):
                decode_layer(core["layers"][L]["fp8"],
                             core["layers"][L]["scales"], ws[0])
                compute_layer(x, ws[0], core["layers"][L]["small"], k, kv_cache)
            torch.nn.functional.linear(x, core["head_w"])
        else:
            main = torch.cuda.current_stream()
            decode_into = lambda L, b: decode_layer(
                core["layers"][L]["fp8"], core["layers"][L]["scales"], ws[b])
            with torch.cuda.stream(dec_stream):
                decode_into(0, 0)
            ready[0].record(dec_stream)
            for L in range(N_LAYERS):
                b = L % 2
                main.wait_event(ready[b])
                compute_layer(x, ws[b], core["layers"][L]["small"], k, kv_cache)
                freed[b].record(main)
                if L + 1 < N_LAYERS:
                    dec_stream.wait_event(freed[(L + 1) % 2])
                    with torch.cuda.stream(dec_stream):
                        decode_into(L + 1, (L + 1) % 2)
                    ready[(L + 1) % 2].record(dec_stream)
            torch.nn.functional.linear(x, core["head_w"])
            dec_stream.synchronize()

    for _ in range(warmup):
        pass_once()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pass_once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    del ws
    torch.cuda.empty_cache()
    return statistics.median(times)


# ------------------------------------------------------------------- run ---
def _decode_parity_check(f) -> dict:
    """Parity of triton dequant vs reference, plus 0xFF-scale census.

    NaN-aware equality: torch-native e8m0 0xFF -> NaN while our helper gives
    inf; for the 375 core blocks with 0xFF scales both interpretations agree
    on NaN/inf behavior only for ±0.0 codes. Recorded as an open semantics
    question; timing is unaffected (0.05% of blocks).
    """
    w = f.get_tensor("layers.20.attn.wq_b.weight").to("cuda")
    s_raw = f.get_tensor("layers.20.attn.wq_b.scale")
    s = _e8m0_to_f32(s_raw).to("cuda")
    n, k = w.shape
    ref = w.float()
    ref = ref.view(s.shape[0], n // s.shape[0], s.shape[1], k // s.shape[1]) \
        * s[:, None, :, None]
    ref = ref.reshape(n, k).to(torch.bfloat16)
    out = torch.empty_like(ref)
    dequant_kernel(w, s_raw, out)
    eq = bool(((ref == out) |
               (torch.isnan(ref.float()) & torch.isnan(out.float())) |
               (torch.isinf(ref.float()) & torch.isinf(out.float()))).all())

    # 0xFF-scale census across the whole core (value-aware: ±0.0 codes are safe)
    ff_blocks = 0
    ff_blocks_nonzero = 0
    for layer in range(N_LAYERS):
        for name in FP8_TENSORS:
            sc = f.get_tensor(f"layers.{layer}.{name}.scale")
            if sc.dtype != torch.float8_e8m0fnu:
                continue
            m = (sc.view(torch.uint8) == 0xFF)
            if not m.any():
                continue
            wc = f.get_tensor(f"layers.{layer}.{name}.weight").view(torch.uint8)
            nb, kb = m.shape
            for i in range(nb):
                for j in range(kb):
                    if m[i, j]:
                        ff_blocks += 1
                        blk = wc.view(nb, 128, kb, 128)[i, :, j, :]
                        if int(((blk != 0) & (blk != 0x80)).sum()) > 0:
                            ff_blocks_nonzero += 1
    return {
        "nan_aware_equal": eq,
        "max_abs_diff_finite": float(
            (ref.float()[~torch.isinf(ref.float())] -
             out.float()[~torch.isinf(ref.float())]).abs().max()),
        "ff_scale_blocks_total": ff_blocks,
        "ff_scale_blocks_with_nonzero_codes": ff_blocks_nonzero,
        "ff_note": "0xFF e8m0 scales decode as inf (helper) vs NaN (torch "
                   "native). Source runtime semantics unverified; coherent "
                   "generation on sandbox implies no NaN poisoning. Must be "
                   "pinned against the source runtime before SOURCE_EXACT "
                   "signoff. Does not affect 2.11C2 timing.",
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.init()
    gpu = torch.cuda.get_device_name(0)

    # Thermal contract: gates require CONTROLLED/STEADY; warm to steady first.
    from mosaicvm.bench.thermal import warmup_steady
    steady = warmup_steady(target_seconds=120.0)
    thermal_start = snapshot()
    print(f"steady: {steady.get('steady')} | thermal at start: "
          f"{thermal_start.get('temp_c', '?')}°C")

    with safe_open(str(CORE_PATH), framework="pt") as f:
        parity = _decode_parity_check(f)
        print(f"decode parity: {parity}")
        if not parity["nan_aware_equal"]:
            print("WARNING: triton decode parity FAILED; results flagged.")

        print("loading FP8-resident core (43 layers)...", flush=True)
        core = load_core_resident(f)
        torch.cuda.synchronize()
        resident_bytes = torch.cuda.memory_allocated()
        print(f"resident after load: {resident_bytes / 1e9:.2f} GB")

        results = {
            "gpu": gpu,
            "thermal_start": thermal_start,
            "decode_parity": parity,
            "vram": {},
            "variants": {},
        }

        for k in K_VALUES:
            results["variants"][f"K{k}"] = {}
            for mode in ("fused",):
                t = run_variant(core, k, mode)
                results["variants"][f"K{k}"][mode] = t
                print(f"K={k} {mode:5s}: {t * 1e3:.1f} ms", flush=True)

        # VRAM census with workspaces allocated (peak-ish)
        ws = [alloc_workspace(), alloc_workspace()]
        x = torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda")
        kv_cache = torch.randn(CTX, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        peak_ws = torch.cuda.memory_allocated()
        decode_layer(core["layers"][0]["fp8"], core["layers"][0]["scales"], ws[0])
        compute_layer(x, ws[0], core["layers"][0]["small"], 56, kv_cache)
        torch.cuda.synchronize()
        peak_run = torch.cuda.max_memory_allocated()
        del ws, x, kv_cache
        torch.cuda.empty_cache()

        b_fp8 = sum(t.numel() * t.element_size()
                    for lp in core["layers"] for t in lp["fp8"].values())
        b_scales = sum(t.numel() * t.element_size()
                       for lp in core["layers"] for t in lp["scales"].values())
        b_small = sum(t.numel() * t.element_size()
                      for lp in core["layers"] for t in lp["small"].values())
        b_head = core["head_w"].numel() * core["head_w"].element_size()

        results["vram"] = {
            "fp8_codes_gb": b_fp8 / 1e9,
            "scales_gb": b_scales / 1e9,
            "small_bf16_gb": b_small / 1e9,
            "lm_head_bf16_gb": b_head / 1e9,
            "resident_total_gb": resident_bytes / 1e9,
            "double_workspace_gb": peak_ws / 1e9 - resident_bytes / 1e9,
            "peak_run_gb": peak_run / 1e9,
            "usable_12gb": 11.6,
            "expert_cache_headroom_gb": 11.6 - peak_run / 1e9,
            "note": "Expert-cache headroom is provisional: KV/context state, "
                    "CUDA graphs, and DSpark state are NOT yet subtracted "
                    "(Phase 2.11E census).",
        }

    thermal_end = snapshot()
    results["thermal_end"] = thermal_end
    results["thermal_label"] = classify([thermal_start, thermal_end])

    # gate evaluation on K56 fused (the production variant)
    t56 = results["variants"]["K56"]["fused"] * 1e3
    if t56 <= 115:
        gate = "EXCELLENT"
    elif t56 <= 130:
        gate = "PASS"
    elif t56 <= 160:
        gate = "VIABLE"
    elif t56 <= 200:
        gate = "WEAK"
    else:
        gate = "FP8_CORE_EXECUTION_MISS"
    results["gate_k56_fused"] = {"t_ms": t56, "gate": gate,
                                 "bf16_ceiling_ms": 102.4,
                                 "ratio_vs_ceiling": t56 / 102.4,
                                 "workspace_decode_reference": {
                                     "sync_ms": 184.6, "async_ms": 183.9,
                                     "gate": "WEAK (superseded by fused)"}}

    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"K56 fused: {t56:.1f} ms -> {gate} (BF16 ceiling 102.4 ms; "
          f"workspace-decode was 183.9 ms WEAK)")
    print(f"thermal: {thermal_start.get('temp_c', '?')}→"
          f"{thermal_end.get('temp_c', '?')}°C [{results['thermal_label']}]")
    print(f"wrote {ARTIFACT}")
    return results


if __name__ == "__main__":
    run()
