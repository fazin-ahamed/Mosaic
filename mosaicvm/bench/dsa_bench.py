"""§60-1: exact DSA attention vs eager stand-in — matched full pipelines.

Both sides are complete attention modules with IDENTICAL weights and
IDENTICAL projection paths (bf16: q_a -> q_norm -> q_b -> per-head RMS;
kv_norm(kv_proj); wo_a grouped einsum -> wo_b). The ONLY difference is the
attention mechanism:

  eager : dense softmax over a fixed 1024-token KV cache (the stand-in that
          fed the N32/56/80/112 = 148/232/260/325 ms core curve in
          mosaicvm/bench/fp8_core.compute_layer_fused)
  exact : real DSA — window gather (128) + compressor (+ indexer on ratio-4
          layers) + sparse attention with sink + inverse-rope

`ratio_N*` is the mechanism cost multiplier for swapping the eager stand-in
for exact DSA inside an otherwise identical module. Level: P1 (single
attention module, local RTX 3060); NOT a P-level speedup claim. Both sides
run bf16 so the mechanism delta is isolated from the FP8 weight path.
"""
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import (ExactAttention, load_exact_attention,
                                       cast_exact_attention)

DEV = "cuda"
WIDTHS = [32, 56, 80, 112]
LAYERS = [0, 2, 3]          # ratio 0 / ratio 4+indexer / ratio 128
KV_LEN = 1024               # eager stand-in cache length (fp8_core convention)
WARMUP, REPS = 8, 30
cfg = V4Config()

torch.manual_seed(0)
f = safe_open(".mosaicvm/core/core.safetensors", "pt")


def _rms(w, x, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def eager_forward(mod: ExactAttention, x: torch.Tensor) -> torch.Tensor:
    """Eager attention block on the SAME module weights: identical
    q/kv projections, per-head q RMS, q/KV rope, inverse-rope on the output,
    and O projection as ExactAttention.forward. Only the attention core
    differs: dense softmax over a fixed 1024-token KV cache (the fp8_core
    stand-in) instead of window/compressor/indexer gather attention."""
    from mosaicvm.model.dsa import apply_rotary_emb
    s = x.shape[0]
    qr = _rms(mod.q_norm.weight, mod.q_a(x), 1e-6)
    q = mod.q_b(qr).view(s, mod.n_heads, mod.head_dim)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6).to(q.dtype)
    f = mod.freqs[:s]
    qr_ = apply_rotary_emb(q[..., -mod.rd:], f)
    q = torch.cat([q[..., :-mod.rd], qr_], -1)
    kv = _rms(mod.kv_norm.weight, mod.kv_proj(x), 1e-6)     # [s, 512]
    kvr = apply_rotary_emb(kv[..., -mod.rd:], f)
    kv = torch.cat([kv[..., :-mod.rd], kvr], -1)
    k = mod._eager_kv                                       # [1024, 512]
    scores = torch.einsum("shd,cd->shc", q.float(), k.float()) * mod.scale
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("shc,cd->shd", p, k.float()).to(x.dtype)
    orot = apply_rotary_emb(o[..., -mod.rd:], f, inverse=True)
    o = torch.cat([o[..., :-mod.rd], orot], -1)
    o = o.reshape(s, mod.o_groups, -1)
    wo_a = mod.wo_a.weight.view(mod.o_groups, cfg.o_lora_rank, -1)
    o = torch.einsum("sgd,grd->sgr", o.to(wo_a.dtype), wo_a).reshape(s, -1)
    return mod.wo_b(o)


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = sorted(ts)
    return ts[len(ts) // 2] * 1e3   # median ms


results = {"level": "P1", "gpu": torch.cuda.get_device_name(0),
           "thermal": "UNKNOWN", "widths": WIDTHS,
           "comparison": "matched full attention modules, identical bf16 "
                         "projections; only attention mechanism differs "
                         "(eager dense 1024-KV vs exact DSA)",
           "layers": {}}

for L in LAYERS:
    attn = ExactAttention(cfg, layer_id=L, max_seq=1024, device=DEV)
    cast_exact_attention(attn, DEV)
    load_exact_attention(attn, L, f.get_tensor)
    attn._eager_kv = torch.randn(KV_LEN, attn.head_dim,
                                 dtype=torch.bfloat16, device=DEV)
    row = {"ratio": cfg.compress_ratios[L]}
    for N in WIDTHS:
        x = torch.randn(N, cfg.hidden_size, dtype=torch.bfloat16, device=DEV)
        with torch.no_grad():
            row[f"exact_ms_N{N}"] = round(bench(attn, x), 3)
            row[f"eager_ms_N{N}"] = round(bench(eager_forward, attn, x), 3)
            row[f"ratio_N{N}"] = round(row[f"exact_ms_N{N}"] /
                                      row[f"eager_ms_N{N}"], 2)
            out = attn(x)
        row[f"finite_frac_N{N}"] = round(
            torch.isfinite(out.float()).float().mean().item(), 4)
    results["layers"][L] = row
    print(f"L{L} (ratio {row['ratio']}):", json.dumps(
        {k: v for k, v in row.items() if k != 'ratio'}))

Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/dsa_bench.json").write_text(json.dumps(results, indent=1))
print("written artifacts/dsa_bench.json")
