"""Phase 2.11C/D: exact core execution + T_core(K,C) + VRAM census.

Loads FP8/BF16 core tensors from .mosaicvm/core/, dequantizes to BF16 (SM86
has no FP8 tensor cores — the FP8→BF16 decode IS the exact execution
strategy for now; a register-decode race is 2.11C proper and can only
improve on this baseline), and measures per-component latency for a single
target forward at verification widths K32–56 under the thermal contract.

Components measured separately:
  MLA/DSA projections (wq_a, wq_b, wkv, wo_a, wo_b)
  shared expert (w1, w2, w3 SwiGLU)
  router gate
  HyperConnections (attn + ffn + head)
  norms
  LM head
  KV ops (context-dependent part; short-context here)
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open

from mosaicvm.bench.thermal import snapshot, classify
from mosaicvm.model.config import V4Config

CORE_PATH = Path(".mosaicvm/core/core.safetensors")
ARTIFACT = Path("artifacts/core_timing.json")

K_VALUES = (32, 40, 48, 56)
N_LAYERS = 43
HIDDEN = 4096
N_HEADS = 64
HEAD_DIM = 512
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
MOE_INTER = 2048
VOCAB = 129280
CTX = 1024  # short-context KV for this measurement


def _load_layer(f, layer: int) -> dict:
    """Load one layer's core tensors, dequantizing FP8→BF16."""
    pre = f"layers.{layer}."
    out = {}

    def get_fp8(suffix):
        w = f.get_tensor(pre + suffix + ".weight")
        s = f.get_tensor(pre + suffix + ".scale")
        # dequant: scale is [n_blocks, k_blocks], each block 128x128
        w32 = w.float()
        n, k = w32.shape
        if s.shape[0] > 0 and n % s.shape[0] == 0 and k % s.shape[1] == 0:
            bn = n // s.shape[0]
            bk = k // s.shape[1]
            w32 = w32.view(s.shape[0], bn, s.shape[1], bk) * s.float()[:, None, :, None]
            w32 = w32.reshape(n, k)
        return w32.to(torch.bfloat16).to('cuda')

    out["wq_a"] = get_fp8("attn.wq_a")  # [1024, 4096]
    out["wq_b"] = get_fp8("attn.wq_b")  # [32768, 1024]
    out["wkv"] = get_fp8("attn.wkv")    # [512, 4096]
    out["wo_a"] = get_fp8("attn.wo_a")  # [8192, 4096]
    out["wo_b"] = get_fp8("attn.wo_b")  # [4096, 8192]
    out["shared_w1"] = get_fp8("ffn.shared_experts.w1")  # [2048, 4096]
    out["shared_w2"] = get_fp8("ffn.shared_experts.w2")  # [4096, 2048]
    out["shared_w3"] = get_fp8("ffn.shared_experts.w3")  # [2048, 4096]
    out["gate_w"] = f.get_tensor(pre + "ffn.gate.weight").to(torch.bfloat16).to("cuda")  # [256, 4096]
    bias_name = pre + "ffn.gate.bias"
    gate_bias = f.get_tensor(bias_name) if bias_name in f.keys() else None
    out["gate_bias"] = (gate_bias.to("cuda") if gate_bias is not None else torch.zeros(256, device="cuda"))
    out["hc_attn_fn"] = f.get_tensor(pre + "hc_attn_fn").to("cuda")  # [24, 16384] F32
    out["hc_attn_base"] = f.get_tensor(pre + "hc_attn_base").to("cuda")  # [24] F32
    out["hc_attn_scale"] = f.get_tensor(pre + "hc_attn_scale").to("cuda")  # [3] F32
    out["hc_ffn_fn"] = f.get_tensor(pre + "hc_ffn_fn").to("cuda")
    out["hc_ffn_base"] = f.get_tensor(pre + "hc_ffn_base").to("cuda")
    out["hc_ffn_scale"] = f.get_tensor(pre + "hc_ffn_scale").to("cuda")
    out["attn_norm_w"] = f.get_tensor(pre + "attn_norm.weight").to("cuda")  # [4096] BF16
    out["ffn_norm_w"] = f.get_tensor(pre + "ffn_norm.weight").to("cuda")
    out["q_norm_w"] = f.get_tensor(pre + "attn.q_norm.weight").to("cuda")  # [1024]
    out["kv_norm_w"] = f.get_tensor(pre + "attn.kv_norm.weight").to("cuda")  # [512]
    out["attn_sink"] = f.get_tensor(pre + "attn.attn_sink").to("cuda")  # [64]
    return out


def _bench_component(fn, n_iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_get_device_name(0) if hasattr(torch.cuda, 'get_get_device_name') else torch.cuda.get_device_name(0)

    thermal_start = snapshot()
    print(f"thermal at start: {thermal_start.get('temp_c', '?')}°C")

    with safe_open(str(CORE_PATH), framework="pt") as f:
        head_w = f.get_tensor("head.weight").to(torch.bfloat16).to("cuda")  # [VOCAB, HIDDEN]
        norm_w = f.get_tensor("norm.weight").to("cuda")  # [HIDDEN]
        hc_head_fn = f.get_tensor("hc_head_fn").to("cuda")  # [4, 16384] F32
        hc_head_base = f.get_tensor("hc_head_base").to("cuda")  # [4] F32
        hc_head_scale = f.get_tensor("hc_head_scale").to("cuda")  # [1] F32

        # Load ONE representative layer for component timing (all layers identical shape).
        # VRAM census computed from the safetensors header (FP8 bytes) + BF16 expansion factor.
        print("loading representative layer...", flush=True)
        layers = [_load_layer(f, 20)]  # mid-model layer with gate bias
        torch.cuda.synchronize()
        core_vram_bytes = torch.cuda.memory_allocated()
        print(f"single layer VRAM: {core_vram_bytes / 1e9:.2f} GB")

        results = {"gpu": gpu, "thermal_start": thermal_start}

        for k in K_VALUES:
            print(f"\n=== K={k} ===")
            x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
            # fake KV cache for short-context attention
            kv_cache = torch.randn(CTX + k, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

            # Pre-move layer 0 tensors for component timing (representative)
            l0 = layers[0]

            comp_times = {}

            # --- MLA projections ---
            # wq_a: hidden -> q_lora [K, 1024]
            def mla_q_a():
                return torch.nn.functional.linear(x, l0["wq_a"])
            comp_times["mla_wq_a"] = _bench_component(mla_q_a)

            q_lora = mla_q_a()
            # q_norm (RMSNorm on lora)
            def mla_q_norm():
                return torch.nn.functional.rms_norm(q_lora, (Q_LORA,), l0["q_norm_w"])
            comp_times["mla_q_norm"] = _bench_component(mla_q_norm)

            q_normed = mla_q_norm()
            # wq_b: q_lora -> [K, 32768] (64 heads × 512)
            def mla_q_b():
                return torch.nn.functional.linear(q_normed, l0["wq_b"])
            comp_times["mla_wq_b"] = _bench_component(mla_q_b)

            # wkv: hidden -> [K, 512]
            def mla_wkv():
                return torch.nn.functional.linear(x, l0["wkv"])
            comp_times["mla_wkv"] = _bench_component(mla_wkv)

            kv = mla_wkv()
            def mla_kv_norm():
                return torch.nn.functional.rms_norm(kv, (HEAD_DIM,), l0["kv_norm_w"])
            comp_times["mla_kv_norm"] = _bench_component(mla_kv_norm)

            # wo_a: [K, 4096] group input -> [K, 8192] (requires attention output reshaped)
            # simulate: attention output is [K, N_HEADS*HEAD_DIM] = [K, 32768]
            attn_out = torch.randn(k, N_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            # wo_a operates on group_head_dim = 32768/8 = 4096 per group
            attn_grouped = attn_out.view(k, O_GROUPS, 4096)
            wo_a_w = l0["wo_a"]  # [8192, 4096]
            wo_a_grouped = wo_a_w.view(O_GROUPS, O_LORA, 4096)
            def mla_wo_a():
                # [K, 8, 4096] @ [8, 4096, 1024] -> [K, 8, 1024]
                return torch.einsum("kgd,grd->kgr", attn_grouped, wo_a_grouped)
            comp_times["mla_wo_a"] = _bench_component(mla_wo_a)

            wo_a_out = mla_wo_a().reshape(k, O_GROUPS * O_LORA)
            def mla_wo_b():
                return torch.nn.functional.linear(wo_a_out, l0["wo_b"])
            comp_times["mla_wo_b"] = _bench_component(mla_wo_b)

            # --- attention itself (DSA-style, short context) ---
            q = torch.randn(k, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            k_cache = torch.randn(CTX, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            k_exp = k_cache.expand(CTX, N_HEADS, HEAD_DIM)
            def attn_compute():
                scores = torch.einsum("khd,chd->kch", q, k_exp) / (HEAD_DIM ** 0.5)
                # DSA-style top-512 selection would go here; approximate with softmax for timing
                attn = torch.softmax(scores, dim=-1)
                return torch.einsum("kch,chd->khd", attn, k_exp)
            comp_times["attn_core"] = _bench_component(attn_compute, n_iters=5)

            # --- shared expert (SwiGLU at moe_intermediate) ---
            def shared_gate():
                return torch.nn.functional.linear(x, l0["shared_w1"])
            comp_times["shared_w1"] = _bench_component(shared_gate)
            def shared_up():
                return torch.nn.functional.linear(x, l0["shared_w3"])
            comp_times["shared_w3"] = _bench_component(shared_up)

            gate_out = shared_gate().float()
            up_out = shared_up().float()
            def shared_swiglu():
                g = gate_out.clamp(max=10.0)
                u = up_out.clamp(-10.0, 10.0)
                z = torch.nn.functional.silu(g) * u
                return z.to(torch.bfloat16)
            comp_times["shared_swiglu"] = _bench_component(shared_swiglu)

            z = shared_swiglu()
            def shared_down():
                return torch.nn.functional.linear(z, l0["shared_w2"])
            comp_times["shared_w2"] = _bench_component(shared_down)

            # --- router ---
            def router_gate():
                logits = torch.nn.functional.linear(x.float(), l0["gate_w"].float())
                s = torch.nn.functional.softplus(logits).sqrt()
                s = s + l0["gate_bias"].float()
                idx = torch.topk(s, 6, dim=-1).indices
                return idx
            comp_times["router"] = _bench_component(router_gate)

            # --- HyperConnections ---
            # hc_attn: mixing for attention stream
            def hc_attn():
                hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32, device="cuda")
                mixes = torch.nn.functional.linear(hc_in, l0["hc_attn_fn"].float())
                pre = torch.sigmoid(mixes * l0["hc_attn_scale"].float().mean() + l0["hc_attn_base"].float()) + 1e-6
                return pre
            comp_times["hc_attn"] = _bench_component(hc_attn)

            def hc_ffn():
                hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32, device="cuda")
                mixes = torch.nn.functional.linear(hc_in, l0["hc_ffn_fn"].float())
                pre = torch.sigmoid(mixes * l0["hc_ffn_scale"].float().mean() + l0["hc_ffn_base"].float()) + 1e-6
                return pre
            comp_times["hc_ffn"] = _bench_component(hc_ffn)

            # --- norms ---
            def rms_norm():
                return torch.nn.functional.rms_norm(x, (HIDDEN,), l0["attn_norm_w"])
            comp_times["rms_norm"] = _bench_component(rms_norm)

            # --- LM head (single call per forward) ---
            def lm_head():
                return torch.nn.functional.linear(x, head_w)
            comp_times["lm_head"] = _bench_component(lm_head, n_iters=5, warmup=2)

            # --- final norm + hc_head ---
            def final_norm():
                return torch.nn.functional.rms_norm(x, (HIDDEN,), norm_w)
            comp_times["final_norm"] = _bench_component(final_norm)

            # Aggregate: per-layer cost (attention block + ffn block, no MoE)
            attn_block = (comp_times["mla_wq_a"] + comp_times["mla_q_norm"] +
                          comp_times["mla_wq_b"] + comp_times["mla_wkv"] +
                          comp_times["mla_kv_norm"] + comp_times["attn_core"] +
                          comp_times["mla_wo_a"] + comp_times["mla_wo_b"] +
                          comp_times["hc_attn"] + comp_times["rms_norm"])
            ffn_block = (comp_times["shared_w1"] + comp_times["shared_w3"] +
                         comp_times["shared_swiglu"] + comp_times["shared_w2"] +
                         comp_times["router"] + comp_times["hc_ffn"] +
                         comp_times["rms_norm"])
            per_layer = attn_block + ffn_block
            t_core_layer = per_layer * N_LAYERS
            t_core_total = t_core_layer + comp_times["lm_head"] + comp_times["final_norm"]

            results[f"K{k}"] = {
                "component_ms": {k2: v * 1e3 for k2, v in comp_times.items()},
                "attn_block_ms": attn_block * 1e3,
                "ffn_block_ms": ffn_block * 1e3,
                "per_layer_ms": per_layer * 1e3,
                "t_core_layer_ms": t_core_layer * 1e3,
                "t_core_total_s": t_core_total,
            }
            print(f"  attn_block: {attn_block*1e3:.3f} ms | ffn_block: {ffn_block*1e3:.3f} ms")
            print(f"  per_layer: {per_layer*1e3:.3f} ms → 43 layers: {t_core_layer*1e3:.1f} ms")
            print(f"  + LM head: {comp_times['lm_head']*1e3:.1f} ms")
            print(f"  T_core(K={k}) = {t_core_total:.4f} s")

    thermal_end = snapshot()
    thermal_label = classify([thermal_start, thermal_end])
    results["thermal_end"] = thermal_end
    results["thermal_label"] = thermal_label

    # VRAM census
    # VRAM census from header bytes: FP8 core = 7.79 GB source, BF16 dequant = 2x
    cc = json.loads(Path("artifacts/core_classification.json").read_text())
    b_fp8 = cc["b_core_resident_gb"]
    b_bf16 = b_fp8 * 2  # dequantized expansion
    results["vram_census"] = {
        "fp8_source_gb": b_fp8,
        "bf16_dequant_gb": b_bf16,
        "single_layer_bf16_gb": core_vram_bytes / 1e9,
        "fits_12gb": b_bf16 + 1.06 < 11.6,  # + LM head BF16
        "note": "Full BF16 dequant of core does NOT fit 12 GB. Runtime must "
                "either execute from FP8 directly (register decode), keep BF16 "
                "per-layer only, or stream layers. FP8 source is 7.79 GB and fits.",
    }

    # Compare to the projected core from wide_verify
    wv_path = Path("artifacts/wide_verify.json")
    if wv_path.exists():
        wv = json.loads(wv_path.read_text())
        core_proj = wv.get("core", {})
        comparison = {}
        for k in K_VALUES:
            measured = results.get(f"K{k}", {}).get("t_core_total_s")
            projected = core_proj.get(str(k), {}).get("core_block_s")
            if measured and projected:
                comparison[str(k)] = {
                    "measured_s": measured,
                    "projected_s": projected,
                    "ratio": measured / projected,
                }
        results["projection_comparison"] = comparison

    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"thermal: {thermal_start.get('temp_c', '?')}→{thermal_end.get('temp_c', '?')}°C [{thermal_label}]")
    print(f"core VRAM: {core_vram_bytes/1e9:.2f} GB")
    if "projection_comparison" in results:
        for k, c in results["projection_comparison"].items():
            print(f"K{k}: measured {c['measured_s']:.4f}s vs projected {c['projected_s']:.4f}s "
                  f"({c['ratio']:.2f}×)")
    print(f"wrote {ARTIFACT}")
    return results


if __name__ == "__main__":
    run()
