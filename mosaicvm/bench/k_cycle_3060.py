"""Native five-token DSpark kill gate — RTX 3060 K1–K6 verifier cycle
(roadmap §30-32).

Measures the TARGET-SIDE verification cost of a native DSpark 5-token
block at K=1..6 on the local RTX 3060 using core.safetensors (same
component method as core_timing.py).

Terms:
  MEASURED (RTX 3060, real core weights):
    * per-layer core: MLA projections, DSA-style attention, shared
      expert, router, HC, norms × 43 layers
    * LM head + final norm
  RECONSTRUCTED (labeled — routed experts absent locally):
    * MoE routed-expert term from artifacts/floor_b_moe.json
      exact_k8_mid (13.8 ms/layer, K8-class batch) — an UPPER-bound-ish
      anchor for K<=6; real K1-K6 cost may differ (launch-bound).
  UNMEASURED:
    * DSpark drafter forward (Blackwell-only so far)
    * movement / scheduler / runtime

Kill-gate arithmetic (roadmap §30):
  perfect native cycle  : 6 tokens/block  -> needs < 200 ms
  measured D1 cycle    : 2.933 tokens/block -> needs < 97.8 ms
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open

from mosaicvm.bench.thermal import snapshot, classify

CORE_PATH = Path(".mosaicvm/core/core.safetensors")
ARTIFACT = Path("artifacts/k_cycle_3060.json")
FLOOR = Path("artifacts/floor_b_moe.json")

K_VALUES = (1, 2, 3, 4, 5, 6)
N_LAYERS = 43
HIDDEN = 4096
N_HEADS = 64
HEAD_DIM = 512
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
CTX = 1024

# Kill gates (roadmap §30)
GATE_PERFECT_MS = 200.0     # 6 tokens incl bonus at 30 tok/s
GATE_D1_MS = 97.8           # 2.933 tokens incl bonus at 30 tok/s
D1_TOKENS_PER_BLOCK = 2.933
PERFECT_TOKENS_PER_BLOCK = 6.0


def _bench(fn, n_iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def run() -> dict:
    assert torch.cuda.is_available(), "requires the local RTX 3060"
    thermal_start = snapshot()
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    moe_layer_s = None
    if FLOOR.exists():
        floor = json.loads(FLOOR.read_text())
        moe_layer_s = floor.get("exact_k8_mid_sec")
        print(f"MoE anchor (RECONSTRUCTED, K8-class): "
              f"{moe_layer_s*1e3:.1f} ms/layer")

    results = {
        "gpu": gpu,
        "profile": "DISPLAY_ATTACHED",
        "thermal_start": thermal_start,
        "classification": {
            "core_terms": "MEASURED_RT3060",
            "moe_term": "RECONSTRUCTED_FROM_FLOOR_B_MOE_K8_ANCHOR"
                        if moe_layer_s else "UNAVAILABLE",
            "drafter": "UNMEASURED_BLACKWELL_ONLY",
            "movement_scheduler": "UNMEASURED",
        },
        "kill_gates": {
            "perfect_native_ms": GATE_PERFECT_MS,
            "d1_native_ms": GATE_D1_MS,
            "tokens_per_block": {
                "perfect": PERFECT_TOKENS_PER_BLOCK,
                "d1_measured": D1_TOKENS_PER_BLOCK,
            },
        },
        "K": {},
    }

    with safe_open(str(CORE_PATH), framework="pt") as f:
        head_w = f.get_tensor("head.weight").to(torch.bfloat16).to("cuda")
        norm_w = f.get_tensor("norm.weight").to("cuda")

        pre = "layers.20."
        def g_fp8(suffix):
            w = f.get_tensor(pre + suffix + ".weight")
            s = f.get_tensor(pre + suffix + ".scale")
            w32 = w.float()
            n, k = w32.shape
            if s.shape[0] > 0 and n % s.shape[0] == 0 and \
                    k % s.shape[1] == 0:
                bn, bk = n // s.shape[0], k // s.shape[1]
                w32 = (w32.view(s.shape[0], bn, s.shape[1], bk)
                       * s.float()[:, None, :, None]).reshape(n, k)
            return w32.to(torch.bfloat16).to("cuda")

        wq_a, wq_b = g_fp8("attn.wq_a"), g_fp8("attn.wq_b")
        wkv, wo_a, wo_b = g_fp8("attn.wkv"), g_fp8("attn.wo_a"), g_fp8("attn.wo_b")
        sh1, sh2, sh3 = (g_fp8("ffn.shared_experts.w1"),
                         g_fp8("ffn.shared_experts.w2"),
                         g_fp8("ffn.shared_experts.w3"))
        gate_w = f.get_tensor(pre + "ffn.gate.weight").to(torch.bfloat16).cuda()
        hc_attn_fn = f.get_tensor(pre + "hc_attn_fn").cuda()
        hc_attn_base = f.get_tensor(pre + "hc_attn_base").cuda()
        hc_attn_scale = f.get_tensor(pre + "hc_attn_scale").cuda()
        hc_ffn_fn = f.get_tensor(pre + "hc_ffn_fn").cuda()
        hc_ffn_base = f.get_tensor(pre + "hc_ffn_base").cuda()
        hc_ffn_scale = f.get_tensor(pre + "hc_ffn_scale").cuda()
        q_norm_w = f.get_tensor(pre + "attn.q_norm.weight").cuda()
        kv_norm_w = f.get_tensor(pre + "attn.kv_norm.weight").cuda()
        attn_norm_w = f.get_tensor(pre + "attn_norm.weight").cuda()
        wo_a_g = wo_a.view(O_GROUPS, O_LORA, 4096)

        for k in K_VALUES:
            x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
            kv_cache = torch.randn(CTX + k, 1, HEAD_DIM,
                                   dtype=torch.bfloat16, device="cuda")
            k_exp = kv_cache.expand(CTX + k, N_HEADS, HEAD_DIM)
            q = torch.randn(k, N_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                            device="cuda")

            def attn_proj():
                ql = torch.nn.functional.linear(x, wq_a)
                ql = torch.nn.functional.rms_norm(ql, (Q_LORA,), q_norm_w)
                qb = torch.nn.functional.linear(ql, wq_b)
                kv = torch.nn.functional.linear(x, wkv)
                kv = torch.nn.functional.rms_norm(kv, (HEAD_DIM,), kv_norm_w)
                return qb, kv
            t_attn_proj = _bench(attn_proj)

            def attn_core():
                scores = torch.einsum("khd,chd->kch", q, k_exp) \
                    / (HEAD_DIM ** 0.5)
                a = torch.softmax(scores, dim=-1)
                return torch.einsum("kch,chd->khd", a, k_exp)
            t_attn = _bench(attn_core, n_iters=5, warmup=2)

            def attn_out_proj():
                ao = torch.randn(k, O_GROUPS, 4096,
                                 dtype=torch.bfloat16, device="cuda")
                r = torch.einsum("kgd,grd->kgr", ao, wo_a_g)
                return torch.nn.functional.linear(
                    r.reshape(k, O_GROUPS * O_LORA), wo_b)
            t_out = _bench(attn_out_proj)

            def shared():
                g = torch.nn.functional.linear(x, sh1).float().clamp(max=10)
                u = torch.nn.functional.linear(x, sh3).float() \
                    .clamp(-10, 10)
                z = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
                return torch.nn.functional.linear(z, sh2)
            t_shared = _bench(shared)

            def router():
                lg = torch.nn.functional.linear(x.float(), gate_w.float())
                s = torch.nn.functional.softplus(lg).sqrt()
                return torch.topk(s, 6, dim=-1).indices
            t_router = _bench(router)

            def norms():
                return torch.nn.functional.rms_norm(x, (HIDDEN,),
                                                     attn_norm_w)
            t_norms = _bench(norms)

            def hc(fn_w, base, scale):
                hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32,
                                    device="cuda")
                mixes = torch.nn.functional.linear(hc_in, fn_w)
                pre = torch.sigmoid(
                    mixes * scale.float().mean()
                    + base.float()) + 1e-6
                return pre
            t_hc_attn = _bench(lambda: hc(hc_attn_fn, hc_attn_base,
                                          hc_attn_scale))
            t_hc_ffn = _bench(lambda: hc(hc_ffn_fn, hc_ffn_base,
                                         hc_ffn_scale))

            # attn_proj already covers BOTH the q path (wq_a->q_norm->wq_b)
            # and the kv path (wkv->kv_norm) in one call: count once.
            per_layer = (t_attn_proj + t_attn + t_out + t_shared
                         + t_router + 3 * t_norms
                         + t_hc_attn + t_hc_ffn)
            t_core_layer = per_layer * N_LAYERS

            def lm_head():
                return torch.nn.functional.linear(x, head_w)
            t_head = _bench(lm_head, n_iters=10, warmup=3)

            core_total = t_core_layer + t_head
            moe_total = (moe_layer_s * N_LAYERS) if moe_layer_s else 0.0
            verify_total = core_total + moe_total

            results["K"][str(k)] = {
                "attn_proj_ms": t_attn_proj * 1e3,
                "attn_core_ms": t_attn * 1e3,
                "attn_out_ms": t_out * 1e3,
                "shared_ms": t_shared * 1e3,
                "router_ms": t_router * 1e3,
                "norms_ms": t_norms * 1e3,
                "hc_attn_ms": t_hc_attn * 1e3,
                "hc_ffn_ms": t_hc_ffn * 1e3,
                "per_layer_ms": per_layer * 1e3,
                "core_43layers_ms": t_core_layer * 1e3,
                "lm_head_ms": t_head * 1e3,
                "core_total_ms_MEASURED": core_total * 1e3,
                "moe_43layers_ms_RECONSTRUCTED": moe_total * 1e3,
                "verify_total_ms": verify_total * 1e3,
            }
            print(f"K{k}: core {core_total*1e3:7.1f} ms (measured) "
                  f"+ MoE {moe_total*1e3:7.1f} ms (reconstructed) "
                  f"= {verify_total*1e3:7.1f} ms")

    # Kill-gate evaluation at the native K5 (+1 bonus) geometry
    k5 = results["K"].get("5") or results["K"]["6"]
    verify_ms = k5["verify_total_ms"]
    results["kill_gate"] = {
        "verify_K5_ms": verify_ms,
        "perfect_native": {
            "tokens": PERFECT_TOKENS_PER_BLOCK,
            "required_cycle_ms": GATE_PERFECT_MS,
            "verify_only_ms": verify_ms,
            "verdict": "PASS_VERIFY_ONLY" if verify_ms < GATE_PERFECT_MS
            else "FAIL_VERIFY_ONLY",
            "note": ("verify term alone; drafter + movement + runtime "
                     "still unmeasured and additive"),
        },
        "d1_native": {
            "tokens": D1_TOKENS_PER_BLOCK,
            "required_cycle_ms": GATE_D1_MS,
            "verify_only_ms": verify_ms,
            "verdict": "PASS_VERIFY_ONLY" if verify_ms < GATE_D1_MS
            else "FAIL_VERIFY_ONLY",
            "implied_rate_if_verify_only": round(
                D1_TOKENS_PER_BLOCK / (verify_ms / 1e3), 2),
        },
    }

    thermal_end = snapshot()
    results["thermal_end"] = thermal_end
    results["thermal_label"] = classify([thermal_start, thermal_end])
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nthermal: {thermal_start.get('temp_c','?')}->"
          f"{thermal_end.get('temp_c','?')}C [{results['thermal_label']}]")
    print(f"K5 verify = {verify_ms:.1f} ms "
          f"(perfect gate {GATE_PERFECT_MS} ms, D1 gate {GATE_D1_MS} ms)")
    print(f"D1 verify-only implied rate = "
          f"{results['kill_gate']['d1_native']['implied_rate_if_verify_only']}"
          f" tok/s")
    print(f"wrote {ARTIFACT}")
    return results


if __name__ == "__main__":
    run()
