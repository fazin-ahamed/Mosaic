"""§119 NEXT 3 — TRUE-P2-class verifier-floor sweep (post-DSA, small→wide N).

Measures the exact-DSA verification-block time T_verify(N) for
N in {1,2,4,8,16,32,56,80,112} on the RTX 3060, decomposed into:

  T_A(N)      phase-A block: workspace core + packed4 bank MoE + head
              (true_p2 block_step, reproduces 1.171/1.250/1.481 s)
  T_Be(N)     banks-free eager block: 43 x compute_layer_eager
  T_Bd(N)     banks-free exact-DSA block: 43 x compute_layer_exact
              (DSA state populated at CTX=4096 via chunked prefill)

Reconstruction (bank+core+DSA cannot co-fit on 12 GB — 11.6+ GB):
  T_verify(N) = T_A(N) + T_Bd(N) - T_Be(N)
  T_moe_head(N) = T_A(N) - T_Be(N)
  T_dsa_delta(N) = T_Bd(N) - T_Be(N)

Then fits T(N) = T0 + alpha*N and emits the E[A]_req table
(30 * T_verify) against B4-O real-drafter and perfect-drafter E[A].

Classification: P2_COREBLOCK_VERIFY_TIMING (post-exact-DSA attention,
0xFF semantics still open — NOT TRUE_P2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from mosaicvm.bench.floors import HIDDEN, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.fp8_core import (
    CORE_PATH, HEAD_DIM, N_HEADS, CTX as KV_CTX,
    load_core_resident, alloc_workspace, decode_layer, compute_layer,
)
from mosaicvm.bench.phase29r import _allocate_packed4_experts, N_EXPERTS
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.wide_verify import _golden_windows
from mosaicvm.bench.p2_exact_dsa import (
    build_dsa_modules, compute_layer_eager, compute_layer_exact,
    populate_state, time_layer, CTX as DSA_CTX,
)

N_VALUES = (1, 2, 4, 8, 16, 32, 56, 80, 112)
OUT = Path("artifacts/verifier_floor_sweep.json")
GOLDEN = Path("artifacts/golden_routes.json")

# B4-O anchors for the E[A]_req table
REAL_DRAFTER_EA = {14: 2.24, 5: 2.01, 84: 2.54, 72: 1.89}   # best-by-N
PERFECT_EA = {56: 28.27, 80: 37.96, 112: 49.88}              # branchpack


def fit_linear(ns, ts):
    n = len(ns)
    sx = sum(ns)
    sy = sum(ts)
    sxx = sum(x * x for x in ns)
    sxy = sum(x * y for x, y in zip(ns, ts))
    d = n * sxx - sx * sx
    if d == 0:
        return 0.0, ts[0] if ts else 0.0
    a = (n * sxy - sx * sy) / d            # slope  (s/token)
    b = (sy - a * sx) / n                  # intercept (s)
    return a, b


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    t_start = snapshot()
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "classification": "P2_COREBLOCK_VERIFY_TIMING "
                          "(post-exact-DSA attention; 0xFF open; "
                          "NOT TRUE_P2)",
        "n_values": list(N_VALUES),
        "phases": {},
        "rows": {},
        "vram": {},
    }

    warmup_steady(120.0)

    golden = json.loads(GOLDEN.read_text())

    # ================= phase A: bank + core + head =================
    from safetensors.torch import safe_open
    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_resident(f)
    bank = _allocate_packed4_experts(N_EXPERTS)
    ws0 = alloc_workspace("cuda")
    torch.cuda.synchronize()
    results["vram"]["after_core_bank_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)
    print(f"phase A resident: {results['vram']['after_core_bank_gb']} GB")

    from mosaicvm.bench.wideverify_bootstrap import (
        layer_token_pools, bootstrap_window)
    pools = layer_token_pools(golden)
    gen_r = torch.Generator().manual_seed(909)

    phase_a = {}
    for K in N_VALUES:
        wins = _golden_windows(K, golden)
        if len(wins) >= 11 * N_LAYERS:
            step_ = max(1, len(wins) // (11 * N_LAYERS))
            routes = [
                (torch.tensor(w[0], device="cuda", dtype=torch.int64),
                 torch.tensor(w[1], device="cuda", dtype=torch.float32))
                for w in wins[::step_][:11 * N_LAYERS]
            ]
            auth = "real_golden"
        else:
            # golden sequences too short for this K: per-layer bootstrap
            routes = [(lambda p: (p[0].to("cuda"), p[1].to("cuda")))(
                bootstrap_window(pools[L % N_LAYERS], K, gen_r))
                for L in range(11 * N_LAYERS)]
            auth = "per_layer_bootstrap"
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        kv = [torch.randn(KV_CTX, 1, HEAD_DIM, dtype=torch.bfloat16,
                          device="cuda") for _ in range(2)]
        idx = 0

        def block_step():
            nonlocal idx
            x = xs[idx % 4]
            kv_cache = kv[idx % 2]
            for L in range(N_LAYERS):
                lp = core["layers"][L]
                decode_layer(lp["fp8"], lp["scales"], ws0)
                h = compute_layer(x, ws0, lp["small"], K, kv_cache)
                h = h + x * 0.001
                out = fused_moe_mxint(h.float().to(torch.bfloat16),
                                      routes[idx][0], routes[idx][1],
                                      bank=bank, weight_repr="packed4",
                                      crossover=0, imma_k=8)
                x = (out + x).to(torch.bfloat16)
                idx += 1
            torch.nn.functional.linear(x, core["head_w"])

        t_a = time_layer(block_step, reps=8, warm=3)
        phase_a[K] = t_a
        print(f"  phase A N={K:3d} [{auth}]: T_A={t_a:.4f}s "
              f"({K / t_a:.1f} raw tok/s)")
        del routes, xs, kv
        torch.cuda.empty_cache()
    results["phases"]["A_bank_core_head"] = {str(k): round(v, 4)
                                             for k, v in phase_a.items()}

    # free bank for phase B
    del bank, ws0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    results["vram"]["after_bank_free_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)

    # ================= phase B: DSA swap (banks-free) =================
    with safe_open(str(CORE_PATH), framework="pt") as f:
        dsa_mods = build_dsa_modules(f)
    torch.cuda.synchronize()
    results["vram"]["after_dsa_load_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)
    print(f"phase B resident: {results['vram']['after_dsa_load_gb']} GB")

    gen = torch.Generator(device="cuda").manual_seed(9100)
    for dsa in dsa_mods:
        populate_state(dsa, gen)
    torch.cuda.synchronize()
    results["vram"]["after_populate_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)

    phase_be, phase_bd = {}, {}
    for K in N_VALUES:
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        kv_cache = torch.randn(KV_CTX, 1, HEAD_DIM, dtype=torch.bfloat16,
                               device="cuda")

        def eager_block():
            for L in range(N_LAYERS):
                compute_layer_eager(xs[0], core["layers"][L], K, kv_cache)

        pos = [DSA_CTX]

        def exact_block():
            p = pos[0]
            for L in range(N_LAYERS):
                compute_layer_exact(xs[p % 4], core["layers"][L], K,
                                    dsa_mods[L], p)
            pos[0] += K

        t_be = time_layer(eager_block, reps=8, warm=3)
        t_bd = time_layer(exact_block, reps=8, warm=3)
        phase_be[K], phase_bd[K] = t_be, t_bd
        print(f"  phase B N={K:3d}: eager={t_be:.4f}s exact={t_bd:.4f}s "
              f"(Δ_dsa={((t_bd - t_be) * 1e3):+.1f} ms)")
        del xs, kv_cache
        torch.cuda.empty_cache()
    results["phases"]["B_eager_banks_free"] = {str(k): round(v, 4)
                                               for k, v in phase_be.items()}
    results["phases"]["B_exact_dsa_banks_free"] = {
        str(k): round(v, 4) for k, v in phase_bd.items()}

    # ================= reconstruction + decomposition =================
    for K in N_VALUES:
        t_a, t_be, t_bd = phase_a[K], phase_be[K], phase_bd[K]
        t_verify = t_a + t_bd - t_be
        results["rows"][str(K)] = {
            "T_A_bank_core_head_s": round(t_a, 4),
            "T_Be_eager_s": round(t_be, 4),
            "T_Bd_exact_dsa_s": round(t_bd, 4),
            "T_moe_head_s": round(t_a - t_be, 4),
            "T_dsa_delta_s": round(t_bd - t_be, 4),
            "T_verify_reconstructed_s": round(t_verify, 4),
            "raw_verify_tok_s": round(K / t_verify, 1),
            "E_req_for_30": round(30.0 * t_verify, 1),
        }

    # ================= fits =================
    ns = list(N_VALUES)
    t_ver = [phase_a[k] + phase_bd[k] - phase_be[k] for k in ns]
    a_v, b_v = fit_linear([float(x) for x in ns], t_ver)
    a_m, b_m = fit_linear([float(x) for x in ns],
                          [phase_a[k] - phase_be[k] for k in ns])
    a_d, b_d = fit_linear([float(x) for x in ns],
                          [phase_bd[k] - phase_be[k] for k in ns])
    results["fits"] = {
        "T_verify": {"T0_s": round(b_v, 4), "alpha_s_per_tok": round(a_v, 5)},
        "T_moe_head": {"T0_s": round(b_m, 4),
                       "alpha_s_per_tok": round(a_m, 5)},
        "T_dsa_delta": {"T0_s": round(b_d, 4),
                        "alpha_s_per_tok": round(a_d, 5)},
    }
    t_end = snapshot()
    results["thermal"] = {"start": t_start, "end": t_end}

    print("\n=== VERIFIER FLOOR SWEEP (post-DSA, reconstructed) ===")
    print(f"{'N':>4}{'T_verify':>10}{'raw/s':>8}{'E_req30':>9}"
          f"{'T_moe+h':>9}{'Δ_dsa':>8}")
    for K in N_VALUES:
        r = results["rows"][str(K)]
        print(f"{K:>4}{r['T_verify_reconstructed_s']:>10.3f}"
              f"{r['raw_verify_tok_s']:>8.1f}{r['E_req_for_30']:>9.1f}"
              f"{r['T_moe_head_s']:>9.3f}{r['T_dsa_delta_s']:>8.3f}")
    print(f"\nfit: T_verify(N) = {b_v:.3f}s + {a_v * 1e3:.2f}ms * N")
    print(f"      T_moe_head(N) = {b_m:.3f}s + {a_m * 1e3:.2f}ms * N")
    print(f"      T_dsa_delta(N) = {b_d:.3f}s + {a_d * 1e3:.2f}ms * N")
    print(f"\nE[A]_req at T0-only (zero-width floor): "
          f"{30 * b_v:.1f} accepted tokens")

    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
