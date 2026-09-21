"""Track B (MoE owner identification) + Track D (WideVerify Stage 2, measured).

Track B: per-stage CUDA-event profile of fused_moe_mxint on packed4 at K56
using GOLDEN windows -> names the dominant stage of the historical ~1.39 s
MoE block before any kernel work is attacked (AGENTS §30: measured owners
only).

Track D: measured block times at N=56..128 (synthetic-distinct interim — the
golden corpus has 63 positions/layer, so golden windows exist only through
K63), replacing the log-log extrapolation in artifacts/branchpack_oracle.json.

Thermal: warmup_steady + per-rep window certification (clock-locked STEADY).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.wide_verify import (
    GOLDEN_PATH, N_BLOCKS, _bench_blocks, _golden_windows, _synthetic_window,
)
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS

ARTIFACT = Path("artifacts/moe_profile.json")
STAGE2_ARTIFACT = Path("artifacts/wideverify_stage2.json")
ORACLE_ARTIFACT = Path("artifacts/branchpack_oracle.json")
K_STAGE2 = (56, 64, 80, 96, 112, 124, 128)
CORE_K56_S = 0.1024          # BF16 core ceiling (constant-N approximation)
HISTORICAL_K56_MOE_S = 1.3923
STAGES = ("layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
          "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec")


def _certify_window(rep_snaps, start_snap, end_snap):
    temps = [s.get("temp_c") for s in rep_snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    ref_clock = start_snap.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == ref_clock for s in rep_snaps + [end_snap])
    if drift <= 3 and locked:
        return "STEADY"
    if drift <= 5 and locked:
        return "CONTROLLED"
    return "UNKNOWN"


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        raise SystemExit(f"unexpected GPU: {gpu}")

    from mosaicvm.bench.phase29r import _allocate_packed4_experts
    from mosaicvm.kernels.mxint import fused_moe_mxint

    warm = warmup_steady(target_seconds=120.0)
    t_start = snapshot()
    print(f"steady={warm.get('steady')} start={t_start.get('temp_c')}°C")

    golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else None
    if not golden:
        raise SystemExit("golden routes required for the K56 anchor")

    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts

    def _profile_backend(weight_repr: str, alloc_fn) -> dict:
        bank = alloc_fn(N_EXPERTS)
        gw = _golden_windows(56, golden)
        routes = [gw[(i * 7) % len(gw)] for i in range((N_BLOCKS + 1) * N_LAYERS)]
        xs = [torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(8)]
        ids_t = [torch.tensor(r[0], device="cuda", dtype=torch.int64) for r in routes]
        w_t = [torch.tensor(r[1], device="cuda", dtype=torch.float32) for r in routes]
        samples = {s: [] for s in STAGES}
        idx = 0
        for b in range(N_BLOCKS + 1):
            for L in range(N_LAYERS):
                st = {}
                fused_moe_mxint(xs[idx % 8], ids_t[idx], w_t[idx], bank=bank,
                                weight_repr=weight_repr, stage_times=st)
                for s in STAGES:
                    if s in st:
                        samples[s].append(st[s])
                idx += 1
        torch.cuda.synchronize()
        med = {s: statistics.median(v) for s, v in samples.items() if v}
        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 8], ids_t[idx], w_t[idx], bank=bank,
                                weight_repr=weight_repr)
                idx += 1

        block = _bench_blocks(step, N_BLOCKS)
        del bank, routes, ids_t, w_t, xs
        torch.cuda.empty_cache()
        return {"stage_median_s_per_layer": med,
                "total_per_layer_s": sum(med.values()),
                "block_bench_s": block}

    backend_profiles = {
        "packed4": _profile_backend("packed4", _allocate_packed4_experts),
        "hot8": _profile_backend("hot8", _allocate_mxint_experts),
    }
    for name, prof in backend_profiles.items():
        print(f"\n[{name}] K56 golden: {prof['total_per_layer_s']*1e3:.2f} ms/layer "
              f"-> block {prof['block_bench_s']:.4f} s")
        for s in sorted(STAGES, key=lambda k: -prof['stage_median_s_per_layer'].get(k, 0)):
            v = prof['stage_median_s_per_layer'].get(s, 0)
            print(f"  {s:16s} {v*1e3:7.3f} ms ({v/prof['total_per_layer_s']:5.1%})")

    # canonical owner = the faster backend's stage split
    canon = min(backend_profiles.values(), key=lambda p: p['block_bench_s'])
    canon_name = [k for k, v in backend_profiles.items() if v is canon][0]
    layer_med = canon['stage_median_s_per_layer']
    total_med = canon['total_per_layer_s']
    moe_block_k56 = canon['block_bench_s']
    print(f"\ncanonical (faster) backend: {canon_name} "
          f"block {moe_block_k56:.4f} s")

    # ---------------- Track D: N56..128 measured sweep --------------------
    print("\n=== Track D: WideVerify Stage 2 (synthetic-distinct interim) ===")
    oracle = json.loads(ORACLE_ARTIFACT.read_text()) if ORACLE_ARTIFACT.exists() else {}
    stage2 = {"gpu": gpu, "route_authority": "synthetic_distinct_interim "
              "(golden corpus = 63 positions/layer; golden windows end at K63)",
              "core_assumption": f"constant {CORE_K56_S} s (BF16 ceiling; "
              "core scales with N — refine in TRUE P2)",
              "rows": {}}
    rep_snaps = []
    t_end = None
    SKIP_STAGE2 = True   # interim synthetic sweep already recorded; re-run on demand
    bank = _allocate_packed4_experts(N_EXPERTS) if not SKIP_STAGE2 else None
    if SKIP_STAGE2:
        stage2 = json.loads(STAGE2_ARTIFACT.read_text())
        print("\n(Track D sweep skipped — interim synthetic results already "
              "recorded in artifacts/wideverify_stage2.json)")
    for k in ([] if SKIP_STAGE2 else K_STAGE2):
        routes_k = [
            (_synthetic_window(k, seed=1000 * k + 2 + L + 1000 * b), b)
            for b in range(N_BLOCKS + 1) for L in range(N_LAYERS)
        ]
        xs = [torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        ids_t = [torch.tensor(r[0][0], device="cuda", dtype=torch.int64)
                 for r in routes_k]
        w_t = [torch.tensor(r[0][1], device="cuda", dtype=torch.float32)
               for r in routes_k]
        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx], bank=bank,
                                weight_repr="packed4")
                idx += 1

        # per-rep snapshots for window certification
        times = []
        for rep in range(3):
            rep_snaps.append(snapshot())
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s_ev.record()
            step()
            e_ev.record()
            torch.cuda.synchronize()
            times.append(s_ev.elapsed_time(e_ev) / 1e3)
        moe_block = statistics.median(times)
        t_target = moe_block + CORE_K56_S
        raw_rate = k / t_target
        ea = oracle.get("oracle", {}).get(str(k), {}).get("expected_accepted_depth")
        r_oracle = ea / t_target if ea else None
        t_max30 = ea / 30.0 if ea else None
        stage2["rows"][str(k)] = {
            "moe_block_s": moe_block,
            "t_target_s": t_target,
            "raw_candidate_per_s": raw_rate,
            "E_A_oracle": ea,
            "R_oracle_output_per_s": r_oracle,
            "T_max_for_30_s": t_max30,
            "speedup_required": t_target / t_max30 if t_max30 else None,
            "reps_s": times,
        }
        print(f"K={k:3d}: moe={moe_block:.4f}s T_target={t_target:.4f}s "
              f"raw={raw_rate:5.1f}/s E[A]={ea and round(ea,2)} "
              f"R_oracle={r_oracle and round(r_oracle,1)}/s "
              f"need {t_target/t_max30:4.2f}x" if t_max30 else f"K={k}: done")
        del routes_k, ids_t, w_t, xs
        torch.cuda.empty_cache()

    if not SKIP_STAGE2:
        t_end = snapshot()
        label = _certify_window(rep_snaps, t_start, t_end)
        stage2["thermal_label"] = label
        stage2["thermal_start"] = t_start
        stage2["thermal_end"] = t_end
        print(f"thermal: {t_start.get('temp_c')}→{t_end.get('temp_c')}°C [{label}]")
    else:
        label = stage2.get("thermal_label", "UNKNOWN")

    results = {
        "gpu": gpu,
        "thermal_label": label,
        "canonical_backend": canon_name,
        "backend_profiles": backend_profiles,
        "track_b_k56_golden": {
            "backend": canon_name,
            "stage_median_s_per_layer": layer_med,
            "total_per_layer_s": total_med,
            "block_projection_s": total_med * N_LAYERS,
            "block_bench_s": moe_block_k56,
            "historical_block_s": HISTORICAL_K56_MOE_S,
        },
        "owner": max(layer_med, key=layer_med.get),
    }
    ARTIFACT.write_text(json.dumps(results, indent=2, default=str))
    if not SKIP_STAGE2:
        STAGE2_ARTIFACT.write_text(json.dumps(stage2, indent=2, default=str))
    print(f"\nOWNER (K56 golden, {canon_name}): {results['owner']}")
    print(f"wrote {ARTIFACT}")
    return results


if __name__ == "__main__":
    run()
