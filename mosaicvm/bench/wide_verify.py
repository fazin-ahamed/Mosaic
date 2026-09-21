"""Phase 2.10 WideVerify: oracle verification-width sweep (K = 1..32).

Measures, per route source (synthetic-distinct mid, and golden V4 routes when
``artifacts/golden_routes.json`` exists — captured on Modal, never locally):

  * full 43-layer MoE block latency for each backend (QPA / MXINT-Hot8 /
    MXINT-packed4) — the empirical backend portfolio per K;
  * exact-core projection (attention + router + shared expert + hyper-conn
    per layer, + lm_head), reusing the Floor C measurement functions;
  * candidate tok/s = K / (T_moe_block + T_core_block) and K_HW = argmax;
  * routing truth: unique experts, mean/median/p90 assignments per expert,
    backend mix (IMMA vs DP4A vs Hexa share), bytes touched, reuse
    elasticity R(K) = unique / (6K).

Verdict gates (AGENTS.md / Phase 2.10):
  best candidate tok/s >= 35  -> freeze expert compute, next 2.11
  30 .. 35                    -> proceed with optimization reserve
  25 .. 30                    -> borderline; core must compensate
  < 25 at every K             -> reopen MX-LUT32 (fires on golden routes;
                                provisional when golden is unavailable)
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import (
    measure_dense_attention,
    measure_hyper_connection,
    measure_router,
    measure_shared_expert,
    _bench,
)
from mosaicvm.bench.floors import HIDDEN, MOE_INTER, N_EXPERTS, N_LAYERS, TOP_K
from mosaicvm.bench.routing_traces import _family_n_unique, make_trace
from mosaicvm.runtime.memory_budget import BudgetExceeded, budget

K_VALUES_210 = (1, 2, 4, 5, 7, 8, 12, 16, 24, 32)
N_BLOCKS = 10
BACKENDS = ("qpa", "mxint_hot8", "mxint_packed4")
ARTIFACT = Path("artifacts/wide_verify.json")
GOLDEN_PATH = Path("artifacts/golden_routes.json")

# per-expert bytes: GU packed (2*inter*K/2) + down packed (hidden*inter/2)
# + e8m0 scales ((2*inter*K + hidden*inter)/32)
SOURCE_BYTES_PER_EXPERT = (
    (2 * MOE_INTER * HIDDEN) // 2 + (HIDDEN * MOE_INTER) // 2
    + (2 * MOE_INTER * HIDDEN + HIDDEN * MOE_INTER) // 32
)
HOT8_BYTES_PER_EXPERT = (
    (2 * MOE_INTER * HIDDEN) + (HIDDEN * MOE_INTER)
    + (2 * MOE_INTER * HIDDEN + HIDDEN * MOE_INTER) // 32
)


def window_stats(expert_ids) -> dict:
    """Routing truth for one window: expert_ids [positions][top_k]."""
    flat = [int(e) for row in expert_ids for e in row]
    assignments = len(flat)
    counts = Counter(flat)
    unique = len(counts)
    ms = sorted(counts.values())
    mean_m = (assignments / unique) if unique else 0.0
    median_m = float(statistics.median(ms)) if ms else 0.0
    p90_m = float(ms[max(0, math.ceil(0.9 * len(ms)) - 1)]) if ms else 0.0
    r_k = (unique / assignments) if assignments else 0.0
    return {
        "unique": unique,
        "assignments": assignments,
        "mean_m": mean_m,
        "median_m": median_m,
        "p90_m": p90_m,
        "r_k": r_k,
    }


def _backend_mix(ids) -> dict:
    """Share of assignments landing in each fused queue (crossover=3)."""
    flat = [int(e) for row in ids for e in row]
    counts = Counter(flat)
    imma = sum(c for c in counts.values() if c > 3)
    dp4a = len(flat) - imma
    n_rows = len(ids)
    hexa = 6 if n_rows == 1 and len(flat) == 6 else 0
    return {
        "imma_assignments": imma,
        "dp4a_assignments": dp4a,
        "hexa": bool(hexa),
        "imma_share": (imma / len(flat)) if flat else 0.0,
    }


def _synthetic_window(k: int, seed: int):
    n_u = min(_family_n_unique("mid", k, TOP_K), N_EXPERTS)
    tr = make_trace(k, TOP_K, n_unique=n_u, seed=seed)
    return tr["expert_ids"].tolist(), tr["weights"].tolist()


def _golden_windows(k: int, golden: dict):
    """All K-wide consecutive windows from every (capture, layer) sequence."""
    wins = []
    for c in golden.get("captures", []):
        for layer in c.get("layers", []):
            ids = layer["expert_ids"]
            ws = layer["weights"]
            for start in range(0, len(ids) - k + 1):
                wins.append((ids[start:start + k], ws[start:start + k], c["category"]))
    return wins


def _measure_core(k: int, lm_at_max: float, k_max: int, lm_status: str) -> dict:
    attn = measure_dense_attention(k, ctx=4096)["sec_per_step"]
    shared = measure_shared_expert(k)["sec_per_step"]
    hc = measure_hyper_connection(k)["sec_per_step"]
    router = measure_router()["sec_per_step"]
    lm_k = 0.0 if lm_status != "ok" else lm_at_max * (k / k_max)
    per_layer = attn + router + shared + hc
    t_block = per_layer * N_LAYERS + lm_k
    return {
        "K": k,
        "attn_s": attn,
        "router_s": router,
        "shared_s": shared,
        "hc_s": hc,
        "lm_s": lm_k,
        "core_per_layer_s": per_layer,
        "core_block_s": t_block,
    }


def _bench_blocks(step_fn, n_blocks: int) -> float:
    """Median full-block sec over n_blocks (each block times 43 layer calls)."""
    secs = []
    for _ in range(n_blocks + 1):  # +1 warmup block
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        secs.append(start.elapsed_time(end) / 1e3)
    return float(statistics.median(secs[1:]))


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts, _allocate_nk_experts
    from mosaicvm.bench.floor_cd import _try_lm_head
    from mosaicvm.bench.phase29r import _allocate_packed4_experts
    from mosaicvm.kernels.mxint import fused_moe_mxint
    from mosaicvm.kernels.qpa_mma import fused_moe_qpa

    golden = None
    if GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text())
    route_sources = ["synthetic_mid"] + (["golden"] if golden else [])
    golden_status = "ok" if golden else "unavailable_no_capture"

    # --- core projection (local kernels, no model) --------------------------
    k_max = max(K_VALUES_210)
    lm_at_max, lm_status = _try_lm_head(k_max)
    core = {k: _measure_core(k, lm_at_max, k_max, lm_status) for k in K_VALUES_210}
    torch.cuda.empty_cache()

    moe_rows = []
    stats_rows = []

    def _sweep_backend(backend: str, bank, route_source: str, golden_data=None):
        if backend == "qpa":
            def call(x, ids, w):
                return fused_moe_qpa(x, ids, w, bank=bank)
        else:
            repr_ = "hot8" if backend == "mxint_hot8" else "packed4"

            def call(x, ids, w):
                return fused_moe_mxint(x, ids, w, bank=bank, weight_repr=repr_)
        for k in K_VALUES_210:
            # route material for this K
            if route_source == "synthetic_mid":
                per_layer_routes = [
                    _synthetic_window(k, seed=1000 * k + 2 + layer + 1000 * b)
                    for b in range(N_BLOCKS + 1) for layer in range(N_LAYERS)
                ]
                stat_windows = [per_layer_routes[i][0] for i in range(0, len(per_layer_routes), 7)]
            else:
                gw = _golden_windows(k, golden_data)
                if not gw:
                    moe_rows.append({"K": k, "backend": backend, "route_source": route_source,
                                     "status": "no_windows"})
                    continue
                per_layer_routes = [gw[(i * 7) % len(gw)] for i in range((N_BLOCKS + 1) * N_LAYERS)]
                stat_windows = [ids for ids, _, _ in gw]

            st = [window_stats(ids) for ids in stat_windows]
            agg = {
                "K": k,
                "backend": "any",
                "route_source": route_source,
                "unique_median": float(statistics.median(s["unique"] for s in st)),
                "assignments": st[0]["assignments"],
                "mean_m": float(statistics.median(s["mean_m"] for s in st)),
                "median_m": float(statistics.median(s["median_m"] for s in st)),
                "p90_m": float(statistics.median(s["p90_m"] for s in st)),
                "r_k": float(statistics.median(s["r_k"] for s in st)),
                "backend_mix": _backend_mix(stat_windows[0]),
                "bytes_touched_source_median": 0.0,
                "bytes_touched_hot8_median": 0.0,
            }
            agg["bytes_touched_source_median"] = agg["unique_median"] * SOURCE_BYTES_PER_EXPERT
            agg["bytes_touched_hot8_median"] = agg["unique_median"] * HOT8_BYTES_PER_EXPERT
            if route_source == "synthetic_mid" and not any(
                s.get("K") == k for s in stats_rows
            ):
                stats_rows.append(agg)

            # block bench: 43 layer calls per block, routes pre-generated
            idx = 0

            def block_step():
                nonlocal idx
                for _ in range(N_LAYERS):
                    route = per_layer_routes[idx]
                    idx += 1
                    ids_l, w_l = route[0], route[1]
                    x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
                    ids_t = torch.tensor(ids_l, device="cuda", dtype=torch.int64)
                    w_t = torch.tensor(w_l, device="cuda", dtype=torch.float32)
                    call(x, ids_t, w_t)

            try:
                moe_block = _bench_blocks(block_step, N_BLOCKS)
                moe_rows.append({
                    "K": k,
                    "backend": backend,
                    "route_source": route_source,
                    "moe_block_median_sec": moe_block,
                    "moe_per_layer_sec": moe_block / N_LAYERS,
                    "status": "ok",
                })
            except Exception as exc:
                moe_rows.append({"K": k, "backend": backend, "route_source": route_source,
                                 "status": f"error:{type(exc).__name__}",
                                 "error": str(exc)[:200]})
            del per_layer_routes
            torch.cuda.empty_cache()

    # --- backend sweeps (banks allocated sequentially, freed between) ------
    def _alloc(backend):
        budget().reset()
        if backend == "qpa":
            return _allocate_nk_experts(N_EXPERTS)
        if backend == "mxint_hot8":
            return _allocate_mxint_experts(N_EXPERTS)
        return _allocate_packed4_experts(N_EXPERTS)

    for backend in BACKENDS:
        try:
            bank = _alloc(backend)
        except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as exc:
            for k in K_VALUES_210:
                for rs in route_sources:
                    moe_rows.append({"K": k, "backend": backend, "route_source": rs,
                                     "status": f"invalid_not_256:{type(exc).__name__}"})
            continue
        for rs in route_sources:
            _sweep_backend(backend, bank, rs, golden_data=golden)
        del bank
        budget().reset()
        torch.cuda.empty_cache()

    # --- full-block projection + gates --------------------------------------
    full_rows = []
    for rs in route_sources:
        for k in K_VALUES_210:
            cands = {}
            for backend in BACKENDS:
                r = next((m for m in moe_rows if m["K"] == k and m["backend"] == backend
                          and m["route_source"] == rs and m.get("moe_block_median_sec")), None)
                if r:
                    cands[backend] = r["moe_block_median_sec"]
            if not cands:
                continue
            best_backend = min(cands, key=cands.get)
            t_moe = cands[best_backend]
            t_core = core[k]["core_block_s"]
            t_full = t_moe + t_core
            full_rows.append({
                "K": k,
                "route_source": rs,
                "best_backend": best_backend,
                "backend_block_secs": cands,
                "t_moe_block_s": t_moe,
                "t_core_block_s": t_core,
                "t_full_block_s": t_full,
                "candidate_toks": k / t_full,
                "e_k": k / t_full,
            })

    def _best(rs):
        rows = [r for r in full_rows if r["route_source"] == rs]
        if not rows:
            return None
        return max(rows, key=lambda r: r["candidate_toks"])

    best_synth = _best("synthetic_mid")
    best_golden = _best("golden")
    k_hw = best_synth["K"] if best_synth else None
    best_cand = best_synth["candidate_toks"] if best_synth else None
    best_cand_golden = best_cand_g = best_golden["candidate_toks"] if best_golden else None

    # hot8 benefit per K (synthetic)
    hot8_benefit = {}
    for k in K_VALUES_210:
        h = next((m for m in moe_rows if m["K"] == k and m["backend"] == "mxint_hot8"
                  and m["route_source"] == "synthetic_mid" and m.get("moe_block_median_sec")), None)
        p4 = next((m for m in moe_rows if m["K"] == k and m["backend"] == "mxint_packed4"
                   and m["route_source"] == "synthetic_mid" and m.get("moe_block_median_sec")), None)
        if h and p4:
            hot8_benefit[k] = {
                "hot8_block_s": h["moe_block_median_sec"],
                "packed4_block_s": p4["moe_block_median_sec"],
                "hot8_speedup_vs_packed4": p4["moe_block_median_sec"] / h["moe_block_median_sec"],
            }

    # R(K) elasticity curve (synthetic + golden)
    r_k_curve = {}
    for rs in route_sources:
        curve = []
        for k in K_VALUES_210:
            if rs == "synthetic_mid":
                r = next((s for s in stats_rows if s["K"] == k), None)
            else:
                gw = _golden_windows(k, golden) if golden else []
                st = [window_stats(ids) for ids, _, _ in gw] if gw else []
                if not st:
                    continue
                r = {
                    "unique_median": float(statistics.median(s["unique"] for s in st)),
                    "r_k": float(statistics.median(s["r_k"] for s in st)),
                    "mean_m": float(statistics.median(s["mean_m"] for s in st)),
                    "median_m": float(statistics.median(s["median_m"] for s in st)),
                    "p90_m": float(statistics.median(s["p90_m"] for s in st)),
                    "assignments": st[0]["assignments"],
                }
            if r:
                curve.append({"K": k, "unique_median": r["unique_median"],
                              "r_k": r["r_k"], "mean_m": r["mean_m"],
                              "median_m": r["median_m"], "p90_m": r["p90_m"]})
        r_k_curve[rs] = curve

    def _gate(cand):
        if cand is None:
            return "incomplete"
        if cand >= 35:
            return "freeze_expert_compute"
        if cand >= 30:
            return "proceed_with_reserve"
        if cand >= 25:
            return "borderline_core_must_compensate"
        return "miss_reopen_lut"

    gate_synth = _gate(best_cand)
    gate_golden = _gate(best_cand_golden) if best_cand_golden else "incomplete"

    out = {
        "gpu": gpu,
        "k_values": list(K_VALUES_210),
        "n_blocks": N_BLOCKS,
        "backends": list(BACKENDS),
        "n_layers": N_LAYERS,
        "golden_status": golden_status,
        "route_sources": route_sources,
        "core": {str(k): core[k] for k in K_VALUES_210},
        "lm_head": lm_status,
        "moe_rows": moe_rows,
        "stats_rows": stats_rows,
        "full_rows": full_rows,
        "hot8_benefit": {str(k): v for k, v in hot8_benefit.items()},
        "r_k_curve": r_k_curve,
        "gates": {
            "best_candidate_toks_synthetic": best_cand,
            "k_hw_synthetic": k_hw,
            "gate_synthetic": gate_synth,
            "best_candidate_toks_golden": best_cand_g,
            "k_hw_golden": best_golden["K"] if best_golden else None,
            "gate_golden": gate_golden,
            "golden_available": bool(golden),
        },
        "bytes_per_expert": {
            "source_fp4": SOURCE_BYTES_PER_EXPERT,
            "hot8": HOT8_BYTES_PER_EXPERT,
        },
        "vram": budget().peak_summary(),
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "best_candidate_toks_synthetic": None if best_cand is None else round(best_cand, 2),
        "k_hw_synthetic": k_hw,
        "gate_synthetic": gate_synth,
        "best_candidate_toks_golden": None if best_cand_g is None else round(best_cand_g, 2),
        "gate_golden": gate_golden,
        "golden_status": golden_status,
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
