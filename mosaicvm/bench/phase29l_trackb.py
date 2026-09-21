"""Phase 2.9L Track B: exact epilogue fusion race on golden full-MoE blocks.

Authoritative metric per AGENTS.md §10: golden-route full 43-layer MoE block
latency. Compares epilogue="torch" (frozen reference path) vs epilogue="fused"
(mxint_swiglu_quant_down + mxint_routed_reduce, one pass each) at the WideVerify
widths. Gates: >=3% useful · >=5% useful · >=8% strong · >=12% likely enough
for raw 40 at K56.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from mosaicvm.bench.floors import N_EXPERTS, N_LAYERS, TOP_K
from mosaicvm.bench.wide_verify import _golden_windows, _bench_blocks
from mosaicvm.runtime.memory_budget import budget

K_RACE = (32, 40, 48, 56)
N_BLOCKS = 10
ARTIFACT = Path("artifacts/phase29l_trackb.json")
GOLDEN_PATH = Path("artifacts/golden_routes.json")


def _load(name: str):
    p = Path("artifacts") / name
    return json.loads(p.read_text()) if p.exists() else None


def _core_block_sec(k: int):
    """Core projection from the WideVerify artifacts (measured minutes ago)."""
    wvx = _load("wide_verify_x.json") or {}
    core = wvx.get("core") or {}
    if str(k) in core:
        return core[str(k)]["core_block_s"]
    wv = _load("wide_verify.json") or {}
    core = wv.get("core") or {}
    if str(k) in core:
        return core[str(k)]["core_block_s"]
    return None


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
    from mosaicvm.bench.phase29r import _allocate_packed4_experts as _alloc_p4
    from mosaicvm.kernels.mxint import fused_moe_mxint

    golden = json.loads(GOLDEN_PATH.read_text())
    rows = []

    def _race(bank, repr_, label):
        for k in K_RACE:
            gw = _golden_windows(k, golden)
            if not gw:
                continue
            per_layer_routes = [gw[(i * 7) % len(gw)] for i in range((N_BLOCKS + 1) * N_LAYERS)]
            rec = {"K": k, "backend": label, "weight_repr": repr_}

            def _block(epi):
                idx = 0

                def step():
                    nonlocal idx
                    for _ in range(N_LAYERS):
                        route = per_layer_routes[idx]
                        idx += 1
                        ids_l, w_l = route[0], route[1]
                        x = torch.randn(k, 4096, dtype=torch.bfloat16, device="cuda")
                        ids_t = torch.tensor(ids_l, device="cuda", dtype=torch.int64)
                        w_t = torch.tensor(w_l, device="cuda", dtype=torch.float32)
                        fused_moe_mxint(x, ids_t, w_t, bank=bank,
                                        weight_repr=repr_, epilogue=epi)

                return step

            # warmup both paths
            for epi in ("torch", "fused"):
                torch.cuda.synchronize()
                _block(epi)()
                torch.cuda.synchronize()
            # interleaved A/B: alternating blocks cancel clock/thermal drift
            secs = {"torch": [], "fused": []}
            for _ in range(N_BLOCKS):
                for epi in ("torch", "fused"):
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    _block(epi)()
                    e.record()
                    torch.cuda.synchronize()
                    secs[epi].append(s.elapsed_time(e) / 1e3)
            rec["moe_block_torch_sec"] = float(statistics.median(secs["torch"]))
            rec["moe_block_fused_sec"] = float(statistics.median(secs["fused"]))
            rec["torch_blocks_s"] = secs["torch"]
            rec["fused_blocks_s"] = secs["fused"]
            rec["improvement_pct"] = 100.0 * (
                rec["moe_block_torch_sec"] / rec["moe_block_fused_sec"] - 1.0)
            core = _core_block_sec(k)
            if core is not None:
                rec["core_block_sec"] = core
                rec["cand_toks_torch"] = k / (rec["moe_block_torch_sec"] + core)
                rec["cand_toks_fused"] = k / (rec["moe_block_fused_sec"] + core)
            rows.append(rec)
            print(f"[{label}] K{k}: torch {rec['moe_block_torch_sec']*1000:.1f} ms -> "
                  f"fused {rec['moe_block_fused_sec']*1000:.1f} ms "
                  f"(+{rec['improvement_pct']:.1f}%)"
                  + (f" cand {rec['cand_toks_torch']:.2f} -> {rec['cand_toks_fused']:.2f}"
                     if core is not None else ""))
            del per_layer_routes
            torch.cuda.empty_cache()

    # Hot8 (speed ceiling)
    try:
        budget().reset()
        hot8 = _allocate_mxint_experts(N_EXPERTS)
        _race(hot8, "hot8", "mxint_hot8")
        del hot8
    except Exception as exc:
        rows.append({"backend": "mxint_hot8", "status": f"error:{type(exc).__name__}"})
    budget().reset()
    torch.cuda.empty_cache()

    # packed4 at the widest K (residency-relevant representation)
    try:
        p4 = _alloc_p4(N_EXPERTS)
        for k in (48, 56):
            gw = _golden_windows(k, golden)
            if not gw:
                continue
            per_layer_routes = [gw[(i * 7) % len(gw)] for i in range((N_BLOCKS + 1) * N_LAYERS)]
            rec = {"K": k, "backend": "mxint_packed4", "weight_repr": "packed4"}
            for epi in ("torch", "fused"):
                idx = 0

                def block_step():
                    nonlocal idx
                    for _ in range(N_LAYERS):
                        route = per_layer_routes[idx]
                        idx += 1
                        ids_l, w_l = route[0], route[1]
                        x = torch.randn(k, 4096, dtype=torch.bfloat16, device="cuda")
                        ids_t = torch.tensor(ids_l, device="cuda", dtype=torch.int64)
                        w_t = torch.tensor(w_l, device="cuda", dtype=torch.float32)
                        fused_moe_mxint(x, ids_t, w_t, bank=p4,
                                        weight_repr="packed4", epilogue=epi)

                rec[f"moe_block_{epi}_sec"] = _bench_blocks(block_step, N_BLOCKS)
            rec["improvement_pct"] = 100.0 * (
                rec["moe_block_torch_sec"] / rec["moe_block_fused_sec"] - 1.0)
            core = _core_block_sec(k)
            if core is not None:
                rec["core_block_sec"] = core
                rec["cand_toks_torch"] = k / (rec["moe_block_torch_sec"] + core)
                rec["cand_toks_fused"] = k / (rec["moe_block_fused_sec"] + core)
            rows.append(rec)
            print(f"[mxint_packed4] K{k}: torch {rec['moe_block_torch_sec']*1000:.1f} ms -> "
                  f"fused {rec['moe_block_fused_sec']*1000:.1f} ms "
                  f"(+{rec['improvement_pct']:.1f}%)"
                  + (f" cand {rec['cand_toks_torch']:.2f} -> {rec['cand_toks_fused']:.2f}"
                     if core is not None else ""))
            del per_layer_routes
            torch.cuda.empty_cache()
        del p4
    except Exception as exc:
        rows.append({"backend": "mxint_packed4", "status": f"error:{type(exc).__name__}"})
    budget().reset()
    torch.cuda.empty_cache()

    # gates
    best_row = max((r for r in rows if r.get("improvement_pct") is not None),
                   key=lambda r: r["improvement_pct"], default=None)
    best_cand = max((r.get("cand_toks_fused") for r in rows
                     if r.get("cand_toks_fused") is not None), default=None)
    imp = best_row["improvement_pct"] if best_row else None
    if imp is None:
        gate = "incomplete"
    elif imp >= 12.0:
        gate = "strong_likely_raw40"
    elif imp >= 8.0:
        gate = "strong"
    elif imp >= 5.0:
        gate = "useful"
    elif imp >= 3.0:
        gate = "marginally_useful"
    else:
        gate = "no_gain"
    expert40 = bool(best_cand is not None and best_cand >= 40.0)

    out = {
        "gpu": gpu,
        "track": "B_exact_epilogue_fusion",
        "k_values": list(K_RACE),
        "n_blocks": N_BLOCKS,
        "rows": rows,
        "gates": {
            "best_improvement_pct": imp,
            "best_improvement_k": best_row["K"] if best_row else None,
            "gate": gate,
            "best_cand_toks_fused": best_cand,
            "expert40_pass": expert40,
            "thresholds": {"useful": 3.0, "useful2": 5.0, "strong": 8.0,
                           "likely_raw40": 12.0},
        },
        "correctness": "BITWISE_MOSAIC_REFERENCE (tests/test_mxint_epilogue.py: "
                       "fused == torch at every stage incl. e4m3 encoder, silu, "
                       "SwiGLU+requant planes/sA, routed reduction, full fused)",
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["gates"], indent=2))
    return out


if __name__ == "__main__":
    run()
