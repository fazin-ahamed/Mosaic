"""Phase 2.10X: verification-width ceiling extension (K > 32).

Stage 1: K40/48/56 on the existing 63-position golden corpus (no recapture).
Stage 2: K64/80/96 on a 160-position Modal-recaptured corpus, run only when
stage-1 improvement K48->K56 >= 5% (AGENTS.md §3).

Protocol identical to Phase 2.10 (wide_verify.py): 3 backends, full 43-layer
block median over 10 blocks, core projection, routing truth, R(K), bytes
touched, backend mix, Hot8 benefit. Merged curve carries K<=32 rows read from
artifacts/wide_verify.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from mosaicvm.bench.floors import N_EXPERTS, N_LAYERS
from mosaicvm.runtime.memory_budget import BudgetExceeded, budget
from mosaicvm.bench.wide_verify import (
    BACKENDS,
    HOT8_BYTES_PER_EXPERT,
    N_BLOCKS,
    SOURCE_BYTES_PER_EXPERT,
    _backend_mix,
    _bench_blocks,
    _golden_windows,
    _measure_core,
    _synthetic_window,
    window_stats,
)

K_STAGE1 = (40, 48, 56)
K_STAGE2 = (64, 80, 96)
ARTIFACT = Path("artifacts/wide_verify_x.json")
BASE_WV = Path("artifacts/wide_verify.json")
GOLDEN_DEFAULT = Path("artifacts/golden_routes.json")
PLATEAU_PCT = 3.5  # midpoint of the 3-5% stop band
RECAPTURE_PCT = 5.0


def run(stage: int = 1,
        golden_path: str | Path = GOLDEN_DEFAULT,
        k_values=None) -> dict:
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

    golden = json.loads(Path(golden_path).read_text())
    n_positions = max(
        (len(l["expert_ids"]) for c in golden.get("captures", []) for l in c.get("layers", [])),
        default=0,
    )
    if k_values is None:
        k_values = K_STAGE1 if stage == 1 else K_STAGE2
    ks = [k for k in k_values if k <= n_positions - 1]
    skipped_ks = [k for k in k_values if k > n_positions - 1]

    # --- core projection for this stage's K --------------------------------
    k_max = max(ks) if ks else 1
    lm_at_max, lm_status = _try_lm_head(k_max)
    core = {k: _measure_core(k, lm_at_max, k_max, lm_status) for k in ks}
    torch.cuda.empty_cache()

    moe_rows = []
    stats_rows = []

    def _sweep_backend(backend: str, bank):
        if backend == "qpa":
            def call(x, ids, w):
                return fused_moe_qpa(x, ids, w, bank=bank)
        else:
            repr_ = "hot8" if backend == "mxint_hot8" else "packed4"

            def call(x, ids, w):
                return fused_moe_mxint(x, ids, w, bank=bank, weight_repr=repr_)
        for k in ks:
            gw = _golden_windows(k, golden)
            if not gw:
                moe_rows.append({"K": k, "backend": backend, "status": "no_windows"})
                continue
            per_layer_routes = [gw[(i * 7) % len(gw)] for i in range((N_BLOCKS + 1) * N_LAYERS)]
            if not any(s.get("K") == k for s in stats_rows):
                st = [window_stats(ids) for ids, _, _ in gw]
                uniq_med = float(statistics.median(s["unique"] for s in st))
                stats_rows.append({
                    "K": k,
                    "corpus_positions": n_positions,
                    "unique_median": uniq_med,
                    "assignments": st[0]["assignments"],
                    "mean_m": float(statistics.median(s["mean_m"] for s in st)),
                    "median_m": float(statistics.median(s["median_m"] for s in st)),
                    "p90_m": float(statistics.median(s["p90_m"] for s in st)),
                    "r_k": float(statistics.median(s["r_k"] for s in st)),
                    "backend_mix": _backend_mix(gw[0][0]),
                    "bytes_touched_source_median": uniq_med * SOURCE_BYTES_PER_EXPERT,
                    "bytes_touched_hot8_median": uniq_med * HOT8_BYTES_PER_EXPERT,
                })
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
                    call(x, ids_t, w_t)

            try:
                moe_block = _bench_blocks(block_step, N_BLOCKS)
                moe_rows.append({
                    "K": k, "backend": backend, "corpus_positions": n_positions,
                    "moe_block_median_sec": moe_block,
                    "moe_per_layer_sec": moe_block / N_LAYERS,
                    "status": "ok",
                })
            except Exception as exc:
                moe_rows.append({"K": k, "backend": backend, "status": f"error:{type(exc).__name__}",
                                 "error": str(exc)[:200]})
            del per_layer_routes
            torch.cuda.empty_cache()

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
            for k in ks:
                moe_rows.append({"K": k, "backend": backend,
                                 "status": f"invalid_not_256:{type(exc).__name__}"})
            continue
        _sweep_backend(backend, bank)
        del bank
        budget().reset()
        torch.cuda.empty_cache()

    # --- full-block projection ---------------------------------------------
    full_rows = []
    for k in ks:
        cands = {}
        for backend in BACKENDS:
            r = next((m for m in moe_rows if m["K"] == k and m["backend"] == backend
                      and m.get("moe_block_median_sec")), None)
            if r:
                cands[backend] = r["moe_block_median_sec"]
        if not cands:
            continue
        best_backend = min(cands, key=cands.get)
        t_moe = cands[best_backend]
        t_core = core[k]["core_block_s"]
        full_rows.append({
            "K": k,
            "corpus_positions": n_positions,
            "best_backend": best_backend,
            "backend_block_secs": cands,
            "t_moe_block_s": t_moe,
            "t_core_block_s": t_core,
            "t_full_block_s": t_moe + t_core,
            "candidate_toks": k / (t_moe + t_core),
        })

    # --- merge with 2.10 K<=32 golden curve ---------------------------------
    merged = []
    if BASE_WV.exists():
        base = json.loads(BASE_WV.read_text())
        for r in (base.get("full_rows") or []):
            if r.get("route_source") == "golden":
                merged.append({"K": r["K"], "candidate_toks": r["candidate_toks"],
                               "corpus_positions": 63, "source": "wide_verify"})
    for r in full_rows:
        merged.append({"K": r["K"], "candidate_toks": r["candidate_toks"],
                       "corpus_positions": n_positions, "source": "wide_verify_x"})
    merged.sort(key=lambda r: r["K"])

    # --- gates (AGENTS.md §3) ------------------------------------------------
    wide_rows = [r for r in full_rows if r["K"] > 32]
    best_wide = max((r["candidate_toks"] for r in wide_rows), default=None)
    best_wide_k = next((r["K"] for r in wide_rows if r["candidate_toks"] == best_wide), None) if wide_rows else None
    crossed_30 = bool(best_wide is not None and best_wide >= 30.0)
    crossed_35 = bool(best_wide is not None and best_wide >= 35.0)

    improvements = {}
    for a, b in zip(merged, merged[1:]):
        if a["candidate_toks"] and b["candidate_toks"]:
            improvements[f"{a['K']}->{b['K']}"] = 100.0 * (b["candidate_toks"] / a["candidate_toks"] - 1.0)

    # recapture: last stage-1 increment >= 5%
    recapture_recommended = False
    if stage == 1:
        imp_48_56 = improvements.get("48->56")
        recapture_recommended = bool(imp_48_56 is not None and imp_48_56 >= RECAPTURE_PCT)

    # plateau (widening stop rule): <3.5% for two consecutive increments.
    # Only increments ENDING inside this stage's measured K range count —
    # historical K<=32 improvements belong to Phase 2.10, not to the
    # decision whether to keep widening past the current frontier.
    ks_set = set(ks)
    extension_increments = [
        v for key, v in improvements.items()
        if int(key.split("->")[1]) in ks_set
    ]
    consecutive_low = 0
    plateau = False
    for v in extension_increments:
        if v < PLATEAU_PCT:
            consecutive_low += 1
            if consecutive_low >= 2:
                plateau = True
                break
        else:
            consecutive_low = 0

    k_hw = max(merged, key=lambda r: r["candidate_toks"])["K"] if merged else None

    out = {
        "gpu": gpu,
        "stage": stage,
        "k_values_requested": list(k_values),
        "k_values_measured": ks,
        "k_values_skipped_short_corpus": skipped_ks,
        "golden_path": str(golden_path),
        "corpus_positions": n_positions,
        "core": {str(k): core[k] for k in ks},
        "lm_head": lm_status,
        "moe_rows": moe_rows,
        "stats_rows": stats_rows,
        "full_rows": full_rows,
        "merged_curve": merged,
        "improvements_pct": improvements,
        "gates": {
            "best_wide_candidate_toks": best_wide,
            "best_wide_k": best_wide_k,
            "crossed_30_wide": crossed_30,
            "crossed_35_wide": crossed_35,
            "k_hw": k_hw,
            "plateau": plateau,
            "recapture_recommended": recapture_recommended,
        },
        "vram": budget().peak_summary(),
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "stage": stage,
        "k_measured": ks,
        "best_wide_candidate_toks": None if best_wide is None else round(best_wide, 2),
        "best_wide_k": best_wide_k,
        "crossed_30_wide": crossed_30,
        "crossed_35_wide": crossed_35,
        "k_hw": k_hw,
        "plateau": plateau,
        "recapture_recommended": recapture_recommended,
        "improvements_pct": {k: round(v, 2) for k, v in improvements.items()},
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    import sys
    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gp = sys.argv[2] if len(sys.argv) > 2 else str(GOLDEN_DEFAULT)
    run(stage=stage, golden_path=gp)
