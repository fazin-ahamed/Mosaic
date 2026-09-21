"""Phase 2.9R: P1 fused MXINT Floor B — 256 experts, Hot8 / packed4 / hybrid.

Kernel-only (P0) numbers live in artifacts/qpa_race.json. This driver measures
the fused path (P1): act_quant + qA integer planes + routing/layout + GU +
SwiGLU + down quant + DOWN + reduction all inside the timer. Weight banks are
prebuilt outside the timer (resident cache); packed4 decodes E2M1->qW in-kernel
so its weight encode IS inside the timer.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.bench.floors import HIDDEN, MOE_INTER, N_EXPERTS, N_LAYERS, TOP_K
from mosaicvm.bench.routing_traces import _family_n_unique, make_trace
from mosaicvm.runtime.memory_budget import BudgetExceeded, budget

MXINT_K = (1, 2, 4, 8, 12, 16)
FAMILY = "mid"
ARTIFACT = Path("artifacts/phase29r.json")


def _allocate_packed4_experts(n_e: int = N_EXPERTS, chunk: int = 32):
    """256 distinct experts as an MXINTPacked4Bank (source-packed, no encode)."""
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
    from mosaicvm.kernels.mxint import MXINTPacked4Bank

    p_gu = torch.empty(n_e, 2 * MOE_INTER, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    s_gu = torch.empty(n_e, 2 * MOE_INTER, HIDDEN // 32, dtype=torch.uint8, device="cuda")
    p_d = torch.empty(n_e, HIDDEN, MOE_INTER // 2, dtype=torch.uint8, device="cuda")
    s_d = torch.empty(n_e, HIDDEN, MOE_INTER // 32, dtype=torch.uint8, device="cuda")
    for i0 in range(0, n_e, chunk):
        i1 = min(n_e, i0 + chunk)
        for j in range(i1 - i0):
            wg = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pg, sg = pack_mxfp4_nk(wg)
            del wg
            wu = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pu, su = pack_mxfp4_nk(wu)
            del wu
            wd = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
            pd, sd = pack_mxfp4_nk(wd)
            del wd
            p_gu[i0 + j, :MOE_INTER] = pg
            s_gu[i0 + j, :MOE_INTER] = sg.view(torch.uint8)
            p_gu[i0 + j, MOE_INTER:] = pu
            s_gu[i0 + j, MOE_INTER:] = su.view(torch.uint8)
            p_d[i0 + j] = pd
            s_d[i0 + j] = sd.view(torch.uint8)
            del pg, sg, pu, su, pd, sd
        torch.cuda.empty_cache()
    b = budget()
    b.charge_device(p_gu, "r.p_gu")
    b.charge_device(s_gu, "r.s_gu")
    b.charge_device(p_d, "r.p_d")
    b.charge_device(s_d, "r.s_d")
    return MXINTPacked4Bank(p_gu=p_gu, s_gu=s_gu, p_d=p_d, s_d=s_d,
                            n_out_gu=2 * MOE_INTER, n_out_d=HIDDEN)


def _allocate_hybrid_experts(n_e: int = N_EXPERTS, chunk: int = 32):
    """One pass over random NK filling BOTH the Hot8 and packed4 banks."""
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
    from mosaicvm.kernels.mxint import MXINTExpertBank, MXINTHybridBank, MXINTPacked4Bank, qint_encode_w

    w_gu = torch.empty(n_e, 2 * MOE_INTER, HIDDEN, dtype=torch.int8, device="cuda")
    s_gu = torch.empty(n_e, 2 * MOE_INTER, HIDDEN // 32, dtype=torch.uint8, device="cuda")
    w_d = torch.empty(n_e, HIDDEN, MOE_INTER, dtype=torch.int8, device="cuda")
    s_d = torch.empty(n_e, HIDDEN, MOE_INTER // 32, dtype=torch.uint8, device="cuda")
    p_gu = torch.empty(n_e, 2 * MOE_INTER, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    p_d = torch.empty(n_e, HIDDEN, MOE_INTER // 2, dtype=torch.uint8, device="cuda")
    for i0 in range(0, n_e, chunk):
        i1 = min(n_e, i0 + chunk)
        for j in range(i1 - i0):
            wg = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pg, sg = pack_mxfp4_nk(wg)
            del wg
            wu = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pu, su = pack_mxfp4_nk(wu)
            del wu
            wd = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
            pd, sd = pack_mxfp4_nk(wd)
            del wd
            w_gu[i0 + j, :MOE_INTER] = qint_encode_w(pg)
            w_gu[i0 + j, MOE_INTER:] = qint_encode_w(pu)
            s_gu[i0 + j, :MOE_INTER] = sg.view(torch.uint8)
            s_gu[i0 + j, MOE_INTER:] = su.view(torch.uint8)
            w_d[i0 + j] = qint_encode_w(pd)
            s_d[i0 + j] = sd.view(torch.uint8)
            p_gu[i0 + j, :MOE_INTER] = pg
            p_gu[i0 + j, MOE_INTER:] = pu
            p_d[i0 + j] = pd
            del pg, sg, pu, su, pd, sd
        torch.cuda.empty_cache()
    b = budget()
    b.charge_device(w_gu, "r.h.w_gu")
    b.charge_device(s_gu, "r.h.s_gu")
    b.charge_device(w_d, "r.h.w_d")
    b.charge_device(s_d, "r.h.s_d")
    b.charge_device(p_gu, "r.h.p_gu")
    b.charge_device(p_d, "r.h.p_d")
    hot8 = MXINTExpertBank(w_gu=w_gu, s_gu=s_gu, w_d=w_d, s_d=s_d,
                           n_out_gu=2 * MOE_INTER, n_out_d=HIDDEN)
    packed4 = MXINTPacked4Bank(p_gu=p_gu, s_gu=s_gu, p_d=p_d, s_d=s_d,
                               n_out_gu=2 * MOE_INTER, n_out_d=HIDDEN)
    return MXINTHybridBank(hot8=hot8, packed4=packed4, hot_mask=None)


def _row(k, sec, unique, mean_m, status="ok", extra=None):
    rec = {
        "level": "P1",
        "k": k,
        "family": FAMILY,
        "sec_per_moe_layer": sec,
        "unique_experts": unique,
        "mean_m_per_expert": mean_m,
        "status": status,
    }
    if extra:
        rec.update(extra)
    return rec


def _trace(k):
    n_u = min(_family_n_unique(FAMILY, k, TOP_K), N_EXPERTS)
    return make_trace(k, TOP_K, n_unique=n_u, seed=1000 * k + 2), n_u


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts, _allocate_nk_experts
    from mosaicvm.kernels.mxint import fused_moe_mxint, last_launch_count as mxint_launches
    from mosaicvm.kernels.qpa_mma import fused_moe_qpa, launch_census

    rows = []
    stage_times_k8_hot8 = None
    census = None
    s3_mxint_block_median_sec = None
    s3_mxint_block_p90_sec = None

    def _sweep(backend, weight_repr, bank, stage_on_k8=False):
        nonlocal stage_times_k8_hot8
        out_rows = []
        for k in MXINT_K:
            tr, n_u = _trace(k)
            ids = tr["expert_ids"].to(device="cuda")
            if int(ids.max().item()) >= N_EXPERTS:
                out_rows.append({"backend": backend, **_row(k, None, 0, 0.0, status="invalid_not_256")})
                continue
            weights = tr["weights"].to(device="cuda", dtype=torch.float32)
            x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
            flat = ids.reshape(-1)
            counts = torch.bincount(flat, minlength=N_EXPERTS)
            used = counts[counts > 0].float()
            unique = int(flat.unique().numel())
            mean_m = float(used.mean().item()) if used.numel() else 0.0
            st = {} if (stage_on_k8 and k == 8) else None
            try:
                def step(xx=x, ii=ids, ww=weights, bb=bank, ss=st):
                    return fused_moe_mxint(xx, ii, ww, bank=bb, weight_repr=weight_repr, stage_times=ss)
                sec = _bench(step)
                step()
                rec = {"backend": backend, **_row(k, sec, unique, mean_m)}
                if st:
                    stage_times_k8_hot8 = dict(st)
                    rec["stage_times"] = dict(st)
                out_rows.append(rec)
            except Exception as exc:
                out_rows.append({"backend": backend,
                                 **_row(k, None, unique, mean_m, status=f"error:{type(exc).__name__}",
                                        extra={"error": str(exc)[:200]})})
            del x, ids, weights
        return out_rows

    # --- QPA reference P1 -------------------------------------------------
    qpa_rows = [{"backend": "fused_moe_qpa", **_row(k, None, 0, 0.0, status="skipped")}
                for k in MXINT_K]
    try:
        budget().reset()
        qpa_bank = _allocate_nk_experts(N_EXPERTS)
        for k in MXINT_K:
            tr, n_u = _trace(k)
            ids = tr["expert_ids"].to(device="cuda")
            weights = tr["weights"].to(device="cuda", dtype=torch.float32)
            x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
            flat = ids.reshape(-1)
            counts = torch.bincount(flat, minlength=N_EXPERTS)
            used = counts[counts > 0].float()
            unique = int(flat.unique().numel())
            mean_m = float(used.mean().item()) if used.numel() else 0.0
            try:
                def step_q(xx=x, ii=ids, ww=weights, bb=qpa_bank):
                    return fused_moe_qpa(xx, ii, ww, bank=bb)
                sec = _bench(step_q)
                qpa_rows[MXINT_K.index(k)] = {
                    "backend": "fused_moe_qpa", "weight_repr": "qpa_fragment",
                    **_row(k, sec, unique, mean_m)}
            except Exception as exc:
                qpa_rows[MXINT_K.index(k)] = {
                    "backend": "fused_moe_qpa",
                    **_row(k, None, unique, mean_m, status=f"error:{type(exc).__name__}")}
            del x, ids, weights
        del qpa_bank
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError):
        qpa_rows = [{"backend": "fused_moe_qpa", **_row(k, None, 0, 0.0, status="invalid_not_256")}
                    for k in MXINT_K]
    budget().reset()
    torch.cuda.empty_cache()
    rows.extend(qpa_rows)

    # --- Hot8 P1 -----------------------------------------------------------
    hot8_rows = [{"backend": "fused_moe_mxint", "weight_repr": "hot8",
                  **_row(k, None, 0, 0.0, status="skipped")} for k in MXINT_K]
    hot8_status = "ok"
    try:
        hot8_bank = _allocate_mxint_experts(N_EXPERTS)
        hot8_rows = _sweep("fused_moe_mxint", "hot8", hot8_bank, stage_on_k8=True)
        # census on K8 mid
        tr, _ = _trace(8)
        ids = tr["expert_ids"].to(device="cuda")
        weights = tr["weights"].to(device="cuda", dtype=torch.float32)
        x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def _census_step():
            return fused_moe_mxint(x, ids, weights, bank=hot8_bank, weight_repr="hot8")

        census = launch_census(_census_step)
        del x, ids, weights
        # 48-block S3, Hot8 only, K8 mid
        golden_path = Path("artifacts/golden_routes.json")
        golden = json.loads(golden_path.read_text()) if golden_path.exists() else None
        s3_route_authority = "golden" if golden and (golden.get("windows") or golden.get("traces")) \
            else "synthetic_distinct_interim"
        n_u_mid = min(_family_n_unique(FAMILY, 8, TOP_K), N_EXPERTS)
        tr0 = make_trace(8, TOP_K, n_unique=n_u_mid, seed=0)
        fused_moe_mxint(torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                        tr0["expert_ids"].to(device="cuda"),
                        tr0["weights"].to(device="cuda", dtype=torch.float32),
                        bank=hot8_bank, weight_repr="hot8")
        torch.cuda.synchronize()
        block_secs = []
        for bidx in range(48):
            acc = 0.0
            for layer in range(N_LAYERS):
                if s3_route_authority == "golden":
                    wins = golden.get("windows") or golden.get("traces") or []
                    w = wins[(bidx * N_LAYERS + layer) % max(len(wins), 1)] if wins else None
                    if w:
                        ids = torch.tensor(w["expert_ids"], device="cuda", dtype=torch.int64)
                        if ids.dim() == 1:
                            ids = ids.view(1, -1).expand(8, -1)
                        weights = torch.tensor(w["weights"], device="cuda", dtype=torch.float32)
                    else:
                        tr = make_trace(8, TOP_K, n_unique=n_u_mid, seed=10_000 * bidx + layer)
                        ids = tr["expert_ids"].to(device="cuda")
                        weights = tr["weights"].to(device="cuda", dtype=torch.float32)
                else:
                    tr = make_trace(8, TOP_K, n_unique=n_u_mid, seed=10_000 * bidx + layer)
                    ids = tr["expert_ids"].to(device="cuda")
                    weights = tr["weights"].to(device="cuda", dtype=torch.float32)
                x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fused_moe_mxint(x, ids, weights, bank=hot8_bank, weight_repr="hot8")
                end.record()
                torch.cuda.synchronize()
                acc += start.elapsed_time(end) / 1e3
                del x, ids, weights
            block_secs.append(acc)
        bt = torch.tensor(block_secs)
        s3_mxint_block_median_sec = float(bt.median())
        s3_mxint_block_p90_sec = float(bt.kthvalue(max(1, int(round(0.9 * 48)))).values)
        del hot8_bank
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as exc:
        hot8_status = f"invalid_not_256:{type(exc).__name__}"
        hot8_rows = [{"backend": "fused_moe_mxint", "weight_repr": "hot8",
                      **_row(k, None, 0, 0.0, status="invalid_not_256")} for k in MXINT_K]
    budget().reset()
    torch.cuda.empty_cache()
    rows.extend(hot8_rows)

    # --- packed4 P1 ---------------------------------------------------------
    packed4_rows = [{"backend": "fused_moe_mxint", "weight_repr": "packed4",
                     **_row(k, None, 0, 0.0, status="skipped")} for k in MXINT_K]
    try:
        p4_bank = _allocate_packed4_experts(N_EXPERTS)
        packed4_rows = _sweep("fused_moe_mxint", "packed4", p4_bank)
        del p4_bank
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError):
        packed4_rows = [{"backend": "fused_moe_mxint", "weight_repr": "packed4",
                         **_row(k, None, 0, 0.0, status="invalid_not_256")} for k in MXINT_K]
    budget().reset()
    torch.cuda.empty_cache()
    rows.extend(packed4_rows)

    # --- hybrid P1 ----------------------------------------------------------
    hybrid_rows = [{"backend": "fused_moe_mxint", "weight_repr": "hybrid",
                    **_row(k, None, 0, 0.0, status="skipped")} for k in MXINT_K]
    hot8_bytes = N_EXPERTS * (2 * MOE_INTER * HIDDEN + HIDDEN * MOE_INTER) \
        + N_EXPERTS * (2 * MOE_INTER * HIDDEN // 32 + HIDDEN * MOE_INTER // 32)
    packed4_bytes = N_EXPERTS * (2 * MOE_INTER * HIDDEN // 2 + HIDDEN * MOE_INTER // 2) \
        + N_EXPERTS * (2 * MOE_INTER * HIDDEN // 32 + HIDDEN * MOE_INTER // 32)
    try:
        hyb_bank = _allocate_hybrid_experts(N_EXPERTS)
        hybrid_rows = _sweep("fused_moe_mxint", "hybrid", hyb_bank)
        for r in hybrid_rows:
            r["hot8_bytes"] = hot8_bytes
            r["packed4_bytes"] = packed4_bytes
        del hyb_bank
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError):
        hybrid_rows = [{"backend": "fused_moe_mxint", "weight_repr": "hybrid",
                        **_row(k, None, 0, 0.0, status="invalid_not_256")} for k in MXINT_K]
    budget().reset()
    torch.cuda.empty_cache()
    rows.extend(hybrid_rows)

    # --- integrate ----------------------------------------------------------
    def _sec(rs, k):
        r = next((x for x in rs if x.get("k") == k and x.get("sec_per_moe_layer")), None)
        return None if r is None else r["sec_per_moe_layer"]

    qpa_k8 = _sec(qpa_rows, 8)
    hot8_k8 = _sec(hot8_rows, 8)
    p4_k8 = _sec(packed4_rows, 8)
    hyb_k8 = _sec(hybrid_rows, 8)
    hot8_ratio = (qpa_k8 / hot8_k8) if (qpa_k8 and hot8_k8) else None
    p4_ratio = (qpa_k8 / p4_k8) if (qpa_k8 and p4_k8) else None

    race = {}
    race_path = Path("artifacts/qpa_race.json")
    if race_path.exists():
        race = json.loads(race_path.read_text())

    out = {
        "gpu": gpu,
        "n_experts": N_EXPERTS,
        "k_values": list(MXINT_K),
        "family": FAMILY,
        "levels": {"P0": "isolated kernel (qpa_race.json)", "P1": "one resident MoE layer (this file)"},
        "p0": race.get("mxint_vs_qpa"),
        "p0_packed4_vs_hot8": race.get("mxint_packed4_vs_hot8"),
        "integrated": {
            "qpa_k8_mid_sec": qpa_k8,
            "hot8_k8_mid_sec": hot8_k8,
            "packed4_k8_mid_sec": p4_k8,
            "hybrid_k8_mid_sec": hyb_k8,
            "hot8_ratio": hot8_ratio,
            "packed4_ratio": p4_ratio,
        },
        "hot8_status": hot8_status,
        "s3_mxint_block_median_sec": s3_mxint_block_median_sec,
        "s3_mxint_block_p90_sec": s3_mxint_block_p90_sec,
        "s3_route_authority": s3_route_authority if s3_mxint_block_median_sec is not None else "not_run",
        "stage_times_k8_hot8": stage_times_k8_hot8,
        "launch_census": census,
        "hot8_bytes": hot8_bytes,
        "packed4_bytes": packed4_bytes,
        "vram": budget().peak_summary(),
        "rows": rows,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "qpa_k8_mid_sec": qpa_k8,
        "hot8_k8_mid_sec": hot8_k8,
        "packed4_k8_mid_sec": p4_k8,
        "hybrid_k8_mid_sec": hyb_k8,
        "hot8_ratio": hot8_ratio,
        "packed4_ratio": p4_ratio,
        "s3_mxint_block_median_sec": s3_mxint_block_median_sec,
        "hot8_status": hot8_status,
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
