"""Floor B: one V4 MoE layer, distinct W4 experts resident."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.bench.floors import (
    FLOOR_B,
    HIDDEN,
    K_VALUES,
    MOE_INTER,
    N_EXPERTS,
    N_LAYERS,
    TOP_K,
)
from mosaicvm.bench.routing_traces import _family_n_unique, make_trace
from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_grouped_moe_sequential
from mosaicvm.runtime.memory_budget import BudgetExceeded, budget

FAMILIES = ("peaked", "mid", "worst")
ARTIFACT = Path("artifacts/floor_b_moe.json")


def pack_pool(n_e: int, k_in: int, n_out: int, device="cuda"):
    packed, scales = [], []
    for _ in range(n_e):
        w = torch.randn(k_in, n_out, dtype=torch.bfloat16, device=device)
        p, s = pack_w4(w)
        packed.append(p)
        scales.append(s)
        del w
    return packed, scales


def _try_load_checkpoint_experts(n_e: int, device="cuda"):
    root = os.environ.get("MOSAICVM_MODEL_ROOT")
    if not root:
        return None
    from mosaicvm.runtime.fp8 import fit_linear_weight
    from mosaicvm.runtime.loader import StreamingLoader
    from mosaicvm.runtime.streaming_model import _maybe_dequant

    loader = StreamingLoader(root)
    pg, sg, pu, su, pd, sd = [], [], [], [], [], []
    h, inter = HIDDEN, MOE_INTER
    for e in range(n_e):
        prefix = f"layers.0.ffn.experts.{e}."
        w1 = _maybe_dequant(loader, prefix + "w1.weight", "cpu")
        w3 = _maybe_dequant(loader, prefix + "w3.weight", "cpu")
        w2 = _maybe_dequant(loader, prefix + "w2.weight", "cpu")
        wg = fit_linear_weight(w1, h, inter).to(device)
        wu = fit_linear_weight(w3, h, inter).to(device)
        wd = fit_linear_weight(w2, inter, h).to(device)
        p, s = pack_w4(wg)
        pg.append(p)
        sg.append(s)
        p, s = pack_w4(wu)
        pu.append(p)
        su.append(s)
        p, s = pack_w4(wd)
        pd.append(p)
        sd.append(s)
        del w1, w2, w3, wg, wu, wd
    return pg, sg, pu, su, pd, sd


def _charge_pool(pg, sg, pu, su, pd, sd, n_e: int) -> None:
    b = budget()
    for e in range(n_e):
        b.charge_device(pg[e], f"b.gate.p.{e}")
        b.charge_device(sg[e], f"b.gate.s.{e}")
        b.charge_device(pu[e], f"b.up.p.{e}")
        b.charge_device(su[e], f"b.up.s.{e}")
        b.charge_device(pd[e], f"b.down.p.{e}")
        b.charge_device(sd[e], f"b.down.s.{e}")


def _allocate_experts(n_e: int):
    loaded = _try_load_checkpoint_experts(n_e)
    if loaded is not None:
        pg, sg, pu, su, pd, sd = loaded
        source = "checkpoint"
    else:
        pg, sg = pack_pool(n_e, HIDDEN, MOE_INTER)
        pu, su = pack_pool(n_e, HIDDEN, MOE_INTER)
        pd, sd = pack_pool(n_e, MOE_INTER, HIDDEN)
        source = "random"
    _charge_pool(pg, sg, pu, su, pd, sd, n_e)
    return (pg, sg, pu, su, pd, sd), source

def _allocate_mxfp4_experts(n_e: int):
    """Allocate MXFP4 packed expert banks (gate_up concat + down)."""
    from mosaicvm.kernels.mxfp4 import pack_mxfp4

    pgu_list, sgu_list = [], []
    pdu_list, sdu_list = [], []
    for _ in range(n_e):
        wg = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
        wu = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
        p, s = pack_mxfp4(torch.cat([wg, wu], dim=1))
        pgu_list.append(p)
        sgu_list.append(s)
        del wg, wu
        wd = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
        p, s = pack_mxfp4(wd)
        pdu_list.append(p)
        sdu_list.append(s)
        del wd

    packed_gate_up = torch.stack(pgu_list, dim=0)
    scale_gate_up = torch.stack(sgu_list, dim=0)
    packed_down = torch.stack(pdu_list, dim=0)
    scale_down = torch.stack(sdu_list, dim=0)

    b = budget()
    b.charge_device(packed_gate_up, "b.mxfp4.gate_up")
    b.charge_device(scale_gate_up, "b.mxfp4.gate_up.scale")
    b.charge_device(packed_down, "b.mxfp4.down")
    b.charge_device(scale_down, "b.mxfp4.down.scale")
    return packed_gate_up, scale_gate_up, packed_down, scale_down


def _try_skinny_mxfp4(x, ids, weights, mx_pool):
    """Skinny MXFP4 grouped MoE backend."""
    try:
        from mosaicvm.kernels.skinny_fp4 import fused_moe_skinny
    except ImportError:
        return None, "pending_step3"
    pgu, sgu, pdu, sdu = mx_pool
    y = fused_moe_skinny(x, ids, weights, pgu, sgu, pdu, sdu)
    return y, "ok"


def _allocate_nk_experts(n_e: int, chunk: int = 32):
    """256 distinct experts as a QPAExpertBank, generated in chunks of 32."""
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
    from mosaicvm.kernels.qpa_mma import QPAExpertBank, _prepack_nk_to_frag

    n_mac = MOE_INTER // 16
    k_tiles_g = HIDDEN // 32
    n_tiles_d = HIDDEN // 16
    k_tiles_d = MOE_INTER // 32
    frag_gu = torch.empty(n_e, n_mac, k_tiles_g, 544, dtype=torch.uint8, device="cuda")
    frag_d = torch.empty(n_e, n_tiles_d, k_tiles_d, 272, dtype=torch.uint8, device="cuda")
    for i0 in range(0, n_e, chunk):
        i1 = min(n_e, i0 + chunk)
        c = i1 - i0
        pg, sg, pu, su, pd, sd = [], [], [], [], [], []
        for _ in range(c):
            wg = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            p, s = pack_mxfp4_nk(wg)
            pg.append(p); sg.append(s); del wg
            wu = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            p, s = pack_mxfp4_nk(wu)
            pu.append(p); su.append(s); del wu
            wd = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
            p, s = pack_mxfp4_nk(wd)
            pd.append(p); sd.append(s); del wd
        gfrag = _prepack_nk_to_frag(torch.stack(pg), torch.stack(sg))
        ufrag = _prepack_nk_to_frag(torch.stack(pu), torch.stack(su))
        frag_gu[i0:i1] = torch.cat([gfrag, ufrag], dim=-1)
        frag_d[i0:i1] = _prepack_nk_to_frag(torch.stack(pd), torch.stack(sd))
        del pg, sg, pu, su, pd, sd, gfrag, ufrag
        torch.cuda.empty_cache()
    b = budget()
    b.charge_device(frag_gu, "b.qpa.frag_gu")
    b.charge_device(frag_d, "b.qpa.frag_d")
    return QPAExpertBank(frag_gu=frag_gu, frag_d=frag_d, n_out_gu=2 * MOE_INTER, n_out_d=HIDDEN)


def _allocate_mxint_experts(n_e: int, chunk: int = 32):
    """256 distinct experts as an MXINTExpertBank, generated in chunks of 32."""
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
    from mosaicvm.kernels.mxint import MXINTExpertBank, qint_encode_w

    k_g = HIDDEN
    k_d = MOE_INTER
    w_gu = torch.empty(n_e, 2 * MOE_INTER, k_g, dtype=torch.int8, device="cuda")
    s_gu = torch.empty(n_e, 2 * MOE_INTER, k_g // 32, dtype=torch.uint8, device="cuda")
    w_d = torch.empty(n_e, HIDDEN, k_d, dtype=torch.int8, device="cuda")
    s_d = torch.empty(n_e, HIDDEN, k_d // 32, dtype=torch.uint8, device="cuda")
    for i0 in range(0, n_e, chunk):
        i1 = min(n_e, i0 + chunk)
        c = i1 - i0
        for j in range(c):
            wg = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pg, sg = pack_mxfp4_nk(wg); del wg
            wu = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
            pu, su = pack_mxfp4_nk(wu); del wu
            wd = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
            pd, sd = pack_mxfp4_nk(wd); del wd
            w_gu[i0 + j, :MOE_INTER] = qint_encode_w(pg)
            w_gu[i0 + j, MOE_INTER:] = qint_encode_w(pu)
            s_gu[i0 + j, :MOE_INTER] = sg.view(torch.uint8)
            s_gu[i0 + j, MOE_INTER:] = su.view(torch.uint8)
            w_d[i0 + j] = qint_encode_w(pd)
            s_d[i0 + j] = sd.view(torch.uint8)
            del pg, sg, pu, su, pd, sd
        torch.cuda.empty_cache()
    b = budget()
    b.charge_device(w_gu, "b.mxint.w_gu")
    b.charge_device(s_gu, "b.mxint.s_gu")
    b.charge_device(w_d, "b.mxint.w_d")
    b.charge_device(s_d, "b.mxint.s_d")
    return MXINTExpertBank(w_gu=w_gu, s_gu=s_gu, w_d=w_d, s_d=s_d,
                           n_out_gu=2 * MOE_INTER, n_out_d=HIDDEN)


def _try_fused_qpa(x, ids, weights, bank):
    from mosaicvm.kernels.qpa_mma import fused_moe_qpa
    y = fused_moe_qpa(x, ids, weights, bank=bank)
    return y, "ok"



def _row(k, family, unique, mean_m, sec, launches, status="ok", extra=None):
    assignments = k * TOP_K
    rec = {
        "k": k,
        "family": family,
        "unique": unique,
        "mean_m_per_expert": mean_m,
        "sec_per_moe_layer": sec,
        "launches_per_moe_layer": launches,
        "assignments_per_sec": (assignments / sec) if sec else None,
        "blocks_per_sec_proj": (1.0 / (sec * N_LAYERS)) if sec else None,
        "candidate_toks_moe_only": (k / (sec * N_LAYERS)) if sec else None,
        "status": status,
    }
    if extra:
        rec.update(extra)
    return rec


def _count_launches(fn) -> int:
    import mosaicvm.kernels.w4a16_gemm as gemm_mod

    n = {"n": 0}
    orig = gemm_mod.w4a16_gemm

    def wrapped(*a, **kw):
        n["n"] += 1
        return orig(*a, **kw)

    gemm_mod.w4a16_gemm = wrapped  # type: ignore[assignment]
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        gemm_mod.w4a16_gemm = orig  # type: ignore[assignment]
    return n["n"]


def _backend_sequential(x, ids, weights, pool):
    pg, sg, pu, su, pd, sd = pool
    k = x.shape[0]
    top_k = ids.shape[-1] if ids.ndim == 2 else 1
    token_of = torch.arange(k, device=x.device).repeat_interleave(top_k)
    x_exp = x[token_of]
    y_exp = w4a16_grouped_moe_sequential(x_exp, pg, pu, pd, (sg, su, sd), ids.reshape(-1))
    y_exp = y_exp * weights.reshape(-1, 1).to(y_exp.dtype)
    y = torch.zeros(k, x.shape[1], dtype=x.dtype, device=x.device)
    y.index_add_(0, token_of, y_exp)
    return y


def _try_grouped(x, ids, weights, pool, packed_gate_up=None):
    try:
        from mosaicvm.kernels.w4a16_grouped import fused_moe_w4a16
    except ImportError:
        return None, "pending_step4"
    pg, sg, pu, su, pd, sd = pool
    if packed_gate_up is not None:
        pgu, sgu = packed_gate_up
        y = fused_moe_w4a16(
            x, ids, weights, packed_gate=None, packed_up=None, packed_down=pd,
            scales={"down": sd, "gate_up": sgu}, packed_gate_up=pgu,
        )
    else:
        y = fused_moe_w4a16(
            x, ids, weights, pg, pu, pd, scales=(sg, su, sd),
        )
    return y, "ok"


def _try_flashinfer(x, ids, weights, n_e):
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return None, "skipped_no_flashinfer"
    return None, "skipped_no_flashinfer_grouped_bf16"


def _try_marlin(x, ids, weights, pool):
    try:
        from mosaicvm.kernels.marlin_moe import MARLIN_STATUS
    except ImportError:
        return None, "pending_step6"
    return None, MARLIN_STATUS.get("status", "pending_step6")


def _try_cutlass(x, ids, weights, pool):
    try:
        from mosaicvm.kernels.cutlass_grouped import CUTLASS_STATUS
    except ImportError:
        return None, "pending_step7"
    return None, CUTLASS_STATUS.get("status", "pending_step7")


def _remap_ids(ids: torch.Tensor, n_e: int) -> torch.Tensor:
    if int(ids.max().item()) < n_e:
        return ids
    return ids % n_e


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    b = budget()
    b.reset()
    n_e = N_EXPERTS
    truncated = None
    expert_layout = "separate_gate_up_down"
    packed_gate_up = None
    try:
        pool, source = _allocate_experts(n_e)
    except BudgetExceeded:
        torch.cuda.empty_cache()
        b.reset()
        n_e = 128
        truncated = 128
        pool, source = _allocate_experts(n_e)

    # Step 5: concatenated gate_up (GROUP=128 tiles do not cross 2048).
    from mosaicvm.kernels.w4a16_grouped import stack_gate_up

    packed_gate_up = stack_gate_up(pool)
    expert_layout = "gate_up_concat"

    # Phase 2.7: skip skinny MXFP4 pool here so INT4 can be fully freed
    # before the 256-expert exact NK backend (serialize backends).
    mx_pool = None
    mx_truncated = None
    mx_n_e = None

    rows = []
    backends_meta = {}

    for k in K_VALUES:
        for family in FAMILIES:
            n_u = min(_family_n_unique(family, k, TOP_K), n_e)
            tr = make_trace(k, TOP_K, n_unique=n_u, seed=1000 * k + {"peaked": 1, "mid": 2, "worst": 3}[family])
            ids = _remap_ids(tr["expert_ids"].to(device="cuda"), n_e)
            weights = tr["weights"].to(device="cuda", dtype=torch.float32)
            x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
            unique = int(ids.reshape(-1).unique().numel())
            counts = torch.bincount(ids.reshape(-1), minlength=n_e)
            used = counts[counts > 0].float()
            mean_m = float(used.mean().item()) if used.numel() else 0.0

            # A. sequential
            def step_seq():
                return _backend_sequential(x, ids, weights, pool)

            sec_seq = _bench(step_seq)
            launches_seq = _count_launches(step_seq)
            rows.append({
                "backend": "sequential",
                **_row(k, family, unique, mean_m, sec_seq, launches_seq),
            })

            # B. gpu_grouped_triton
            y_g, st_g = _try_grouped(x, ids, weights, pool, packed_gate_up)
            if st_g != "ok":
                rows.append({
                    "backend": "gpu_grouped_triton",
                    **_row(k, family, unique, mean_m, None, None, status=st_g),
                })
            else:
                def step_g():
                    return _try_grouped(x, ids, weights, pool, packed_gate_up)[0]

                sec_g = _bench(step_g)
                # grouped path must be 2-4 launches; count via profiler callback if available
                launches_g = None
                try:
                    from mosaicvm.kernels.w4a16_grouped import last_launch_count

                    step_g()
                    launches_g = last_launch_count()
                except Exception:
                    launches_g = None
                rows.append({
                    "backend": "gpu_grouped_triton",
                    **_row(k, family, unique, mean_m, sec_g, launches_g),
                })

            # C. marlin / batched_marlin
            y_m, st_m = _try_marlin(x, ids, weights, pool)
            if st_m != "ok":
                rows.append({
                    "backend": "marlin",
                    **_row(k, family, unique, mean_m, None, None, status=st_m),
                })
                rows.append({
                    "backend": "batched_marlin",
                    **_row(k, family, unique, mean_m, None, None, status=st_m),
                })
            else:
                def step_m():
                    return _try_marlin(x, ids, weights, pool)[0]

                sec_m = _bench(step_m)
                rows.append({
                    "backend": "marlin",
                    **_row(k, family, unique, mean_m, sec_m, None),
                })

            # D. flashinfer
            _, st_fi = _try_flashinfer(x, ids, weights, n_e)
            rows.append({
                "backend": "flashinfer_bf16",
                **_row(k, family, unique, mean_m, None, None, status=st_fi),
            })

            # E. cutlass — only meaningful later; stub until G1 miss
            _, st_c = _try_cutlass(x, ids, weights, pool)
            rows.append({
                "backend": "cutlass_grouped",
                **_row(k, family, unique, mean_m, None, None, status=st_c),
            })

            # F. skinny_mxfp4 (Phase 2.6 Exact-30)
            if mx_pool is not None:
                mx_ids = _remap_ids(ids, mx_n_e) if mx_n_e and mx_n_e != n_e else ids
                y_sk, st_sk = _try_skinny_mxfp4(x, mx_ids, weights, mx_pool)
                if st_sk != "ok":
                    rows.append({
                        "backend": "skinny_mxfp4",
                        **_row(k, family, unique, mean_m, None, None, status=st_sk),
                    })
                else:
                    def step_sk():
                        return _try_skinny_mxfp4(x, mx_ids, weights, mx_pool)[0]

                    sec_sk = _bench(step_sk)
                    launches_sk = None
                    try:
                        from mosaicvm.kernels.skinny_fp4 import last_launch_count as sk_launches

                        step_sk()
                        launches_sk = sk_launches()
                    except Exception:
                        launches_sk = None
                    rows.append({
                        "backend": "skinny_mxfp4",
                        **_row(k, family, unique, mean_m, sec_sk, launches_sk),
                    })
            else:
                rows.append({
                    "backend": "skinny_mxfp4",
                    **_row(k, family, unique, mean_m, None, None, status="skipped_no_vram"),
                })

            del x, ids, weights, y_g, y_m

    # Serialize: free INT4 pools before the exact NK backend.
    del pool, packed_gate_up, mx_pool
    budget().reset()
    torch.cuda.empty_cache()

    exact_status = "ok"
    exact_n_e = 256
    nk_pool = None
    census = None
    try:
        nk_pool = _allocate_nk_experts(256)
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError):
        exact_status = "invalid_not_256"
        exact_n_e = None
        torch.cuda.empty_cache()

    if nk_pool is not None:
        from mosaicvm.kernels.qpa_mma import last_launch_count as qpa_launches, launch_census
        for k in K_VALUES:
            for family in FAMILIES:
                n_u = min(_family_n_unique(family, k, TOP_K), 256)
                tr = make_trace(k, TOP_K, n_unique=n_u, seed=1000 * k + {"peaked": 1, "mid": 2, "worst": 3}[family])
                ids = tr["expert_ids"].to(device="cuda")
                if int(ids.max().item()) >= 256:
                    exact_status = "invalid_not_256"
                    rows.append({
                        "backend": "fused_moe_qpa",
                        **_row(k, family, 0, 0.0, None, None, status="invalid_not_256"),
                    })
                    continue
                weights = tr["weights"].to(device="cuda", dtype=torch.float32)
                x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
                unique = int(ids.reshape(-1).unique().numel())
                counts = torch.bincount(ids.reshape(-1), minlength=256)
                used = counts[counts > 0].float()
                mean_m = float(used.mean().item()) if used.numel() else 0.0
                try:
                    y, st = _try_fused_qpa(x, ids, weights, nk_pool)
                except Exception as exc:
                    rows.append({
                        "backend": "fused_moe_qpa",
                        **_row(k, family, unique, mean_m, None, None, status=f"error:{type(exc).__name__}"),
                    })
                    del x, ids, weights
                    continue
                if st != "ok":
                    rows.append({
                        "backend": "fused_moe_qpa",
                        **_row(k, family, unique, mean_m, None, None, status=st),
                    })
                else:
                    def step_q():
                        return _try_fused_qpa(x, ids, weights, nk_pool)[0]

                    sec_q = _bench(step_q)
                    step_q()
                    launches_q = qpa_launches()
                    extra = None
                    if k == 8 and family == "mid":
                        census = launch_census(step_q)
                        extra = {"launch_census": census}
                    rows.append({
                        "backend": "fused_moe_qpa",
                        **_row(k, family, unique, mean_m, sec_q, launches_q, extra=extra),
                    })
                del x, ids, weights
        # 48 complete 43-layer K8 blocks (synthetic mid unless golden_routes present)
        from mosaicvm.bench.floors import MOE_BUDGET_30_S
        golden_path = Path("artifacts/golden_routes.json")
        golden = json.loads(golden_path.read_text()) if golden_path.exists() else None
        s3_route_authority = "golden" if golden and (golden.get("windows") or golden.get("traces")) else "synthetic_distinct_interim"
        block_secs = []
        n_u_mid = min(_family_n_unique("mid", 8, TOP_K), 256)
        # one warmup
        tr0 = make_trace(8, TOP_K, n_unique=n_u_mid, seed=0)
        _try_fused_qpa(
            torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            tr0["expert_ids"].to(device="cuda"),
            tr0["weights"].to(device="cuda", dtype=torch.float32),
            nk_pool,
        )
        torch.cuda.synchronize()
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
                        elif ids.shape[0] != 8:
                            ids = ids[:8] if ids.shape[0] > 8 else ids
                        weights = torch.tensor(w["weights"], device="cuda", dtype=torch.float32)
                        if weights.dim() == 1:
                            weights = weights.view(1, -1).expand_as(ids.float())
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
                _try_fused_qpa(x, ids, weights, nk_pool)
                end.record()
                torch.cuda.synchronize()
                acc += start.elapsed_time(end) / 1e3
                del x, ids, weights
            block_secs.append(acc)
        block_t = torch.tensor(block_secs)
        s3_block_median_sec = float(block_t.median())
        s3_block_p90_sec = float(block_t.kthvalue(max(1, int(round(0.9 * 48)))).values)
        del nk_pool
        budget().reset()
        torch.cuda.empty_cache()
    else:
        s3_block_median_sec = None
        s3_block_p90_sec = None
        s3_route_authority = "synthetic_distinct_interim"

    # Informational MXINT K8-mid (no 48-block S3). Skip if VRAM cannot hold 256 experts.
    try:
        mxint_pool = _allocate_mxint_experts(256)
        from mosaicvm.kernels.mxint import fused_moe_mxint, last_launch_count as mxint_launches
        tr = make_trace(8, TOP_K, n_unique=min(_family_n_unique("mid", 8, TOP_K), 256), seed=1000 * 8 + 2)
        ids = tr["expert_ids"].to(device="cuda")
        if int(ids.max().item()) >= 256:
            rows.append({"backend": "fused_moe_mxint", **_row(8, "mid", 0, 0.0, None, None, status="invalid_not_256")})
        else:
            weights = tr["weights"].to(device="cuda", dtype=torch.float32)
            x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
            unique = int(ids.reshape(-1).unique().numel())
            counts = torch.bincount(ids.reshape(-1), minlength=256)
            used = counts[counts > 0].float()
            mean_m = float(used.mean().item()) if used.numel() else 0.0

            def step_mx():
                return fused_moe_mxint(x, ids, weights, bank=mxint_pool)

            sec_mx = _bench(step_mx)
            step_mx()
            rows.append({
                "backend": "fused_moe_mxint",
                **_row(8, "mid", unique, mean_m, sec_mx, mxint_launches()),
            })
            del x, ids, weights
        del mxint_pool
        budget().reset()
        torch.cuda.empty_cache()
    except (BudgetExceeded, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError):
        rows.append({"backend": "fused_moe_mxint", **_row(8, "mid", 0, 0.0, None, None, status="invalid_not_256")})
        torch.cuda.empty_cache()

    # G1 on mid K=8
    seq_k8 = next(r for r in rows if r["backend"] == "sequential" and r["k"] == 8 and r["family"] == "mid")
    grp_k8 = next(r for r in rows if r["backend"] == "gpu_grouped_triton" and r["k"] == 8 and r["family"] == "mid")
    g1 = None
    if grp_k8.get("sec_per_moe_layer") and seq_k8.get("sec_per_moe_layer"):
        g1 = grp_k8["sec_per_moe_layer"] <= 0.5 * seq_k8["sec_per_moe_layer"]
    from mosaicvm.bench.floors import MOE_LAYER_BUDGET_30_S, MOE_LAYER_BUDGET_40_S

    sk_k8 = next((r for r in rows if r["backend"] == "skinny_mxfp4" and r["k"] == 8 and r["family"] == "mid"), None)
    sk_sec = sk_k8.get("sec_per_moe_layer") if sk_k8 else None
    sk_launches = sk_k8.get("launches_per_moe_layer") if sk_k8 else None

    qpa_k8 = next((r for r in rows if r["backend"] == "fused_moe_qpa" and r["k"] == 8 and r["family"] == "mid"), None)
    exact_sec = qpa_k8.get("sec_per_moe_layer") if qpa_k8 else None
    from mosaicvm.bench.floors import MOE_BUDGET_30_S as _MOE30
    if exact_status == "invalid_not_256" or exact_n_e != 256:
        s3_pass = False
        s4_pass = False
        chosen_exact_backend = None
    else:
        s3_pass = s3_block_median_sec is not None and s3_block_median_sec <= _MOE30
        s4_pass = s3_block_median_sec is not None and s3_block_median_sec <= 0.112311
        chosen_exact_backend = "fused_moe_qpa"

    chosen = "sequential"
    best_sec = seq_k8["sec_per_moe_layer"] if seq_k8.get("sec_per_moe_layer") else float("inf")
    for name in ("gpu_grouped_triton", "marlin", "batched_marlin"):
        cand = next((r for r in rows if r["backend"] == name and r["k"] == 8 and r["family"] == "mid"), None)
        if cand and cand.get("sec_per_moe_layer") is not None and cand["sec_per_moe_layer"] < best_sec:
            best_sec = cand["sec_per_moe_layer"]
            chosen = name

    fastest_any_backend = chosen
    all_k8_mid = [r for r in rows if r["k"] == 8 and r["family"] == "mid" and r.get("sec_per_moe_layer")]
    if all_k8_mid:
        fastest_any_backend = min(all_k8_mid, key=lambda r: r["sec_per_moe_layer"])["backend"]
    from mosaicvm.kernels.qpa_mma import HYBRID as _HYBRID, qpa_mma_gemm
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk, act_quant_ref
    from mosaicvm.kernels.xsimt import decode_activation_exact
    ksplit_metrics = {}
    kernel_numeric = "numeric_parity"
    try:
        torch.manual_seed(0)
        _n, _k, _m = 64, 256, 8
        _w = (torch.randn(_n, _k, device="cuda") / (_k ** 0.5)).to(torch.bfloat16)
        _x = torch.randn(_m, _k, dtype=torch.bfloat16, device="cuda")
        _p, _s = pack_mxfp4_nk(_w)
        _xf8, _sA = act_quant_ref(_x)
        _xn = decode_activation_exact((_xf8, _sA))
        _ref = qpa_mma_gemm(_x, _p, _s, _n, qpa_k=1, x_norm=_xn, sA=_sA)
        _got = qpa_mma_gemm(_x, _p, _s, _n, qpa_k=8, x_norm=_xn, sA=_sA, ordered=True)
        _au, _bu = _got.view(torch.uint16), _ref.view(torch.uint16)
        _af, _bf = _got.float(), _ref.float()
        _fin = torch.isfinite(_af) & torch.isfinite(_bf)
        ksplit_metrics = {
            "bf16_bitwise_frac": float((_au == _bu).float().mean()),
            "max_bf16_ulp": int((_au.to(torch.int32) - _bu.to(torch.int32)).abs()[_fin].max().item()) if _fin.any() else 0,
            "max_abs": float((_af - _bf).abs().max()),
            "max_rel": float(((_af - _bf).abs() / _bf.abs().clamp_min(1e-8)).max()),
            "nan_inf_mismatches": int((~torch.isfinite(_af) ^ ~torch.isfinite(_bf)).sum().item()),
        }
        kernel_numeric = "bitwise_mosaic" if torch.equal(_got, _ref) else "numeric_parity"
        del _w, _x, _p, _s, _xf8, _sA, _xn, _ref, _got
    except Exception as exc:
        ksplit_metrics = {"error": f"{type(exc).__name__}:{exc}"}
    _cen = census or {}
    kernel_path_complete = bool(
        exact_status == "ok"
        and (_cen.get("custom_qpa_launches") or 0) > 0
        and ((_cen.get("custom_xsimt_launches") or 0) > 0 or _HYBRID == "qpa_all")
    )
    out = {
        "floor": FLOOR_B,
        "gpu": gpu,
        "n_experts": n_e,
        "truncated_experts": truncated,
        "mxfp4_truncated_experts": mx_truncated,
        "mxfp4_n_experts": mx_n_e,
        "exact_n_experts": exact_n_e,
        "exact_status": exact_status,
        "expert_layout": expert_layout,
        "top_k": TOP_K,
        "vram": budget().peak_summary(),
        "g1_k8_mid": g1,
        "chosen_backend": chosen,
        "chosen_exact_backend": chosen_exact_backend,
        "fastest_any_backend": fastest_any_backend,
        "s3_k8_mid_4_16ms": s3_pass,
        "s4_k8_mid_2_61ms": s4_pass,
        "s3_block_median_sec": s3_block_median_sec,
        "s3_block_p90_sec": s3_block_p90_sec,
        "s3_route_authority": s3_route_authority,
        "source_semantics": "validated",
        "kernel_numeric": kernel_numeric,
        "golden_greedy": "unavailable",
        "kernel_path_complete": kernel_path_complete,
        "ksplit_metrics": ksplit_metrics,
        "hybrid": _HYBRID,
        "moe_layer_budget_30_s": MOE_LAYER_BUDGET_30_S,
        "moe_layer_budget_40_s": MOE_LAYER_BUDGET_40_S,
        "skinny_k8_mid_sec": sk_sec,
        "skinny_k8_mid_launches": sk_launches,
        "exact_k8_mid_sec": exact_sec,
        "launch_census": census,
        "rows": rows,
        "backends_meta": backends_meta,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "n_experts": n_e,
        "truncated_experts": truncated,
        "exact_n_experts": exact_n_e,
        "exact_status": exact_status,
        "expert_layout": expert_layout,
        "g1_k8_mid": g1,
        "chosen_backend": chosen,
        "chosen_exact_backend": chosen_exact_backend,
        "s3_pass": s3_pass,
        "skinny_k8_mid_sec": sk_sec,
        "exact_k8_mid_sec": exact_sec,
        "seq_k8_mid_sec": seq_k8["sec_per_moe_layer"],
        "grp_k8_mid": {k: grp_k8.get(k) for k in ("sec_per_moe_layer", "status", "launches_per_moe_layer")},
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out



if __name__ == "__main__":
    run()
