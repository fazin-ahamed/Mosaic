"""Floor C (full layer) and Floor D (43-layer projection)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import (
    measure_dense_attention,
    measure_hyper_connection,
    measure_router,
    measure_shared_expert,
    _bench,
)
from mosaicvm.bench.floors import (
    FLOOR_C,
    FLOOR_D,
    HIDDEN,
    K_VALUES,
    N_LAYERS,
    TOP_K,
    candidate_toks,
)

VOCAB = 129280
ARTIFACT = Path("artifacts/floor_cd.json")


def _best_moe_sec(floor_b: dict) -> dict[int, float]:
    rows = floor_b.get("rows") or []
    chosen = floor_b.get("chosen_backend") or "sequential"
    out = {}
    for k in K_VALUES:
        mid = [
            r for r in rows
            if r.get("k") == k and r.get("family") == "mid" and r.get("sec_per_moe_layer") is not None
        ]
        pref = [r for r in mid if r.get("backend") == chosen]
        if not pref:
            pref = [r for r in mid if r.get("backend") == "sequential"]
        if pref:
            out[k] = float(pref[0]["sec_per_moe_layer"])
    return out


def _try_lm_head(k: int) -> tuple[float, str]:
    try:
        from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_gemm
        from mosaicvm.runtime.memory_budget import BudgetExceeded, budget

        w = torch.randn(HIDDEN, VOCAB, dtype=torch.bfloat16, device="cuda")
        packed, scale = pack_w4(w)
        del w
        b = budget()
        try:
            b.charge_device(packed, "lm_head.packed")
            b.charge_device(scale, "lm_head.scale")
        except BudgetExceeded:
            del packed, scale
            torch.cuda.empty_cache()
            return 0.0, "omitted_vram"
        x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def step():
            return w4a16_gemm(x, packed, scale)

        sec = _bench(step)
        del packed, scale, x
        torch.cuda.empty_cache()
        return sec, "ok"
    except Exception as e:
        torch.cuda.empty_cache()
        return 0.0, f"omitted_{type(e).__name__}"


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    floor_b = json.loads(Path("artifacts/floor_b_moe.json").read_text())
    moe = _best_moe_sec(floor_b)
    router_s = measure_router()["sec_per_step"]

    lm_s, lm_status = _try_lm_head(max(K_VALUES))
    # lm_head measured at max K; scale linearly with k (same GEMM, M=k)
    lm_at_max = lm_s
    k_max = max(K_VALUES)

    floor_c = []
    floor_d = []
    for k in K_VALUES:
        attn_s = measure_dense_attention(k, ctx=4096)["sec_per_step"]
        shared_s = measure_shared_expert(k)["sec_per_step"]
        hc_s = measure_hyper_connection(k)["sec_per_step"]
        moe_s = moe[k]
        lm_k = 0.0 if lm_status != "ok" else lm_at_max * (k / k_max)
        per_layer = attn_s + router_s + shared_s + moe_s + hc_s
        t_block = per_layer * N_LAYERS + lm_k
        bps = 1.0 / t_block
        toks = candidate_toks(bps, k)
        rest = attn_s + router_s + shared_s + hc_s
        t_block_no_moe = rest * N_LAYERS + lm_k
        amdahl_inf = k / t_block_no_moe if t_block_no_moe > 0 else None
        c_row = {
            "K": k,
            "floor": FLOOR_C,
            "attn_s": attn_s,
            "router_s": router_s,
            "shared_s": shared_s,
            "moe_s": moe_s,
            "hc_s": hc_s,
            "per_layer": per_layer,
        }
        d_row = {
            "K": k,
            "floor": FLOOR_D,
            "verification_block_latency_s": t_block,
            "blocks_per_sec": bps,
            "draft_tokens_per_block": k,
            "mean_accepted_prefix_length": k,
            "candidate_toks": toks,
            "assignments_per_sec": k * TOP_K * N_LAYERS * bps,
            "amdahl_if_moe_inf_toks": amdahl_inf,
            "need_5bps": bps / 5.0,
            "need_6bps": bps / 6.0,
            "lm_s": lm_k,
            "lm_head": lm_status,
            "moe_backend": floor_b.get("chosen_backend"),
        }
        floor_c.append(c_row)
        floor_d.append(d_row)

    out = {
        "gpu": gpu,
        "top_k": TOP_K,
        "chosen_backend": floor_b.get("chosen_backend"),
        "lm_head": lm_status,
        "floor_c": floor_c,
        "floor_d": floor_d,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    k8 = next(r for r in floor_d if r["K"] == 8)
    print(json.dumps({
        "wrote": str(ARTIFACT),
        "k8": {
            "blocks_per_sec": k8["blocks_per_sec"],
            "candidate_toks": k8["candidate_toks"],
            "mean_accepted_prefix_length": k8["mean_accepted_prefix_length"],
            "amdahl_if_moe_inf_toks": k8["amdahl_if_moe_inf_toks"],
            "lm_head": lm_status,
        },
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
