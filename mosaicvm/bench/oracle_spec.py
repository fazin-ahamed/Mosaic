"""Oracle speculative-verifier ceiling. Perfect residency, no trained drafter."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import (
    HIDDEN,
    MOE_INTER,
    N_LAYERS,
    TOP_K,
    measure_dense_attention,
    measure_hyper_connection,
    measure_router,
    measure_shared_expert,
    _bench,
)

ORACLE_K = (2, 3, 4, 5, 6, 8, 10)
N_POOL = 16


def _pack_pool(n_e: int, k_in: int, n_out: int, device="cuda"):
    from mosaicvm.kernels.w4a16_gemm import pack_w4

    packed, scales = [], []
    for _ in range(n_e):
        w = torch.randn(k_in, n_out, dtype=torch.bfloat16, device=device)
        p, s = pack_w4(w)
        packed.append(p)
        scales.append(s)
    return packed, scales


def measure_grouped_routed(k, top_k, use_w4, aec, packed=None) -> dict:
    """One w4a16_gemm (or bf16 GEMM) per unique expert, not K independent GEMVs."""
    m = k * top_k
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    expert_ids = torch.randint(0, N_POOL, (m,), device="cuda")
    kernel = "bf16"
    if use_w4 and packed is not None:
        from mosaicvm.kernels.w4a16_gemm import w4a16_grouped_moe

        pg, sg, pu, su, pd, sd = packed

        def step():
            return w4a16_grouped_moe(x, pg, pu, pd, (sg, su, sd), expert_ids)

        sec = _bench(step)
        kernel = "w4a16"
    else:
        wg = torch.randn(N_POOL, HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
        wu = torch.randn(N_POOL, HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda")
        wd = torch.randn(N_POOL, MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def step():
            y = torch.zeros(m, HIDDEN, dtype=x.dtype, device=x.device)
            for e in expert_ids.unique().tolist():
                mask = expert_ids == e
                xe = x[mask]
                g = torch.nn.functional.silu(xe @ wg[e])
                u = xe @ wu[e]
                y[mask] = (g * u) @ wd[e]
            return y

        sec = _bench(step)
        kernel = "bf16_unique_loop"

    aec_sec = 0.0
    if aec:
        xd = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
        w1 = torch.randn(HIDDEN, 64, dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def step_aec():
            return torch.nn.functional.gelu(xd @ w1) @ w2

        aec_sec = _bench(step_aec)
    return {"K": k, "top_k": top_k, "sec_per_step": sec, "aec_sec": aec_sec, "kernel": kernel}


def run(top_k: int = TOP_K, aec: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        raise SystemExit(f"unexpected GPU: {gpu}")

    packed = None
    use_w4 = True
    try:
        from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_gemm

        w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        w4a16_gemm(x, *pack_w4(w))
        pg, sg = _pack_pool(N_POOL, HIDDEN, MOE_INTER)
        pu, su = _pack_pool(N_POOL, HIDDEN, MOE_INTER)
        pd, sd = _pack_pool(N_POOL, MOE_INTER, HIDDEN)
        packed = (pg, sg, pu, su, pd, sd)
    except Exception as e:
        print("W4A16 unavailable, oracle uses bf16:", type(e).__name__, e)
        use_w4 = False

    attn = {k: measure_dense_attention(k=k, ctx=4096)["sec_per_step"] for k in ORACLE_K}
    router = measure_router()["sec_per_step"]
    shared = {k: measure_shared_expert(k)["sec_per_step"] for k in ORACLE_K}
    hc = {k: measure_hyper_connection(k)["sec_per_step"] for k in ORACLE_K}

    rows = []
    for k in ORACLE_K:
        routed = measure_grouped_routed(k, top_k, use_w4, aec, packed=packed)
        per_layer = attn[k] + router + shared[k] + routed["sec_per_step"] + hc[k] + routed["aec_sec"]
        t_block = per_layer * N_LAYERS
        accepted = k / t_block
        rows.append({
            "K": k,
            "top_k": top_k,
            "verification_block_latency_s": t_block,
            "draft_tokens_per_block": k,
            "mean_accepted_tokens_per_block": k,
            "accepted_output_toks": accepted,
            "t_block_ms": t_block * 1e3,
            "need_133_ratio": (t_block * 1e3) / 133.0,
            "need_100_ratio": (t_block * 1e3) / 100.0,
            "routed_kernel": routed["kernel"],
            "aec": aec,
            "per_layer_sec": per_layer,
        })

    k5 = next(r for r in rows if r["K"] == 5)
    k8 = next(r for r in rows if r["K"] == 8)
    out = {
        "gpu": gpu,
        "top_k": top_k,
        "aec": aec,
        "rows": rows,
        "gates": {
            "k5_ge_25": k5["accepted_output_toks"] >= 25,
            "k8_ge_35": k8["accepted_output_toks"] >= 35,
            "k8_ge_30": k8["accepted_output_toks"] >= 30,
            "k5_toks": k5["accepted_output_toks"],
            "k8_toks": k8["accepted_output_toks"],
        },
    }
    Path("artifacts").mkdir(exist_ok=True)
    path = Path("artifacts/oracle_spec_3060.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"rows": rows, "gates": out["gates"], "wrote": str(path)}, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
