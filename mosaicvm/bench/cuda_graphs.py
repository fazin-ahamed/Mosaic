"""CUDA Graphs for the MoE-only K-set. Eager stays if capture fails."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.floors import HIDDEN, MOE_INTER, TOP_K
from mosaicvm.bench.routing_traces import make_trace
from mosaicvm.kernels.w4a16_gemm import pack_w4
from mosaicvm.kernels.w4a16_grouped import fused_moe_w4a16, stack_gate_up

GRAPH_K = (1, 2, 3, 5, 8)
N_E = 32  # graph capture of the fused path; 256-expert bank already timed in Floor B


def _pool(n_e: int):
    pg, sg, pu, su, pd, sd = [], [], [], [], [], []
    for _ in range(n_e):
        p, s = pack_w4(torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda"))
        pg.append(p)
        sg.append(s)
        p, s = pack_w4(torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device="cuda"))
        pu.append(p)
        su.append(s)
        p, s = pack_w4(torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda"))
        pd.append(p)
        sd.append(s)
    return pg, sg, pu, su, pd, sd


def _capture(k: int, pool, packed_gate_up):
    pg, sg, pu, su, pd, sd = pool
    pgu, sgu = packed_gate_up
    tr = make_trace(k, TOP_K, n_unique=min(k * TOP_K // 2, N_E), seed=8000 + k)
    x = torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = (tr["expert_ids"].to(device="cuda") % N_E).contiguous()
    weights = tr["weights"].to(device="cuda", dtype=torch.float32).contiguous()
    y = torch.empty_like(x)

    static_x = x.clone()
    static_ids = ids.clone()
    static_w = weights.clone()
    static_y = y.clone()

    def eager():
        return fused_moe_w4a16(
            static_x, static_ids, static_w, None, None, pd,
            scales={"gate_up": sgu, "down": sd}, packed_gate_up=pgu,
        )

    # warmup
    for _ in range(3):
        static_y.copy_(eager())
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            out = eager()
            static_y.copy_(out)
    except Exception as e:
        return {
            "k": k,
            "cuda_graph": "failed",
            "reason": f"{type(e).__name__}: {e}",
        }

    # overwrite ids in-place and replay
    tr2 = make_trace(k, TOP_K, n_unique=min(k * TOP_K // 2, N_E), seed=9000 + k)
    static_ids.copy_((tr2["expert_ids"].to(device="cuda") % N_E).contiguous())
    static_w.copy_(tr2["weights"].to(device="cuda", dtype=torch.float32).contiguous())
    g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(20):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return {
        "k": k,
        "cuda_graph": "ok",
        "replay_sec": start.elapsed_time(end) / 20 / 1e3,
        "y_norm": float(static_y.float().norm().item()),
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    pool = _pool(N_E)
    packed_gate_up = stack_gate_up(pool)
    rows = []
    for k in GRAPH_K:
        rows.append(_capture(k, pool, packed_gate_up))
        torch.cuda.empty_cache()
    out = {"gpu": gpu, "n_experts_graph": N_E, "rows": rows}
    Path("artifacts").mkdir(exist_ok=True)
    path = Path("artifacts/cuda_graphs.json")
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print("wrote", path)
    return out


if __name__ == "__main__":
    run()
