"""§81 NEXT 1 — single-stack fit test + bank-split closure (AGENTS §48–53).

Empirical ground truth for short-context residency:

  banks  = PARTITIONED Mixed-A (two-call split): hot experts keep
           p4-GU + Hot8-DOWN (Mixed-A semantics); cold experts run
           pure packed4. Banks partitioned, never duplicated:
             p4_gu split (2.15 GB total) + h8_down_hot (8.39 MB x N_hot)
             + p4_down_cold (4.19 MB x N_cold)
  core   = 5.91 GB (no head — separately measured, P2_COREBLOCK convention)
  DSA    = BF16 storage, right-sized max_seq=4224 (4K ctx + 112 verify)
  state  = 4K populated on all 43 layers
  blocks = real eager + exact-DSA (fused kernel) steps at K112

Cold set = least-frequent experts by golden_routes frequency (production
intent: coldest experts pay the packed4-DOWN latency). Reports exact
bytes per §69, fit verdict, and the split-MoE block latency vs the
Mixed-A reference (1.1959 s at N112) — the §51 trade measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.fp8_core import CORE_PATH, HEAD_DIM
from mosaicvm.bench.p2_exact_dsa import (
    build_dsa_modules, compute_layer_eager, compute_layer_exact,
    load_core_no_head, populate_state)
from mosaicvm.kernels.mxint import (
    MXINTExpertBank, MXINTPacked4Bank, fused_moe_mxint)
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from safetensors.torch import safe_open

OUT = Path("artifacts/single_stack_fit.json")
GOLDEN = Path("artifacts/golden_routes.json")
MAXSEQ = 4224
K = 112
N_COLD = 104                      # ~430 MB reclaim target
MOE_REF_N112 = 1.1959             # Mixed-A reference (§11)


def stage(name, log):
    torch.cuda.synchronize()
    row = {"stage": name,
           "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
           "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
           "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    if log:
        row["delta_gb"] = round(row["allocated_gb"] - log[-1]["allocated_gb"], 3)
    log.append(row)
    print(f"{name:<40} alloc {row['allocated_gb']:>6.2f} GB "
          f"(Δ {row.get('delta_gb', 0.0):>6.2f})", flush=True)
    return row


def golden_freq_order():
    golden = json.loads(GOLDEN.read_text())
    counts = {}
    for c in golden["captures"]:
        for layer in c["layers"]:
            ids = layer["expert_ids"]
            flat = ids if not isinstance(ids[0], list) else \
                [e for t in ids for e in t]
            for e in flat:
                counts[e] = counts.get(e, 0) + 1
    all_ids = list(range(N_EXPERTS))
    return sorted(all_ids, key=lambda e: -counts.get(e, 0))  # hottest first


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    log = []
    _ = torch.zeros(1, device="cuda")
    stage("cuda_init", log)

    # ---------- partitioned banks ----------
    order = golden_freq_order()
    hot = sorted(order[:N_EXPERTS - N_COLD])
    cold = sorted(order[N_EXPERTS - N_COLD:])
    hot_pos = {e: i for i, e in enumerate(hot)}
    cold_pos = {e: i for i, e in enumerate(cold)}

    # staged construction: never co-resident with both full banks
    h8_full = _allocate_mxint_experts(N_EXPERTS)
    hidx = torch.tensor(hot, device=h8_full.w_d.device)
    h8_down_hot = MXINTExpertBank(
        w_gu=h8_full.w_gu[hidx][:1].clone(), s_gu=h8_full.s_gu[hidx][:1].clone(),
        w_d=h8_full.w_d[hidx].contiguous(), s_d=h8_full.s_d[hidx].contiguous(),
        n_out_gu=h8_full.n_out_gu, n_out_d=h8_full.n_out_d)
    del h8_full
    torch.cuda.empty_cache()

    p4_full = _allocate_packed4_experts(N_EXPERTS)
    cidx = torch.tensor(cold, device=p4_full.p_gu.device)
    p4_gu_hot = MXINTPacked4Bank(
        p_gu=p4_full.p_gu[hidx.to(p4_full.p_gu.device)].contiguous(),
        s_gu=p4_full.s_gu[hidx.to(p4_full.p_gu.device)].contiguous(),
        p_d=p4_full.p_d[hidx.to(p4_full.p_gu.device)][:1].clone(),
        s_d=p4_full.s_d[hidx.to(p4_full.p_gu.device)][:1].clone(),
        n_out_gu=p4_full.n_out_gu, n_out_d=p4_full.n_out_d)
    p4_cold = MXINTPacked4Bank(
        p_gu=p4_full.p_gu[cidx].contiguous(), s_gu=p4_full.s_gu[cidx].contiguous(),
        p_d=p4_full.p_d[cidx].contiguous(), s_d=p4_full.s_d[cidx].contiguous(),
        n_out_gu=p4_full.n_out_gu, n_out_d=p4_full.n_out_d)
    del p4_full
    torch.cuda.empty_cache()
    stage(f"banks_split({N_COLD} cold)", log)

    # ---------- core + DSA (right-sized) ----------
    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_no_head(f)
        dsa_mods = build_dsa_modules(f, bf16_storage=True, max_seq=MAXSEQ)
    torch.cuda.synchronize()
    stage("core+dsa_bf16(maxseq4224)", log)

    gen = torch.Generator(device="cuda").manual_seed(4242)
    for m in dsa_mods:
        populate_state(m, gen)
    torch.cuda.synchronize()
    stage("dsa_state_4K", log)

    # ---------- resident block steps (eager + exact DSA) ----------
    kv_cache = torch.randn(1024, 1, HEAD_DIM, dtype=torch.bfloat16,
                           device="cuda")
    xs = torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for _ in range(3):
                for L in (0, 2, 3):
                    lp = core["layers"][L]
                    compute_layer_eager(xs, lp, K, kv_cache)
                    compute_layer_exact(xs, lp, K, dsa_mods[L], 4096)
        torch.cuda.synchronize()
        stage("block_steps_K112_resident", log)
        block_ok = True
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.append({"stage": "block_steps_K112_resident", "result": "OOM"})
        print("block_steps_K112_resident      OOM")
        block_ok = False

    # ---------- split-MoE block latency at K112 (§51 trade) ----------
    # random uniform routes (value-insensitive timing; hot set from golden
    # frequency), split per-assignment like hybrid_bank_sweep
    moe_ms = None
    if block_ok:
        torch.manual_seed(7)
        routes = []
        for _ in range(8 * N_LAYERS):
            ids = torch.randint(0, N_EXPERTS, (K, 6), device="cuda")
            wts = torch.rand(K, 6, device="cuda", dtype=torch.float32)
            wts = wts / wts.sum(-1, keepdim=True)
            flat = ids.reshape(-1)
            is_hot = torch.tensor([int(e) in hot_pos for e in flat.tolist()],
                                  device="cuda")
            tok = torch.arange(flat.numel(), device="cuda") // 6
            hid = torch.tensor([hot_pos[int(e)] for e in flat[is_hot].tolist()],
                               dtype=torch.int64, device="cuda").reshape(-1, 1)
            cid = torch.tensor([cold_pos[int(e)] for e in flat[~is_hot].tolist()],
                               dtype=torch.int64, device="cuda").reshape(-1, 1)
            routes.append((tok[is_hot], hid, wts.reshape(-1, 1)[is_hot],
                           tok[~is_hot], cid, wts.reshape(-1, 1)[~is_hot]))
        idx = [0]

        def moe_step():
            i = idx[0]
            ht, hi, hw, ct, ci, cw = routes[i % len(routes)]
            out = torch.zeros(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
            if hi.numel():
                oh = fused_moe_mxint(xs[ht], hi, hw, gu_bank=p4_gu_hot,
                                     down_bank=h8_down_hot, crossover=0)
                out = out.index_add(0, ht, oh.to(torch.bfloat16))
            if ci.numel():
                oc = fused_moe_mxint(xs[ct], ci, cw, bank=p4_cold,
                                     weight_repr="packed4", crossover=0)
                out = out.index_add(0, ct, oc.to(torch.bfloat16))
            idx[0] += 1

        with torch.no_grad():
            for _ in range(3):
                moe_step()
            torch.cuda.synchronize()
            ts = []
            for _ in range(24):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(N_LAYERS):
                    moe_step()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
        ts.sort()
        moe_ms = ts[len(ts) // 2]
        print(f"split-MoE 43-layer block @K112: {moe_ms:.1f} ms "
              f"(Mixed-A ref {MOE_REF_N112*1e3:.0f} ms, "
              f"delta {moe_ms - MOE_REF_N112*1e3:+.1f} ms)")

    # ---------- head attempt (informational) ----------
    head_ok = False
    try:
        with safe_open(str(CORE_PATH), framework="pt") as f:
            head_w = f.get_tensor("head.weight").to(torch.bfloat16).to("cuda")
        torch.cuda.synchronize()
        stage("lm_head_attempt", log)
        head_ok = True
        del head_w
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.append({"stage": "lm_head_attempt", "result": "OOM"})
        print("lm_head_attempt                    OOM")
    torch.cuda.empty_cache()

    steady = next((r["allocated_gb"] for r in reversed(log)
                   if r.get("allocated_gb") is not None), None)
    peak = log[-1].get("max_allocated_gb") if log[-1].get("stage") == \
        "block_steps_K112_resident" else max(
        (r.get("max_allocated_gb", 0) for r in log), default=0)
    results = {
        "config": {"n_cold": N_COLD, "max_seq": MAXSEQ, "K": K,
                   "cold_set": "golden-frequency tail (least used)"},
        "stages": log, "block_steps_ok": block_ok, "head_fits": head_ok,
        "steady_gb": steady,
        "split_moe_block_ms": moe_ms,
        "moe_ref_n112_s": MOE_REF_N112,
        "moe_delta_ms": (round(moe_ms - MOE_REF_N112 * 1e3, 1)
                         if moe_ms else None),
    }
    print("\nRESULT:", json.dumps(
        {k: v for k, v in results.items() if k != "stages"}, indent=1))
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
