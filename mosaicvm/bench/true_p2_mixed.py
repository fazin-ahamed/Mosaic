"""TRUE P2: full 43-layer interleaved target block, production-fit config.

Per §29-38 of the canonical roadmap:
  core   = source-FP8 resident (load_core_resident, includes BF16 LM head)
  MoE    = Mixed-A banks (gu=packed4, down=Hot8), crossover=0, exact
  block  = for each of 43 layers: compute_layer_fused -> fused_moe_mxint
           -> natural residual; LM head once at the end
  widths = N56 (real-golden sanity) / N80 / N112 (corpus-wrap)
  gates  = R_oracle = E[A]/T_block vs §39 bands; Δ_core = T_block - T_MoE
           vs §38 bands; completeness: full path incl. LM head -> TRUE P2
           (else P2_COREBLOCK + separate omission measurement)

VRAM sequencing: banks are built BEFORE the core loads (peak 6.7 GB during
full-hot8 slicing), then core 7.01 GB → ~11.55 GB resident.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.wide_verify import _bench_blocks, _golden_windows
from mosaicvm.bench.wideverify_v3 import capture_layer_tokens, win_wrap
from mosaicvm.kernels.mxint import (
    MXINTExpertBank, MXINTPacked4Bank, fused_moe_mxint,
)
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from mosaicvm.bench.fp8_core import (
    CORE_PATH, HEAD_DIM, load_core_resident, compute_layer_fused,
)
from safetensors.torch import safe_open

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/true_p2_mixed.json")
ORACLE = Path("artifacts/branchpack_oracle.json")
K_VALUES = (56, 80, 112)
MOE_REF = {"56": 0.8379, "80": 1.0046, "112": 1.1959}  # prodfit_frontier.json
CTX_KV = 1024


def certify(snaps, start, end):
    temps = [s.get("temp_c") for s in snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    c0 = start.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == c0 for s in snaps + [end])
    return ("STEADY" if drift <= 3 else "CONTROLLED" if drift <= 5 else "UNKNOWN") \
        if locked else "UNKNOWN"


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()

    # ---------- banks first (peak VRAM during construction) ----------
    h8_full = _allocate_mxint_experts(N_EXPERTS)
    h8_down = MXINTExpertBank(
        w_gu=h8_full.w_gu[:1].clone(), s_gu=h8_full.s_gu[:1].clone(),
        w_d=h8_full.w_d, s_d=h8_full.s_d,
        n_out_gu=h8_full.n_out_gu, n_out_d=h8_full.n_out_d)
    del h8_full
    p4_full = _allocate_packed4_experts(N_EXPERTS)
    p4_gu = MXINTPacked4Bank(
        p_gu=p4_full.p_gu, s_gu=p4_full.s_gu,
        p_d=p4_full.p_d[:1].clone(), s_d=p4_full.s_d[:1].clone(),
        n_out_gu=p4_full.n_out_gu, n_out_d=p4_full.n_out_d)
    del p4_full
    torch.cuda.empty_cache()
    banks_gb = torch.cuda.memory_allocated() / 1e9
    # ---------- LM head measured separately BEFORE core load (VRAM) ----------
    # banks 4.4 GB + head 1.06 GB fits; core+banks+head does not (§33:
    # P2_COREBLOCK + separate omission measurement, T_reconstructed reported).
    with safe_open(str(CORE_PATH), framework="pt") as f:
        head_w = f.get_tensor("head.weight").to(torch.bfloat16).to("cuda")
    t_head = {}
    for K in K_VALUES:
        xh = torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            torch.nn.functional.linear(xh, head_w)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); torch.nn.functional.linear(xh, head_w); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 1e3)
        ts.sort()
        t_head[str(K)] = ts[len(ts)//2]
        print(f"LM head @K={K}: {ts[len(ts)//2]*1e3:.2f} ms")
    del head_w, xh
    torch.cuda.empty_cache()

    # ---------- core minus LM head ----------
    def _load_core_no_head(f) -> dict:
        from mosaicvm.bench.fp8_core import FP8_TENSORS, N_LAYERS as NL
        core = {"layers": []}
        core["norm_w"] = f.get_tensor("norm.weight").to("cuda")
        core["hc_head_scale"] = f.get_tensor("hc_head_scale").to("cuda")
        for layer in range(NL):
            pre = f"layers.{layer}."
            lp = {"fp8": {}, "scales": {}, "small": {}}
            for name in FP8_TENSORS:
                lp["fp8"][name] = f.get_tensor(pre + name + ".weight").to("cuda")
                lp["scales"][name] = f.get_tensor(pre + name + ".scale").to("cuda")
            lp["small"]["gate_w"] = f.get_tensor(pre + "ffn.gate.weight").to(torch.bfloat16).to("cuda")
            bias_name = pre + "ffn.gate.bias"
            gb = f.get_tensor(bias_name) if bias_name in f.keys() else None
            lp["small"]["gate_bias"] = (gb.to("cuda") if gb is not None
                                        else torch.zeros(256, device="cuda"))
            for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                lp["small"][nm] = f.get_tensor(pre + nm).to("cuda")
            for src, dst in (("attn_norm.weight", "attn_norm"),
                             ("ffn_norm.weight", "ffn_norm"),
                             ("attn.q_norm.weight", "q_norm"),
                             ("attn.kv_norm.weight", "kv_norm"),
                             ("attn.attn_sink", "attn_sink")):
                lp["small"][dst] = f.get_tensor(pre + src).to("cuda")
            core["layers"].append(lp)
        return core

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = _load_core_no_head(f)
    torch.cuda.synchronize()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    peak_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"core-no-head + banks: {resident_gb:.2f} GB (peak {peak_load:.2f})")

    golden = json.loads(GOLDEN.read_text())
    oracle = json.loads(ORACLE.read_text())["oracle"]
    gw56 = _golden_windows(56, golden)
    pools = capture_layer_tokens(golden)
    gen = torch.Generator().manual_seed(5150)

    warmup_steady(120.0)
    t_start = snapshot()
    snaps = []
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "banks_gb": banks_gb, "resident_gb": resident_gb,
        "load_peak_gb": peak_load,
        "config": "43-layer interleaved: compute_layer_fused (FP8 core) + "
                  "Mixed-A fused_moe_mxint c0 + natural residual; "
                  "P2_COREBLOCK (LM head measured separately, VRAM)",
        "rows": {}, "completeness": "P2_COREBLOCK: LM head omitted from "
                  "resident block (co-residency limit), measured separately "
                  "at identical shapes; T_reconstructed = block + head.",
    }

    for K in K_VALUES:
        if K == 56:
            routes = [(torch.tensor(r[0]), torch.tensor(r[1], dtype=torch.float32))
                      for r in [gw56[(i * 7) % len(gw56)] for i in range(11 * N_LAYERS)]]
            auth = "real_golden"
        else:
            routes = [win_wrap(pools[(i * 5) % len(pools)][2], K, gen)
                      for i in range(11 * N_LAYERS)]
            auth = "corpus_wrap"
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        kvs = [torch.randn(CTX_KV, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
               for _ in range(2)]
        ids_t = [r[0].to("cuda") for r in routes]
        w_t = [r[1].to("cuda") for r in routes]
        idx = 0

        def block_step():
            """One full 43-layer block: FP8 core -> exact MoE -> residual."""
            nonlocal idx
            x = xs[idx % 4]
            kv_cache = kvs[idx % 2]
            for L in range(N_LAYERS):
                lp = core["layers"][L]
                h = compute_layer_fused(x, lp["fp8"], lp["scales"],
                                        lp["small"], K, kv_cache)
                m = fused_moe_mxint(h, ids_t[idx], w_t[idx],
                                    gu_bank=p4_gu, down_bank=h8_down,
                                    crossover=0)
                x = (h + m).to(torch.bfloat16)
                idx += 1

        snaps.append(snapshot())
        try:
            t = _bench_blocks(block_step, 8)
            peak_run = torch.cuda.max_memory_allocated() / 1e9
            ea = oracle.get(str(K), {}).get("expected_accepted_depth")
            t_rec = t + t_head[str(K)]
            r_or = ea / t_rec if ea else None
            d_core = t - MOE_REF[str(K)]
            if d_core <= 0.180:
                cc = "ASSUMPTION CONFIRMED"
            elif d_core <= 0.200:
                cc = "SMALL WIDTH PENALTY"
            elif d_core <= 0.250:
                cc = "MATERIAL CORE SCALING"
            else:
                cc = "CONSTANT-CORE ASSUMPTION FAILED"
            if r_or and r_or >= 40:
                cls = "P2 STRONG"
            elif r_or and r_or >= 35:
                cls = "P2 GOOD"
            elif r_or and r_or >= 33:
                cls = "P2 VIABLE"
            elif r_or and r_or >= 30:
                cls = "P2 MINIMUM"
            else:
                cls = "P2 MISS"
            results["rows"][str(K)] = {
                "t_coreblock_s": t, "t_head_s": t_head[str(K)],
                "t_reconstructed_s": t_rec,
                "route_auth": auth, "E_A": ea,
                "R_oracle_per_s": r_or, "p2_class": cls,
                "moe_ref_s": MOE_REF[str(K)], "delta_core_s": d_core,
                "core_assumption": cc, "run_peak_gb": peak_run,
            }
            print(f"K={K:3d} [{auth:12s}]: coreblock={t:.4f}s +head "
                  f"{t_head[str(K)]*1e3:.1f}ms = {t_rec:.4f}s "
                  f"R_oracle={r_or and round(r_or,1)}/s [{cls}] "
                  f"Δcore={d_core*1e3:.0f}ms [{cc}]", flush=True)
        except torch.OutOfMemoryError as exc:
            results["rows"][str(K)] = {"error": f"OOM: {str(exc)[:120]}"}
            print(f"K={K}: OOM — {str(exc)[:100]}")
        del routes, xs, kvs, ids_t, w_t
        torch.cuda.empty_cache()

    t_end = snapshot()
    results["thermal"] = {"label": certify(snaps, t_start, t_end),
                          "start": t_start, "end": t_end}
    print(f"thermal [{results['thermal']['label']}]")
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
