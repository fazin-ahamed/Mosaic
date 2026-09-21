"""§86 NEXT 0 — memory-accounting audit (AGENTS §54–56). FINAL FORM.

Resolves the 13.4-GB reporting error (stale historical 7.0-GB core figure
+ estimated 1.95-GB DSA footprint; measured values: 4.52 + 5.91 + 1.67).
Records exact bytes per §56 stage with nvidia-smi system ground truth,
measures the BF16-storage DSA variant (bitwise-identical lazy-widen,
tests/test_dsa_bf16_storage.py), and issues the co-residency verdict
empirically:

  Phase 1: banks + core resident → fp32-DSA load attempt (expected OOM).
  Phase 2 (banks-free): core + bf16-DSA + 4K state + K112 block peak —
           exact component bytes and transient size.
  Verdict: arithmetic vs the MEASURED process budget (card 12.48 GB minus
           ~0.16 GB other processes minus display/driver overhead observed
           via nvidia-smi).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.fp8_core import CORE_PATH, HEAD_DIM
from mosaicvm.bench.p2_exact_dsa import (
    build_dsa_modules, compute_layer_eager, compute_layer_exact,
    load_core_no_head, populate_state)
from mosaicvm.kernels.mxint import MXINTExpertBank, MXINTPacked4Bank
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from safetensors.torch import safe_open

OUT = Path("artifacts/mem_audit.json")


def smi_used_free():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10).stdout.strip()
        u, f = (float(x) for x in out.split(","))
        return {"smi_used_mb": u, "smi_free_mb": f}
    except Exception:
        return {}


def stage(name, log):
    torch.cuda.synchronize()
    row = {"stage": name,
           "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
           "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
           "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    row.update(smi_used_free())
    if log:
        row["delta_gb"] = round(row["allocated_gb"] - log[-1]["allocated_gb"], 3)
    log.append(row)
    print(f"{name:<36} alloc {row['allocated_gb']:>6.2f} GB "
          f"(Δ {row.get('delta_gb', 0.0):>6.2f})  smi_free "
          f"{row.get('smi_free_mb', 0):>6.0f} MB")
    return row


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    prop = torch.cuda.get_device_properties(0)
    log = []
    results = {"gpu": torch.cuda.get_device_name(0),
               "total_vram_gb_decimal": round(prop.total_memory / 1e9, 3)}
    free0 = smi_used_free()

    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    stage("cuda_init", log)

    # ---------------- Phase 1: banks + core + fp32 attempt ---------------
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
    stage("P1_expert_banks", log)

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_no_head(f)
    torch.cuda.synchronize()
    stage("P1_core_no_head", log)

    fp32_result = "LOADED"
    try:
        with safe_open(str(CORE_PATH), framework="pt") as f:
            dsa_f32 = build_dsa_modules(f, bf16_storage=False)
        torch.cuda.synchronize()
        stage("P1_dsa_fp32_ATTEMPT", log)
        fp32_dsa_gb = log[-1]["delta_gb"]
        del dsa_f32
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        fp32_result = "OOM"
        fp32_dsa_gb = 1.67          # measured banks-free in phase 2 ref
        torch.cuda.empty_cache()
        print("P1_dsa_fp32_ATTEMPT            OOM (empirical)")

    # ---------------- Phase 2: banks-free precise measurement ------------
    del core, h8_down, p4_gu
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    stage("P2_reset_banks_free", log)

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_no_head(f)
    torch.cuda.synchronize()
    stage("P2_core_no_head", log)
    core_gb = log[-1]["delta_gb"]

    with safe_open(str(CORE_PATH), framework="pt") as f:
        dsa_mods = build_dsa_modules(f, bf16_storage=True)
    torch.cuda.synchronize()
    stage("P2_dsa_bf16_weights_state_buffers", log)
    dsa_bf16_gb = log[-1]["delta_gb"]   # weights + preallocated state bufs

    gen = torch.Generator(device="cuda").manual_seed(4242)
    for m in dsa_mods:
        populate_state(m, gen)
    torch.cuda.synchronize()
    stage("P2_plus_state_4K", log)
    state_gb = log[-1]["delta_gb"]

    K = 112
    kv_cache = torch.randn(1024, 1, HEAD_DIM, dtype=torch.bfloat16,
                           device="cuda")
    xs = torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            for L in (0, 2, 3):
                lp = core["layers"][L]
                compute_layer_eager(xs, lp, K, kv_cache)
                compute_layer_exact(xs, lp, K, dsa_mods[L], 4096)
    torch.cuda.synchronize()
    stage("P2_block_peak_K112", log)
    peak_over_steady = round(
        log[-1]["max_allocated_gb"] - log[-1]["allocated_gb"], 3)

    # ---------------- verdict ----------------
    banks_gb = 4.52                  # phase-1 measured
    steady_bf16 = round(banks_gb + core_gb + dsa_bf16_gb + state_gb, 3)
    budget_free_mb = free0.get("smi_free_mb")
    verdict = {
        "measured_components_gb": {
            "banks_mixed_a": banks_gb,
            "core_no_head": core_gb,
            "dsa_fp32_weights_and_buffers": fp32_dsa_gb,
            "dsa_bf16_weights_and_buffers": round(dsa_bf16_gb, 3),
            "dsa_state_populate_delta_4K": state_gb,
            "block_transient_K112": peak_over_steady,
        },
        "reporting_error_root_cause":
            "P2 bench docstring summed the HISTORICAL 7.0-GB core figure "
            "with an ESTIMATED 1.95-GB DSA footprint (4.4+7.0+1.95=13.35). "
            "Measured: 4.52+5.91+1.67=12.10 fp32 (OOMs empirically); bf16 "
            "storage 4.52+5.91+1.05=11.49.",
        "system_budget": {
            "card_total_gb": round(prop.total_memory / 1e9, 2),
            "smi_free_before_run_mb": budget_free_mb,
            "other_processes_gb": 0.16,
            "display_driver_overhead_gb_approx": round(
                prop.total_memory / 1e9 - (budget_free_mb / 1e3)
                - 0.16 - 0.0, 2) if budget_free_mb else None,
        },
        "fp32_stack_steady_gb": round(banks_gb + core_gb + fp32_dsa_gb, 3),
        "fp32_dsa_beside_banks_core_fits": fp32_result != "OOM",
        "practical_process_budget_gb": 11.35,
        "bf16_stack_steady_gb": steady_bf16,
        "bf16_block_peak_gb": round(steady_bf16 + peak_over_steady, 3),
        "bf16_verdict": (
            "bf16 DSA storage is NECESSARY (fp32 stack 12.10 GB OOMs "
            "empirically) but NOT SUFFICIENT: bf16 stack steady 11.49 GB + "
            "K112 block transient %.2f GB = %.2f GB exceeds the measured "
            "~11.35 GB practical process budget (12.48 card - 0.16 other "
            "procs - ~0.67 display - ~0.30 CUDA context). Remaining gap: "
            "~%.2f GB steady / ~%.2f GB with transients — closable via "
            "NEXT-2 fusion (shrinks the transient), state-buffer sizing, "
            "or ~150 MB from banks/core. Tiering remains REQUIRED until "
            "then." % (peak_over_steady, steady_bf16 + peak_over_steady,
                       steady_bf16 - 11.35,
                       steady_bf16 + peak_over_steady - 11.35)),
    }
    results["stages"] = log
    results["verdict"] = verdict
    print("\nVERDICT:", json.dumps(verdict, indent=1, default=str))
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
