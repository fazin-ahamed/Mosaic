"""NEXT 1+2: matrix-selective single-call hybrid — parity + K56 golden timing.

gu and down stages inside fused_moe_mxint are already separate kernel
launches, so mixed mode (gu_bank/down_bank kwargs) adds ZERO launches.

Direction evidence (artifacts/moe_profile.json):
  hot8 wins DOWN (8.7 vs 12.0 ms/layer); packed4 wins GU (24.2 vs 25.8)
=> Mixed-A (gu=packed4, down=hot8) is profile-favored AND VRAM-fitting
   (8.39+8.39 MB/e ~ 4.3 GB banks + 7.0 core = 11.3 GB <= 11.6).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
os.environ.setdefault("MOSAICVM_EXACT_TARGET", "1")

from mosaicvm.bench.wide_verify import _bench_blocks, _golden_windows
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.kernels.exact_moe import exact_moe
from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
from mosaicvm.kernels.mxint import (
    MXINTExpertBank, MXINTPacked4Bank, fused_moe_mxint, qint_encode_w,
)

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/matrix_hybrid.json")
CORE_PROD = 0.173


def build_shared(n_e, h, inter, seed=7):
    """One deterministic source -> (packed4 bank, hot8 gu codes, hot8 down codes)."""
    torch.manual_seed(seed)
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(n_e):
        pg, sg = pack_mxfp4_nk(
            (torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg)
        pd, sd = pack_mxfp4_nk(
            (torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pdu.append(pd); sdu.append(sd)
    PG, SG = torch.stack(pgu), torch.stack(sgu)
    PD, SD = torch.stack(pdu), torch.stack(sdu)
    p4 = MXINTPacked4Bank(p_gu=PG, s_gu=SG, p_d=PD, s_d=SD,
                          n_out_gu=2 * inter, n_out_d=h)
    WGU = torch.stack([qint_encode_w(PG[i]) for i in range(n_e)])
    WD = torch.stack([qint_encode_w(PD[i]) for i in range(n_e)])
    return p4, WGU, WD, SG, SD


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    h, inter = 4096, 2048
    golden = json.loads(GOLDEN.read_text())
    gw = _golden_windows(56, golden)

    # ---------- parity (64 experts, shared source) ----------
    p4s, WGU, WD, SG, SD = build_shared(64, h, inter)
    h8_down = MXINTExpertBank(
        w_gu=WGU[:1].clone(), s_gu=SG[:1].clone(),
        w_d=WD, s_d=SD, n_out_gu=2 * inter, n_out_d=h)
    h8_gu = MXINTExpertBank(
        w_gu=WGU, s_gu=SG,
        w_d=WD[:1].clone(), s_d=SD[:1].clone(),
        n_out_gu=2 * inter, n_out_d=h)
    del WGU, WD
    torch.cuda.empty_cache()

    print("=== parity: mixed vs exact_moe (shared source, 64 experts) ===")
    n_ok = 0
    for i in range(4):
        ids, wts, _ = gw[(i * 11) % len(gw)]
        ids_t = torch.tensor(ids, device="cuda")
        if ids_t.dim() == 1:
            ids_t = ids_t.reshape(-1, 1)
        ids_r = ids_t % 64
        wts_g = torch.tensor(wts, device="cuda", dtype=torch.float32)
        x = torch.randn(56, h, dtype=torch.bfloat16, device="cuda")
        y_ref = exact_moe(x, ids_r, wts_g, p4s.p_gu, p4s.s_gu, p4s.p_d, p4s.s_d)
        yA = fused_moe_mxint(x, ids_r, wts_g, gu_bank=p4s, down_bank=h8_down,
                             crossover=0)
        yB = fused_moe_mxint(x, ids_r, wts_g, gu_bank=h8_gu, down_bank=p4s,
                             crossover=0)
        eqA = torch.equal(yA, y_ref)
        eqB = torch.equal(yB, y_ref)
        n_ok += eqA and eqB
        print(f"  win{i}: A {'BITWISE-EQ' if eqA else 'DIFF'}"
              f"  B {'BITWISE-EQ' if eqB else 'DIFF'}")
    print(f"parity: {n_ok}/4")
    del p4s, h8_down, h8_gu, SG, SD
    torch.cuda.empty_cache()

    # ---------- K56 golden timing (256-expert banks, VRAM-fit partials) ----------
    warmup_steady(120.0)
    from mosaicvm.bench.phase29r import _allocate_packed4_experts
    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
    p4_full = _allocate_packed4_experts(N_EXPERTS)
    h8_full = _allocate_mxint_experts(N_EXPERTS)
    h8_down_full = MXINTExpertBank(
        w_gu=h8_full.w_gu[:1].clone(), s_gu=h8_full.s_gu[:1].clone(),
        w_d=h8_full.w_d, s_d=h8_full.s_d,
        n_out_gu=h8_full.n_out_gu, n_out_d=h8_full.n_out_d)
    del h8_full
    p4_gu_only = MXINTPacked4Bank(
        p_gu=p4_full.p_gu, s_gu=p4_full.s_gu,
        p_d=p4_full.p_d[:1].clone(), s_d=p4_full.s_d[:1].clone(),
        n_out_gu=p4_full.n_out_gu, n_out_d=p4_full.n_out_d)
    # full packed4 for the pure baseline
    del p4_full
    torch.cuda.empty_cache()
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"\nbanks (p4_gu 2.15GB + h8_down 2.15GB): {gb:.2f} GB "
          f"(+core 7.0 = {7.0 + gb:.2f} <= 11.6)")

    routes = [gw[(i * 7) % len(gw)] for i in range(11 * N_LAYERS)]
    xs = [torch.randn(56, h, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    ids_t = [torch.tensor(r[0], device="cuda", dtype=torch.int64) for r in routes]
    w_t = [torch.tensor(r[1], device="cuda", dtype=torch.float32) for r in routes]

    # Mixed-A timing + stage split
    agg = {"gu_sec": [], "down_sec": [], "total_sec": []}
    idx = 0

    def step_mixed():
        nonlocal idx
        for _ in range(N_LAYERS):
            st = {}
            fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx],
                            gu_bank=p4_gu_only, down_bank=h8_down_full,
                            crossover=0, stage_times=st)
            if "gu_sec" in st:
                agg["gu_sec"].append(st["gu_sec"])
                agg["down_sec"].append(st["down_sec"])
                agg["total_sec"].append(st["total_sec"])
            idx += 1

    t0 = snapshot()
    t_mixed = _bench_blocks(step_mixed, 10)
    t1 = snapshot()
    gu_ms = statistics.median(agg["gu_sec"]) * 1e3
    dn_ms = statistics.median(agg["down_sec"]) * 1e3
    tot_ms = statistics.median(agg["total_sec"]) * 1e3

    print(f"\nMixed-A (gu=p4, down=h8) K56 golden: {t_mixed:.4f} s")
    print(f"  per-layer: gu={gu_ms:.2f} down={dn_ms:.2f} total={tot_ms:.2f} ms")
    print(f"  R_oracle={28.27 / (t_mixed + CORE_PROD):.1f}/s (prodcore)")
    print(f"  refs: p4 0.9545 | h8 0.6491 | budget {0.9423 - CORE_PROD:.3f}")

    OUT.write_text(json.dumps({
        "banks_gb": gb, "vram_total_with_core": 7.0 + gb,
        "mixed_A_gu_p4_down_h8": {
            "moe_block_s": t_mixed,
            "per_layer_ms": {"gu": gu_ms, "down": dn_ms, "total": tot_ms},
            "R_oracle_prodcore": 28.27 / (t_mixed + CORE_PROD),
        },
        "reference": {"packed4_pure": 0.9545, "hot8_pure": 0.6491,
                      "moe_budget_for_exact30": 0.9423 - CORE_PROD},
        "thermal": {"start": t0, "end": t1},
        "parity": f"{n_ok}/4 windows bitwise both directions",
    }, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
