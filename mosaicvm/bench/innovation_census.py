"""Mosaic Innovation Execution — exact input-innovation census.

Hypothesis (unproved as a speedup): an expert input that never repeats can
still be written x = B c + r at a source-compatible integer boundary, and a
small residual support can avoid reading most weight columns.

Pinned boundary
---------------
Source RoutedLinear L41/L42, ops w1/w2/w3. The GEMM consumes act_quant(x):
FP8 E4M3 codes q plus per-128 E8M0 scales sa. This census does NOT decompose
raw BF16 x and does NOT recombine BF16 outputs.

The integer identity used is the project's own MXINT K32 contract
(mosaicvm/kernels/mxint.py):

    qA = 512 * A_E4M3          exact integer, |qA| <= 229376
    qW = 2 * W_E2M1            s8 in [-12, 12]
    per-K32 |sum qA*qW| <= 51.38e6 < 2^31, so int32 accumulation is exact
    then ONE fp32 multiply by sw * sa / 1024

So the reusable object is the unscaled integer dot of one K32 block:

    dot(qA, qW) = Z c + dot(qA_r, qW_r)

with Z = qW @ B cached while the weights were present, and r the exact
integer residual of qA. Scales stay outside the cached dot and are applied
once, in source order. This census measures the INPUT side only. It does
not execute the reconstructed GEMM and does not claim source-output equality.

Accounting
----------
Ops accumulate across every trajectory. A per-trajectory clear is the bug
that made result_cache_census.py analyse 1,260 ops while reporting 5,151.

Storage is physical pages, not ideal column counts. One K32 group is
32 weight columns. Packed FP4 is 16 bytes/row/group plus 1 scale byte.
A 4096-byte page therefore holds 4096/17 = 240 output rows of one group
when the bank is stored group-major (the layout an action cache would
need). Row-major packed4 cannot skip a column without touching every row's
page; that layout is reported as "no column skip" and is not credited.

Action-cache bytes = 4 * n_rows_page_covered * r_b (int32 dots). The cache
is smaller than the group's weight bytes only when r_b < k/8. Here one
group has k = 32, so r_b < 4. Bases larger than that are recorded and then
rejected by the gate; they are not hidden.

Basis vectors come only from calls already seen. Residual is never dropped.
Reconstruction qA = B c + r is asserted with torch.equal on every call.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.kernels.fp4_source import decode_e4m3

OUT = Path("artifacts/innovation_census.json")
CAP_DIR = Path("artifacts/d3_scale1000_data")
N_PROMPTS = 4
# MXINT exactness holds per K32 group. k=32 => action cache beats the
# group's own weight bytes only for r_b < 32/8 = 4.
BLOCK = 32
MAX_BASIS = 3
# "Cheap" is not a correctness gate. It only decides whether this call is
# counted as served (residual retained) or admitted as a new basis vector.
SERVE_NNZ_MAX = 8
PAGE = 4096
# group-major record: 16 packed FP4 bytes + 1 E8M0 scale byte per row.
GROUP_STRIDE = 17
QA_SCALE = 512
QA_MAX = 229376


def make_probe():
    tail = SourceTail(device="cpu", backend="reference", max_seq=256, cache_cap=8)
    # (layer, expert, op, q_u8[rows, K], sa_u8[rows, K/128])
    tail.ops = []

    for layer, block in zip((41, 42), tail.blocks):
        for expert_id, expert in enumerate(block.ffn.experts):
            for name in ("w1", "w2", "w3"):
                mod = getattr(expert, name)
                orig = mod.forward

                def make_fw(ow, layer=layer, ex=expert_id, op=name):
                    def fw(x):
                        q, sa = tail.km.act_quant(
                            x, 128, tail.km.scale_fmt, tail.km.scale_dtype)
                        flat_q = q.reshape(-1, q.shape[-1]).detach().cpu()
                        flat_sa = sa.reshape(-1, sa.shape[-1]).detach().cpu()
                        tail.ops.append((
                            layer, ex, op,
                            flat_q.view(torch.uint8).clone(),
                            flat_sa.view(torch.uint8).clone(),
                        ))
                        return ow(x)
                    return fw

                mod.forward = make_fw(orig)
    return tail


def replay_capture(tail, cap):
    """Append. Never clear tail.ops — accumulation is the contract."""
    h40 = cap["h40"].to(dtype=torch.bfloat16)
    n = min(len(h40), tail.max_seq)
    h40 = h40[:n]
    toks = [int(t) for t in cap["tokens"][:n]]
    # Captures store the committed prefix length. Falling back to half the
    # trajectory would invent a split the source did not use.
    if "S" not in cap:
        raise KeyError("capture has no committed-prefix length S")
    S = int(cap["S"])
    if not 0 < S <= n:
        raise ValueError(f"committed prefix S={S} outside 1..{n}")
    with torch.inference_mode():
        tail.replay(h40, toks, S)


def qA_block(q_u8, col0, k=BLOCK):
    """One K32 slice of one row -> exact qA int64, or None if not integer.

    qA = round(512 * decode_e4m3(code)). The MXINT note says this is exact
    and |qA| <= 229376. A code that is non-finite, non-integer after the
    scale, or outside that range is rejected — never approximated.
    """
    codes = q_u8[col0:col0 + k]
    if codes.numel() != k:
        return None
    decoded = decode_e4m3(codes.contiguous())
    scaled = decoded * QA_SCALE
    rounded = torch.round(scaled)
    if not bool(torch.isfinite(decoded).all()):
        return None
    if not bool(torch.equal(scaled, rounded)):
        return None
    qA = rounded.to(torch.int64)
    if int(qA.abs().max()) > QA_MAX:
        return None
    return qA


def exact_decompose(x, basis):
    """x = B c + r, residual kept in full, reconstruction asserted.

    c is the integer rounding of a float64 least-squares fit against basis
    vectors already seen. r = x - B c exactly in int64. If the rounded
    combination overflows int64 or fails equality, the call is refused.
    """
    if not basis:
        return torch.zeros(0, dtype=torch.int64), x.clone()
    B = torch.stack(basis).to(torch.float64).T          # [k, r_b]
    xf = x.to(torch.float64)
    sol = torch.linalg.lstsq(B, xf.unsqueeze(1), driver="gelsd").solution
    sol = sol.squeeze(1)
    ci = torch.round(sol).to(torch.int64)
    # Reconstruct in int64. A residual that is not an integer means the
    # rounded combination left the exact lattice; refuse it.
    combo = B.to(torch.int64) @ ci
    r = x - combo
    if not torch.equal(combo + r, x):
        raise AssertionError("exact reconstruction failed")
    return ci, r


def page_rows(n_rows):
    """Output rows of one K32 group that share a page under group-major layout."""
    rows_per_page = PAGE // GROUP_STRIDE          # 240
    return rows_per_page


def census_group(calls, n_rows):
    """Chronological bounded basis. Residual always retained.

    A call is "served" when it decomposes against the current basis with
    nnz(r) <= SERVE_NNZ_MAX. Otherwise it is admitted (FIFO once the basis
    is full). Served does not mean weights were skipped: the page model
    below decides whether the residual columns are actually fewer pages.
    """
    basis = []
    rows = []
    for x in calls:
        c, r = exact_decompose(x, basis)
        nnz = int((r != 0).sum())
        cols = torch.nonzero(r, as_tuple=False).flatten().tolist()
        cheap = bool(basis) and nnz <= SERVE_NNZ_MAX
        if not cheap:
            if len(basis) >= MAX_BASIS:
                basis.pop(0)
            basis.append(x.clone())
        rows.append({
            "served": cheap,
            "residual_nnz": nnz,
            "residual_cols": cols,
            "basis_size": len(basis),
        })
    return rows, len(basis)


def physical_bytes(rows, n_rows):
    """Bytes of one K32 group actually touched, group-major, 4 KiB pages.

    Full group: every row, one scale byte + 16 code bytes.
    Served call: only pages that contain a residual column. Because the
    group is stored row-major *inside the group* is NOT the model — each
    column is a contiguous run of n_rows strides only if we store
    column-major within the group. We store column-major within the 32:
    column j occupies bytes [j * n_rows * GROUP_STRIDE, (j+1) * ...).
    A residual with nnz columns then touches nnz separate ranges.

    Scales live in the same per-column stride (the +1 in GROUP_STRIDE),
    so they are inside the charged pages, not a second pass.

    Action cache: int32 dot per output row per basis vector. Charged for
    every basis vector present at the call, for the rows this page covers.
    Compared against the same group's weight bytes, not the whole matrix.
    """
    stride = n_rows * GROUP_STRIDE
    full_pages = (BLOCK * stride + PAGE - 1) // PAGE
    full_bytes = full_pages * PAGE
    weight_bytes = BLOCK * stride                 # unpacked-to-page before pad

    served_bytes = 0
    served_n = 0
    basis_bytes_at_serve = 0
    zero = 0
    for row in rows:
        if not row["served"]:
            continue
        served_n += 1
        nnz = row["residual_nnz"]
        if nnz == 0:
            zero += 1
            touched = 0
        else:
            # each residual column is its own contiguous stride; no sharing
            # across columns. Pages cannot be shared between columns.
            touched = nnz * ((stride + PAGE - 1) // PAGE) * PAGE
        served_bytes += touched
        # int32 accumulator per output row per basis vector
        basis_bytes_at_serve += row["basis_size"] * n_rows * 4

    return {
        "n_rows": n_rows,
        "group_weight_bytes": weight_bytes,
        "group_weight_pages": full_pages,
        "group_paged_bytes": full_bytes,
        "served_calls": served_n,
        "served_residual_paged_bytes": served_bytes,
        "served_action_cache_bytes": basis_bytes_at_serve,
        "zero_residual_served": zero,
        "rows_per_page_if_row_packed": PAGE // GROUP_STRIDE,
    }


def main():
    caps = sorted(CAP_DIR.glob("*.pt"))[:N_PROMPTS]
    if not caps:
        raise SystemExit(f"no captures in {CAP_DIR}")
    print(f"innovation census: {len(caps)} trajectories, block {BLOCK}, "
          f"basis cap {MAX_BASIS}", flush=True)
    tail = make_probe()
    for cp in caps:
        before = len(tail.ops)
        cap = torch.load(cp, weights_only=False, map_location="cpu")
        replay_capture(tail, cap)
        print(f"  {cp.name}: +{len(tail.ops) - before} ops "
              f"(running {len(tail.ops)})", flush=True)

    # Grain: one activation row of one (layer, expert, op, K32 block).
    # The probe flattens each forward first. Captured decode forwards are
    # [1, K]; a prefill forward with S rows is S separate rows of the SAME
    # weight stripe. Page bytes below use the op's full output width, so
    # summing those rows counts the stripe once per activation row. That is
    # an over-count when S>1. The run records op shapes so the over-count
    # is visible; decode-only rows ([1, K]) are not over-counted.
    groups = defaultdict(list)
    n_ops = len(tail.ops)
    n_ineligible = 0
    shapes = defaultdict(int)
    for layer, ex, op, q_u8, sa_u8 in tail.ops:
        shapes[f"{op}:{tuple(q_u8.shape)}"] += 1
        rows, k = q_u8.shape
        if k % BLOCK != 0:
            n_ineligible += rows
            continue
        for r in range(rows):
            for col0 in range(0, k, BLOCK):
                qA = qA_block(q_u8[r], col0)
                if qA is None:
                    n_ineligible += 1
                    continue
                groups[(layer, ex, op, col0)].append(qA)

    print(f"ops {n_ops}; row-blocks kept {sum(len(v) for v in groups.values())}; "
          f"ineligible {n_ineligible}; groups {len(groups)}", flush=True)

    # Representative output width: w1/w3 write 2048, w2 writes 4096.
    # Page accounting uses that width because a K32 group is one stripe
    # across every output row of the op. We do not have per-row outputs
    # here; the stripe width is the op's N.
    N_OF = {"w1": 2048, "w2": 4096, "w3": 2048}

    per_group = {}
    totals = {
        "calls": 0,
        "served": 0,
        "basis_adds": 0,
        "zero_residual": 0,
        "weight_bytes": 0,
        "served_residual_paged_bytes": 0,
        "served_action_cache_bytes": 0,
        "full_paged_bytes_of_served": 0,
    }
    nnz_hist = defaultdict(int)
    basis_hist = defaultdict(int)

    for key, calls in groups.items():
        layer, ex, op, col0 = key
        if len(calls) < 2:
            continue
        n_rows = N_OF[op]
        detail, final_basis = census_group(calls, n_rows)
        phys = physical_bytes(detail, n_rows)
        served = phys["served_calls"]
        adds = len(detail) - served
        totals["calls"] += len(detail)
        totals["served"] += served
        totals["basis_adds"] += adds
        totals["zero_residual"] += phys["zero_residual_served"]
        totals["weight_bytes"] += phys["group_weight_bytes"]
        totals["served_residual_paged_bytes"] += phys["served_residual_paged_bytes"]
        totals["served_action_cache_bytes"] += phys["served_action_cache_bytes"]
        totals["full_paged_bytes_of_served"] += served * phys["group_paged_bytes"]
        for row in detail:
            nnz_hist[row["residual_nnz"]] += 1
            basis_hist[row["basis_size"]] += 1
        gid = f"{layer}_{ex}_{op}_k{col0}"
        per_group[gid] = {
            "n_calls": len(detail),
            "final_basis": final_basis,
            "n_served": served,
            "n_basis_adds": adds,
            "zero_residual_served": phys["zero_residual_served"],
            "served_residual_paged_bytes": phys["served_residual_paged_bytes"],
            "served_action_cache_bytes": phys["served_action_cache_bytes"],
            "group_paged_bytes": phys["group_paged_bytes"],
        }

    served = totals["served"]
    # Net bytes for served calls = residual pages + the action-cache dots
    # those calls required. Compared with reading the whole group.
    net_served = (totals["served_residual_paged_bytes"]
                  + totals["served_action_cache_bytes"])
    avoided = totals["full_paged_bytes_of_served"] - net_served
    avoided_pct = (100.0 * avoided / totals["full_paged_bytes_of_served"]
                   if totals["full_paged_bytes_of_served"] else 0.0)
    # Against ALL group traffic, not just the served subset. Unserved calls
    # read the whole group (they were admitted) plus they built the cache.
    all_full = 0
    for g in per_group.values():
        all_full += g["n_calls"] * g["group_paged_bytes"]
    # Admitted calls pay full pages. Served calls pay residual pages.
    # Action cache is paid on top for served calls. Basis-build compute is
    # not storage; the call that was admitted already paid its full read,
    # which is the build cost in bytes.
    net_all = (all_full
               - totals["full_paged_bytes_of_served"]
               + net_served)
    avoided_all = all_full - net_all
    avoided_all_pct = 100.0 * avoided_all / all_full if all_full else 0.0

    storage_gate = (
        f"one K32 group, k={BLOCK}: action cache beats that group's weight "
        f"bytes only when r_b < {BLOCK // 8}. Basis cap is {MAX_BASIS}."
    )
    print(storage_gate, flush=True)
    print(f"served {served}/{totals['calls']} "
          f"({100.0 * served / max(totals['calls'], 1):.2f}%)", flush=True)
    print(f"zero-residual served {totals['zero_residual']}", flush=True)
    print(f"avoided among served calls {avoided_pct:.2f}% "
          f"(after residual pages + action cache)", flush=True)
    print(f"avoided among ALL calls {avoided_all_pct:.2f}%", flush=True)
    print(f"bar: 86-94% of expert weight bytes. "
          f"this census is K32-group grain, L41/L42 only.", flush=True)

    out = {
        "classification": "INNOVATION_CENSUS",
        "evidence_class": "SIMULATED",
        "hypothesis": (
            "expert inputs may never repeat, yet their exact qA innovations "
            "on a K32 MXINT block might be cheap"
        ),
        "boundary": (
            "source RoutedLinear L41/L42 w1/w2/w3; operand is act_quant q; "
            "integer identity is MXINT qA = 512 * A_E4M3 on one K32 group; "
            "scales applied once after the integer dot, not inside it"
        ),
        "not_claimed": (
            "no reconstructed GEMM was compared to source output; no BF16 "
            "recombination; no tok/s; page model assumes column-major layout "
            "within each K32 group, which the resident packed4 bank does not use"
        ),
        "n_trajectories": len(caps),
        "captures": [p.name for p in caps],
        "total_expert_ops": n_ops,
        "op_shapes": dict(shapes),
        "row_block_calls": totals["calls"],
        "row_blocks_ineligible": n_ineligible,
        "groups_tested": len(per_group),
        "calls_served": served,
        "calls_basis_adds": totals["basis_adds"],
        "zero_residual_served": totals["zero_residual"],
        "basis_cap": MAX_BASIS,
        "serve_nnz_max": SERVE_NNZ_MAX,
        "block": BLOCK,
        "storage_gate": storage_gate,
        "page_bytes": PAGE,
        "group_stride_bytes": GROUP_STRIDE,
        "layout_assumed": (
            "column-major within a K32 group: column j is n_rows*(16+1) "
            "contiguous bytes. Row-major packed4 cannot skip columns; that "
            "layout saves 0 bytes and is not what these numbers credit."
        ),
        "served_only_avoided_pct": round(avoided_pct, 4),
        "all_calls_avoided_pct": round(avoided_all_pct, 4),
        "all_calls_full_paged_bytes": all_full,
        "all_calls_net_bytes": net_all,
        "bar_avoidance_pct": "86-94",
        "nnz_hist": {str(k): nnz_hist[k] for k in sorted(nnz_hist)},
        "basis_size_hist": {str(k): basis_hist[k] for k in sorted(basis_hist)},
        "groups": per_group,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
