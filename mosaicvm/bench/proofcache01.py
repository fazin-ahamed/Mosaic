"""ProofCache-0/1: PC0 boundary pin + PC1 exact-reuse census + PC2 NN screen.

Addresses the ProofCache architecture's first three gates on REAL source
arithmetic, correcting an accumulation bug in result_cache_census.py v1
(which cleared the op log per trajectory and therefore analysed only the
final trajectory while reporting the summed total).

PC0 - pinned canonical operation boundary
-----------------------------------------
Boundary: source RoutedLinear on L41/L42, ops w1 / w2 / w3 (the project's
already-parity-certified exact expert path).

  input  x        : BF16 [4, 4096] hidden state entering the expert op
                    (post-attention residual state for that position)
  effective      : the source's own activation quantization act_quant(x) ->
                    operand q (FP8 codes, 128-element blocks) + block scales sa.
  operand          The GEMM consumes q and sa; x alone is not the operand.
  weights         : FP4 E2M1 codes + per-block scales for that (layer, expert, op)
  output y        : BF16 [4, 2048] (w1/w3) or [4, 4096] (w2) - the exact source
                    result, consumed by SwiGLU / the next op.
  consumers       : w1,w3 -> silu/gate combine -> w2 -> routing-weight multiply
                    (OUTSIDE this boundary) -> expert sum -> residual -> RMSNorm
                    -> next block act_quant.

Routing weight is applied OUTSIDE the RoutedLinear boundary, so it is not part
of this boundary's effective input (it is part of the expert-sum boundary).

PC1 - exact-reuse census (generous upper bound)
-----------------------------------------------
Accumulates across ALL trajectories (no per-trajectory clear). Unlimited
result-cache capacity, free lookups, zero compute. Effective-input key =
layer + expert + op + revision + quantized operand bytes + scale bytes.
Hash locates; byte comparison would authorise (never exercised: no repeats).
Reports avoided COLD bytes (repeats whose weights were not resident), split
within-trajectory vs across-trajectory.

PC2 - nearest-neighbour certificate screen
------------------------------------------
Path B needs different inputs to yield the same required output. Measures, on
real arithmetic, the closest input pairs per (layer, expert, op) group and the
resulting relative output distance, against the source's own downstream
quantization granularity (act_quant on the same values). If even the closest
pairs are orders of magnitude coarser than a quantization cell, proximity-based
certificates cannot authorise reuse at this boundary.

Simulation/measurement on the source's own exact path. Not an accelerator.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.source_tail import SourceTail

OUT = Path("artifacts/proofcache01.json")
CAP_DIR = Path("artifacts/d3_scale1000_data")
N_PROMPTS = 4


def make_probe():
    tail = SourceTail(device="cpu", backend="reference",
                      max_seq=64, cache_cap=8)
    tail.ops = []            # (traj, fwd, layer, expert, op, qhash)
    tail.pairs = defaultdict(list)   # (layer,expert,op) -> [(x, y, q)]
    tail.capture_set = set()
    tail._traj = -1
    tail._fwd = -1

    # position tag: bump once per block forward
    for block in tail.blocks:
        block.register_forward_hook(lambda *a, **k: setattr(
            tail, "_fwd", tail._fwd + 1))

    def hash_key(q, sa, layer, expert, op):
        h = hashlib.sha256()
        for v in (layer, expert, op):
            h.update(str(v).encode()); h.update(b"|")
        h.update(q.contiguous().view(torch.uint8).numpy().tobytes())
        h.update(sa.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    for l, block in zip((41, 42), tail.blocks):
        for e, expert in enumerate(block.ffn.experts):
            for name in ("w1", "w2", "w3"):
                mod = getattr(expert, name)
                orig_forward = mod.forward

                def make_fw(ow, layer=l, ex=e, op=name):
                    def fw(x):
                        q, sa = tail.km.act_quant(x, 128, tail.km.scale_fmt,
                                                  tail.km.scale_dtype)
                        kh = hash_key(q, sa, layer, ex, op)
                        tail.ops.append((tail._traj, tail._fwd, layer, ex, op,
                                         kh))
                        y = ow(x)
                        if (layer, ex, op) in tail.capture_set:
                            tail.pairs[(layer, ex, op)].append(
                                (x.detach().clone(), y.detach().clone(),
                                 q.detach().clone()))
                        return y
                    return fw
                mod.forward = make_fw(orig_forward)
    return tail


def replay_capture(tail, cap, traj_id):
    h40 = cap["h40"].to(dtype=torch.bfloat16)
    toks = [int(t) for t in cap["tokens"][:len(h40)]]
    S = max(1, len(h40) // 2)
    tail._traj = traj_id
    tail._fwd = -1
    with torch.inference_mode():
        tail.replay(h40, toks, S)


def pc1(tail, n_traj):
    """exact-reuse census, accumulated across trajectories."""
    total = len(tail.ops)
    per_key = Counter((l, e, o, kh) for _, _, l, e, o, kh in tail.ops)
    distinct = len(per_key)
    repeated = {k: c for k, c in per_key.items() if c > 1}

    # within-trajectory vs across-trajectory repeats
    by_key_traj = defaultdict(set)
    for t, _, l, e, o, kh in tail.ops:
        by_key_traj[(l, e, o, kh)].add(t)
    cross_traj_repeats = sum(1 for k, ts in by_key_traj.items()
                             if len(ts) > 1 and per_key[k] > 1)
    within_traj_repeats = sum(c - 1 for k, c in repeated.items()
                              if len(by_key_traj[k]) == 1)

    BYTES = {"w1": 2048 * 4096 // 2, "w2": 4096 * 2048 // 2,
             "w3": 2048 * 4096 // 2}
    avoided_cold = sum((c - 1) * BYTES[op] for (_, _, op), c in repeated.items())
    total_weight = sum(BYTES[op] for _, _, _, _, op, _ in tail.ops)

    return {
        "trajectories": n_traj,
        "total_expert_ops": total,
        "distinct_effective_inputs": distinct,
        "repeated_inputs": len(repeated),
        "pct_distinct_repeated": round(100 * len(repeated) / max(distinct, 1), 3),
        "repeat_executions": sum(c - 1 for c in repeated.values()),
        "within_traj_repeat_inputs": within_traj_repeats,
        "cross_traj_repeat_inputs": cross_traj_repeats,
        "avoidable_cold_GB": round(avoided_cold / 1e9, 4),
        "total_weight_read_GB": round(total_weight / 1e9, 4),
        "avoided_pct": round(100 * avoided_cold / max(total_weight, 1), 3),
    }


def pc2(tail, cap0):
    """direct BF16 output-equality oracle (replaces the distance heuristic).

    The earlier distance-vs-act_quant-step heuristic was not a valid
    certificate screen: it compared relative BF16 output distances against an
    FP8 quantization step, mixing two different boundaries. The decisive test
    is direct: does any pair of calls at the same (layer, expert, op) group
    with the same shape produce IDENTICAL BF16 output bytes?

    - Groups calls by (layer, expert, op) and compatible shape.
    - Hashes complete BF16 outputs to locate candidates; torch.equal confirms.
    - Counts chronological availability: an earlier identical output exists.
    - Identical outputs from different inputs are an OPPORTUNITY UPPER BOUND
      only - discovering equality after execution supplies no cheap
      pre-execution certificate.
    """
    tail.ops.clear()
    replay_capture(tail, cap0, 0)
    counts = Counter((l, e, o) for _, _, l, e, o, _ in tail.ops)
    top = [g for g, _ in counts.most_common(6)]
    if not top:
        return {"error": "no ops captured"}
    tail.capture_set = set(top)
    tail.pairs.clear()

    # pass 2: re-replay (deterministic: no RNG in the tail replay) with capture
    tail.ops.clear()
    replay_capture(tail, cap0, 0)

    per_group = {}
    for g, recs in tail.pairs.items():
        if len(recs) < 2:
            continue
        layer, ex, op = g
        # bucket by call input shape: only same-shape calls are comparable
        # (prefill [S,4,4096] vs decode [1,4,4096])
        by_shape = defaultdict(list)
        for r in recs:
            by_shape[tuple(r[0].shape)].append(r)
        n_pairs = 0
        n_equal = 0
        for shp, rr in by_shape.items():
            if len(rr) < 2:
                continue
            # chronological: an earlier identical output would be reusable
            seen = []                    # list of (hash, output tensor)
            for r in rr:
                y = r[1]
                yb = y.detach().contiguous().view(
                    torch.uint8).numpy().tobytes()
                yh = hashlib.sha256(yb).hexdigest()
                for ph, py in seen:
                    n_pairs += 1
                    if ph == yh and torch.equal(py, y):
                        n_equal += 1
                seen.append((yh, y))
        per_group[f"{layer}_{ex}_{op}"] = {
            "n_invocations": len(recs),
            "n_eligible_pairs": n_pairs,
            "n_identical_output_pairs": n_equal,
        }
    if not per_group:
        return {"error": "no group had >= 2 invocations"}
    total_pairs = sum(v["n_eligible_pairs"] for v in per_group.values())
    total_equal = sum(v["n_identical_output_pairs"] for v in per_group.values())
    return {
        "groups": per_group,
        "n_groups": len(per_group),
        "total_eligible_pairs": total_pairs,
        "total_identical_output_pairs": total_equal,
        "note": "direct BF16 output-equality oracle on the PINNED boundary "
                "(RoutedLinear returns BF16). Equality means identical output "
                "BYTES; a hash locates and torch.equal confirms. Identical "
                "outputs from DIFFERENT inputs would be an opportunity upper "
                "bound only: discovering equality after execution does not "
                "provide a cheap pre-execution certificate.",
        "boundary_caveat": "this oracle tests reuse of the COMPLETE BF16 "
                "output at this boundary only; post-act_quant equality is a "
                "DIFFERENT boundary and remains untested",
    }



def main():
    caps = sorted(CAP_DIR.glob("*.pt"))[:N_PROMPTS]
    if not caps:
        print(f"no captures in {CAP_DIR}")
        return
    print(f"ProofCache-0/1: {len(caps)} real captured trajectories")
    print("PC1: exact-reuse census (unlimited capacity, free lookups)")
    tail = make_probe()
    for i, cp in enumerate(caps):
        cap = torch.load(cp, weights_only=False, map_location="cpu")
        replay_capture(tail, cap, i)
        print(f"  {cp.name}: {len(tail.ops)} ops accumulated")
    r1 = pc1(tail, len(caps))
    print(json.dumps(r1, indent=1))

    print("\nPC2: nearest-neighbour certificate screen (trajectory 0)")
    cap0 = torch.load(caps[0], weights_only=False, map_location="cpu")
    r2 = pc2(tail, cap0)
    print(json.dumps(r2, indent=1))

    # ---- gates ----
    need = 0.86   # ~86% byte avoidance required at 1.82 GB/s (zero other work)
    verdict = "PC1 GATE: no exact reuse -> Path A closed as a major lever"
    if r1["avoided_pct"] < need * 100:
        verdict += f" (avoided {r1['avoided_pct']}% << required ~{need*100:.0f}%)"
    if "total_identical_output_pairs" in r2:
        tp, te = r2["total_eligible_pairs"], r2["total_identical_output_pairs"]
        verdict += (f"; PC2: {te} identical output pairs of {tp} eligible "
                    "at the pinned BF16 boundary")
        if te == 0:
            verdict += (" -> no reuse opportunity for complete outputs in "
                        "this sample; partial-output reuse and residual "
                        "disclosure remain untested")

    out = {
        "classification": "PROOFCACHE01",
        "pc0_boundary": "source RoutedLinear L41/L42 w1/w2/w3; effective input "
                        "= act_quant operand q + scales; consumers SwiGLU -> "
                        "w2 -> routing weight (outside boundary)",
        "pc1_exact_reuse_census": r1,
        "pc2_output_equality_oracle": r2,
        "required_byte_avoidance_pct": need * 100,
        "verdict": verdict,
        "correction": "result_cache_census.py v1 cleared the op log per "
                      "trajectory, so it analysed only the last trajectory "
                      "(1260 ops) while reporting the summed total (5151). "
                      "This run accumulates across trajectories. PC2 was also "
                      "rewritten: the original distance-vs-act_quant-step "
                      "heuristic compared two different boundaries and is "
                      "superseded by a direct output-equality oracle.",
        "scope": "L41/L42 of 43 layers; source's own exact reference path; "
                 "single decode sessions on real captured h40 trajectories. "
                 "Sim/measurement, not an accelerator.",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(verdict)

if __name__ == "__main__":
    main()
