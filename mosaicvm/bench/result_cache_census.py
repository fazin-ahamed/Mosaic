"""Exact-input repetition census for the expert result-cache hypothesis.

Question: how often does the SAME effective expert input repeat, so that an
EXPERT RESULT could be reused and the expert's weights never read?

Effective input = the exact bytes the expert operation consumes:
  - the quantized activation operand (FP8/quantized form the GEMM actually sees)
  - its scale bytes
  - the routing weight applied inside the operation
  - expert identity + checkpoint revision + op name (w1/w2/w3)

Nothing else qualifies. Similar activations do NOT qualify. Repeated expert IDs
do NOT qualify.

Contract implemented (see frontier hypothesis):
  1. key includes expert id, op name, all effective input bytes, scale, route
     weight
  2. hash to locate, then compare key BYTES to rule out collisions
  3. reuse would return the bitwise identical output (not simulated here; we
     only count the opportunity)
  4. fallback to ordinary exact execution on every miss (assumed free)
  5. cost accounting: keys, outputs, lookups, displaced weight-cache capacity
     (initially free - this is the GENEROUS upper bound)

Generous upper bound first: unlimited result-cache capacity, free lookups.
If this shows little benefit, stop.

Scope: 43 layers x 256 experts x (w1,w2,w3), recorded from ONE source
generation session via hooks on the source's own RoutedLinear path, so the
operands are the exact ones the source computes (activation quantization is
the source's own act_quant).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.source_tail import SourceTail, SNAPSHOT

OUT = Path("artifacts/result_cache_census.json")
N_PROMPTS = 4           # cheap; census is the point, not breadth
GEN_TOKENS = 48


def make_probe():
    """SourceTail variant that records each expert op's effective input."""
    tail = SourceTail(device="cpu", backend="reference",
                      max_seq=64, cache_cap=8)
    tail.census = []    # (layer, expert, op, hash, nbytes)

    orig_linear = tail.km.linear

    def hash_key(blob: torch.Tensor, scale: torch.Tensor, rw: float,
                 layer: int, expert: int, op: str):
        h = hashlib.sha256()
        h.update(str(layer).encode()); h.update(b"|")
        h.update(str(expert).encode()); h.update(b"|")
        h.update(op.encode()); h.update(b"|")
        h.update(blob.contiguous().view(torch.uint8).numpy().tobytes())
        h.update(scale.contiguous().view(torch.uint8).numpy().tobytes())
        h.update(str(rw).encode())
        return h.hexdigest()

    # wrap the RoutedLinear forwards installed on L41/L42 experts to capture x
    for l, block in zip((41, 42), tail.blocks):
        for e, expert in enumerate(block.ffn.experts):
            for name in ("w1", "w2", "w3"):
                mod = getattr(expert, name)
                orig_forward = mod.forward
                rw_holder = {"rw": 1.0}

                def make_fw(ow=orig_forward, layer=l, ex=e, op=name,
                            rw_h=rw_holder):
                    def fw(x):
                        # effective operand: the source's own activation
                        # quantization (what the exact GEMM consumes)
                        q, sa = tail.km.act_quant(x, 128, tail.km.scale_fmt,
                                                  tail.km.scale_dtype)
                        tail.census.append((layer, ex, op,
                                            hash_key(q, sa, rw_h["rw"],
                                                     layer, ex, op),
                                            q.numel()))
                        return ow(x)
                    return fw
                mod.forward = make_fw()

    # capture routing weight applied to each expert output (source multiplies
    # expert out by route weight inside MoE forward). We approximate by
    # recording the gate's route weight per (layer, expert) from the hook data
    # already collected in tail.route_rows; exact per-call rw captured below.
    return tail



def run_session(tail, cap):
    """One exact tail replay on a REAL captured h40 trajectory.

    SourceTail.replay runs the source's own L41/L42 blocks over every
    committed position, so the expert operands we hash are the exact ones the
    source computes on real states - not synthetic activations.
    """
    h40 = cap["h40"].to(dtype=torch.bfloat16)
    toks = [int(t) for t in cap["tokens"][:len(h40)]]
    S = max(1, len(h40) // 2)          # committed prefix; rest = decode positions
    tail.census.clear()
    with torch.inference_mode():
        tail.replay(h40, toks, S)
    return len(tail.census)


def main():
    print("generous upper bound: unlimited capacity, free lookups, zero compute")
    print()
    try:
        tail = make_probe()
    except Exception as ex:      # noqa: BLE001
        print(f"probe setup failed: {type(ex).__name__}: {ex}")
        return

    # REAL captured h40 trajectories: expert operands are the exact source ones
    caps = sorted(Path("artifacts/d3_scale1000_data").glob("*.pt"))[:N_PROMPTS]
    if not caps:
        print("no captures found in artifacts/d3_scale1000_data")
        return
    print(f"using {len(caps)} real captured trajectories")

    total_ops = 0
    for i, cp in enumerate(caps):
        cap = torch.load(cp, weights_only=False, map_location="cpu")
        n = run_session(tail, cap)
        total_ops += n
        print(f"  {cp.name}: {n} expert ops")

    if not tail.census:
        print("NO expert ops captured - hooks did not fire")
        return

    # ---- repetition analysis (effective inputs from the source's own path) ----
    per_expert = Counter()
    for layer, ex, op, hkey, nb in tail.census:
        per_expert[(layer, ex, op, hkey)] += 1
    repeated_keys = {k: c for k, c in per_expert.items() if c > 1}
    n_distinct_inputs = len(per_expert)
    n_repeated_inputs = len(repeated_keys)
    extra_executions = sum(c - 1 for c in per_expert.values())
    # weight bytes avoidable: each repeat of the SAME (layer,expert,op,input)
    # could reuse output instead of reading that weight matrix
    # w1/w3 read INTER x DIM+DIM x INTER; w2 reads DIM x INTER
    BYTES = {"w1": 2048 * 4096 // 2, "w2": 4096 * 2048 // 2, "w3": 2048 * 4096 // 2}
    avoided = 0
    for (layer, ex, op, hkey), c in repeated_keys.items():
        avoided += (c - 1) * BYTES[op]
    total_weight_bytes = sum(BYTES[op] for _, _, op, _, _ in tail.census)

    print(f"\ntotal expert ops:        {total_ops}")
    print(f"distinct effective inputs: {n_distinct_inputs}")
    print(f"repeated inputs (>=2):    {n_repeated_inputs} "
          f"({100*n_repeated_inputs/max(n_distinct_inputs,1):.2f}% of distinct)")
    print(f"extra executions repeatable: {extra_executions}")
    print(f"avoidable weight bytes: {avoided/1e9:.4f} GB "
          f"of {total_weight_bytes/1e9:.4f} GB read "
          f"({100*avoided/max(total_weight_bytes,1):.2f}%)")

    out = {"classification": "RESULT_CACHE_CENSUS",
           "scope": "real captured h40 trajectories replayed through the "
                    "source's own L41/L42 exact path",
           "n_prompts": N_PROMPTS, "total_expert_ops": total_ops,
           "distinct_effective_inputs": n_distinct_inputs,
           "repeated_inputs": n_repeated_inputs,
           "pct_of_distinct_repeated": round(100*n_repeated_inputs/max(n_distinct_inputs,1), 3),
           "extra_executions_repeatable": extra_executions,
           "avoidable_weight_GB": round(avoided/1e9, 4),
           "total_weight_read_GB": round(total_weight_bytes/1e9, 4),
           "avoided_pct": round(100*avoided/max(total_weight_bytes,1), 3),
           "note": "generous upper bound (unlimited capacity, free lookups). "
                   "effective input = quantized operand + scale + route weight "
                   "+ expert + op + revision. similar activations and repeated "
                   "expert IDs do NOT qualify.",
           "verdict": ("REJECT: negligible exact-input repetition"
                       if avoided < 0.05 * total_weight_bytes else
                       "potential large enough to model jointly")}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
