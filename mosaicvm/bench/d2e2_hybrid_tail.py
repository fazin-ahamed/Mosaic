"""D2-E2: hybrid-tail parity (roadmap S43-S46).

Production topology under test:
    GPU source attention/core/router
    + CPU VNNI routed experts
    + GPU source continuation/shared/head

Method: for validation captures from d3_data_105, replay the
captured exact h40 trajectory through:
  (a) the pure-GPU source tail  (SourceTail cuda reference)
  (b) the hybrid tail           (same blocks, routed experts
      executed on CPU via the E1-validated VNNI adapter)
and compare, per stage:
  - h42 bitwise fraction + max_abs (hybrid vs GPU ref, and hybrid
    vs the CAPTURED source h42)
  - route equality (L41/L42 gate outputs)
  - head greedy agreement
  - first divergence position

Outcome classes (roadmap S45):
  - hybrid == GPU ref (bitwise/ULP)  -> promote hybrid tail; the
    old all-CPU 4.0 tail drift is not expert-caused.
  - divergence matches the 4 E1 ULP expert cases -> localize
    adapter/kernel boundary handling.
  - divergence elsewhere -> experts exonerated for the tail drift.
"""
from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from source_tail import SourceTail, SNAPSHOT
from d1b_tail_robustness import reset_decode_state
from vnni_source import prepare_weights, quantized_gemm

OUT = Path("artifacts/d2e2_hybrid_tail.json")
PROMPTS = ("convo_13", "math_13", "coding_13", "json_13")
CACHE_CAP = 96
EXEC_CALLS = []   # (name, M) of every hybrid expert execution


class HybridLinear(torch.nn.Module):
    """Routed-expert linear executed on CPU via the VNNI adapter;
    everything else in the block stays on its device."""

    def __init__(self, name, km, snap, index):
        super().__init__()
        self.name, self.km = name, km
        self.snap, self.index = snap, index
        self.cache = OrderedDict()

    def _prepared(self):
        w = self.cache.get(self.name)
        if w is None:
            from safetensors import safe_open
            shard = self.index[self.name + ".weight"]
            with safe_open(str(self.snap / shard), framework="pt") as f:
                pw = f.get_tensor(self.name + ".weight")
                ps = f.get_tensor(self.name + ".scale")
            w = prepare_weights(pw, ps)
            self.cache[self.name] = w
            if len(self.cache) > CACHE_CAP:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(self.name)
        return w

    def forward(self, x):
        global EXEC_CALLS
        cpu = x.detach().to("cpu")
        q, sa = self.km.act_quant(cpu, 128, self.km.scale_fmt,
                                  self.km.scale_dtype)
        k = q.shape[-1]
        out = quantized_gemm(q.reshape(-1, k),
                             sa.reshape(-1, sa.shape[-1]),
                             self._prepared())
        import hashlib
        EXEC_CALLS.append((self.name, cpu.shape[0],
                           hashlib.sha256(
                               cpu.contiguous()
                               .view(torch.uint8).numpy().tobytes()
                           ).hexdigest()[:16]))
        return (out.to(x.dtype).to(x.device)
                .reshape(*x.shape[:-1], -1))


def main():
    t0 = time.time()
    snap = SNAPSHOT
    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=512, cache_cap=96)
    print(f"source tail loaded ({time.time()-t0:.0f}s)", flush=True)

    # ---- reference (pure GPU) replays first ----
    caps = []
    for cat in PROMPTS:
        c = torch.load(Path("artifacts/d3_data_105") / f"{cat}.pt",
                       map_location="cpu", weights_only=True)
        caps.append(c)

    refs = {}
    for c in caps:
        S = c["S"]
        K = min(32, c["h40"].shape[0] - S)
        exact = torch.cat([c["h40"][:S], c["h40"][S:S + K]])
        h42_ref = tail.replay(exact, c["tokens"], S)
        routes = {L: [r.clone() for r in rows]
                  for L, rows in tail.route_rows.items()}
        refs[c["category"]] = (h42_ref, routes, S, K)
        print(f"{c['category']}: GPU ref replay done "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- swap routed experts to CPU-VNNI execution ----
    hybs = []
    for L, block in zip((41, 42), tail.blocks):
        for e_idx, expert in enumerate(block.ffn.experts):
            if expert is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(expert, nm, HybridLinear(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}",
                    tail.km, snap, tail.index))
                hybs.append(getattr(expert, nm))
    print(f"swapped {len(hybs)} routed expert linears to CPU-VNNI",
          flush=True)

    rows = []
    for c in caps:
        cat = c["category"]
        h42_ref, routes_ref, S, K = refs[cat]
        exact = torch.cat([c["h40"][:S], c["h40"][S:S + K]])
        h42_hyb = tail.replay(exact, c["tokens"], S)
        routes_hyb = tail.route_rows

        cap_h42 = c["h42"][:S + K]

        def cmp(a, b):
            if torch.equal(a, b):
                return {"bitwise": True, "max_abs": 0.0}
            d = (a.float() - b.float()).abs()
            return {"bitwise": False,
                    "max_abs": float(d.max().item()),
                    "frac_equal": float(
                        (a.view(torch.int16)
                         == b.view(torch.int16)).float().mean().item())}

        # first divergence position (hybrid vs ref)
        diff = (h42_hyb.view(torch.int16)
                != h42_ref.view(torch.int16)).any(dim=-1)
        first_div = int(diff.nonzero()[0, 0]) if diff.any() else None

        route_eq = {}
        for L in (41, 42):
            r_ref = torch.cat(routes_ref[L])
            r_hyb = torch.cat(routes_hyb[L])
            route_eq[str(L)] = bool(torch.equal(r_ref, r_hyb))

        # head greedy agreement
        lg_ref = tail.logits(h42_ref)
        lg_hyb = tail.logits(h42_hyb)
        greedy_eq = bool(torch.equal(lg_ref.argmax(-1),
                                     lg_hyb.argmax(-1)))

        rows.append({
            "category": cat, "S": S, "K": K,
            "hyb_vs_gpuref": cmp(h42_hyb, h42_ref),
            "hyb_vs_capture": cmp(h42_hyb, cap_h42),
            "gpuref_vs_capture": cmp(h42_ref, cap_h42),
            "first_divergence_pos": first_div,
            "routes_equal": route_eq,
            "head_greedy_equal": greedy_eq,
        })
        print(f"{cat}: hyb-vs-ref "
              f"{rows[-1]['hyb_vs_gpuref']} first_div "
              f"{first_div} greedy {greedy_eq} "
              f"({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    ref_bit = sum(r["hyb_vs_gpuref"]["bitwise"] for r in rows)
    cap_bit = sum(r["hyb_vs_capture"]["bitwise"] for r in rows)
    ref_max = max(r["hyb_vs_gpuref"]["max_abs"] for r in rows)
    cap_max = max(r["hyb_vs_capture"]["max_abs"] for r in rows)
    # E1-closure: EXACT input identity (name + M + activation
    # hash) of the non-bitwise E1 expert cases within this replay.
    # A match proves the identical computation executed in the
    # hybrid; combined with the bitwise h42 result, the ULP
    # difference is traced-closed. Unmatched cases stay OPEN.
    e1_path = Path("artifacts/d2e1_gpu_cpu_expert.json")
    e1_cov = None
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        nb = [c for c in e1["cases"]
              if not c.get("gpu_vs_vnni", {}).get("bitwise", True)]
        executed = set(EXEC_CALLS)   # (name, M, xhash)
        matched, unmatched = [], []
        for c in nb:
            key = (f"layers.{c['layer']}.ffn.experts."
                   f"{c['expert']}.{c['matrix']}", c["M"],
                   c["x_sha256_16"])
            tag = (f"L{c['layer']}.{c['matrix']}.e{c['expert']}"
                   f".M{c['M']}.{c['x_sha256_16']}"
                   f"@{c.get('prompt', '?')}/{c.get('phase', '?')}")
            (matched if key in executed else unmatched).append(tag)
        e1_cov = {
            "match_basis": "name + M + activation sha256 (exact "
                           "input identity)",
            "n_executed_calls_total": len(EXEC_CALLS),
            "e1_non_bitwise_cases": matched + unmatched,
            "exact_input_matched": matched,
            "unmatched_stay_open": unmatched,
        }
    result = {
        "classification": "D2E2_HYBRID_TAIL_PARITY",
        "e1_case_coverage": e1_cov,
        "topology": "GPU attention/core/router + CPU VNNI routed "
                    "experts + GPU continuation/shared/head",
        "prompts": list(PROMPTS),
        "n_prompts": n,
        "hyb_vs_gpuref_bitwise": f"{ref_bit}/{n}",
        "hyb_vs_gpuref_max_abs": ref_max,
        "hyb_vs_capture_bitwise": f"{cap_bit}/{n}",
        "hyb_vs_capture_max_abs": cap_max,
        "all_routes_equal": all(all(r["routes_equal"].values())
                                for r in rows),
        "all_head_greedy_equal": all(r["head_greedy_equal"]
                                     for r in rows),
        "rows": rows,
        "verdict": ("PASS — hybrid tail reproduces the GPU source "
                    "tail bitwise" if ref_bit == n else
                    "PARTIAL/OPEN — see per-prompt rows; localize "
                    "first divergence before any promotion"),
        "production_timing": None,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "rows"}, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
