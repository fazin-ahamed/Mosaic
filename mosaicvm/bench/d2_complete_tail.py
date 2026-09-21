"""D2-COMPLETE-TAIL (roadmap S44-S46): complete L41/L42 draft-tail
wall time at K = 1,4,8,16,32, serial vs chunked geometry,
pure-GPU vs hybrid (CPU-VNNI experts), on genuine routes from
captured real h40 trajectories.

Geometry:
  serial  prefill committed prefix (S positions, batched) then K
          speculative positions one at a time (decode semantics).
  chunk   prefill committed prefix then ALL K speculative
          positions as ONE batched pass (prefill semantics) — the
          MoE bincount-grouping aggregates rows per expert, so
          expert calls see M = chunk rows routed to that expert.

Placements:
  gpu     all-experts on GPU (StreamLinear-style packed FP4 torch)
  hybrid  routed experts on CPU via VNNI (E1/E2-certified path)

Timing: wall time of the full replay (prefill + speculation),
p50 over reps, with an expert-call census (calls, total rows,
max M) per geometry. Component decomposition (attn/router/etc.)
is coarse: attention+core time is inferred as wall minus
expert-rail time (hybrid CPU rail timed via its python boundary).

NOT measured here: student time, verifier time, H2 scheduler.
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

OUT = Path("artifacts/d2_complete_tail.json")
PROMPTS = ("convo_13", "math_13", "coding_13", "json_13")
KS = (1, 4, 8, 16, 32)
REPS = 10
# cache caps per placement: GPU free VRAM ~3GB -> ~400 packed FP4
# weights; hybrid prepared weights are 10.49MB each (16.11GB full
# bank, measured) -> 1200 fits host RAM with margin
CACHE_CAP_GPU = 128
CACHE_CAP_CPU = 1200


class CensusLinear(torch.nn.Module):
    """Wraps an expert linear; counts calls/rows; optionally runs
    on CPU via VNNI."""

    def __init__(self, name, km, snap, index, cpu_vnni, census):
        super().__init__()
        self.name, self.km = name, km
        self.snap, self.index, self.cpu = snap, index, cpu_vnni
        self.census = census
        self.cache = OrderedDict()

    def _w(self):
        w = self.cache.get(self.name)
        if w is None:
            from safetensors import safe_open
            shard = self.index[self.name + ".weight"]
            with safe_open(str(self.snap / shard),
                           framework="pt") as f:
                pw = f.get_tensor(self.name + ".weight")
                ps = f.get_tensor(self.name + ".scale")
            if self.cpu:
                w = prepare_weights(pw, ps)
            else:
                w = pw.view(torch.float4_e2m1fn_x2).to("cuda")
                w.scale = ps.to("cuda")
            self.cache[self.name] = w
            cap = CACHE_CAP_CPU if self.cpu else CACHE_CAP_GPU
            if len(self.cache) > cap:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(self.name)
        return w

    def forward(self, x):
        self.census["calls"] += 1
        self.census["rows"] += x.shape[0]
        self.census["maxM"] = max(self.census["maxM"],
                                  x.shape[0])
        if self.cpu:
            t = time.perf_counter()
            cpu = x.detach().to("cpu")
            q, sa = self.km.act_quant(cpu, 128, self.km.scale_fmt,
                                      self.km.scale_dtype)
            k = q.shape[-1]
            out = quantized_gemm(q.reshape(-1, k),
                                 sa.reshape(-1, sa.shape[-1]),
                                 self._w())
            y = out.to(x.dtype).to(x.device).reshape(
                *x.shape[:-1], -1)
            self.census["cpu_s"] += time.perf_counter() - t
            return y
        return self.km.linear(x, self._w(), None)


def main():
    t0 = time.time()
    snap = SNAPSHOT
    # max_seq=64: captures need at most S+32 ~= 47 positions;
    # the 512 default allocates far larger KV buffers
    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=4)
    KM = sys.modules["model"]
    index = tail.index

    # capture states
    caps = []
    for cat in PROMPTS:
        c = torch.load(Path("artifacts/d3_data_105") / f"{cat}.pt",
                       map_location="cpu", weights_only=True)
        caps.append(c)

    originals = {}
    for L, block in zip((41, 42), tail.blocks):
        for e_idx, expert in enumerate(block.ffn.experts):
            if expert is None:
                continue
            for nm in ("w1", "w3", "w2"):
                name = f"layers.{L}.ffn.experts.{e_idx}.{nm}"
                originals[name] = getattr(expert, nm)

    def install(cpu_vnni):
        census = {"calls": 0, "rows": 0, "maxM": 0, "cpu_s": 0.0}
        for L, block in zip((41, 42), tail.blocks):
            for e_idx, expert in enumerate(block.ffn.experts):
                if expert is None:
                    continue
                for nm in ("w1", "w3", "w2"):
                    name = (f"layers.{L}.ffn.experts."
                            f"{e_idx}.{nm}")
                    setattr(expert, nm, CensusLinear(
                        name, KM, snap, index, cpu_vnni, census))
        return census

    def uninstall():
        for L, block in zip((41, 42), tail.blocks):
            for e_idx, expert in enumerate(block.ffn.experts):
                if expert is None:
                    continue
                for nm in ("w1", "w3", "w2"):
                    name = (f"layers.{L}.ffn.experts."
                            f"{e_idx}.{nm}")
                    setattr(expert, nm, originals[name])

    def replay(c, K, chunk):
        S = c["S"]
        exact = torch.cat([c["h40"][:S], c["h40"][S:S + K]])
        if chunk:
            # single batched pass over all S+K positions.
            # inference_mode is MANDATORY: without it autograd
            # retains every expert einsum intermediate (~10GB on
            # a K32 chunk) — this was the OOM.
            reset_decode_state(tail.blocks)
            x = exact.to(device="cuda", dtype=torch.bfloat16
                         ).unsqueeze(0)
            ids = torch.tensor(c["tokens"][:len(exact)],
                               device="cuda").unsqueeze(0)
            with torch.inference_mode(), torch.device("cuda"):
                for block in tail.blocks:
                    x = block(x, 0, ids)
            return x
        return tail.replay(exact, c["tokens"], S)

    results = {"classification": "D2_COMPLETE_TAIL",
               "method": ("captured real h40 trajectories; genuine "
                          "routes; wall p50 over reps; prefill "
                          "committed prefix included in wall"),
               "geometries": {}}
    for placement, cpu_vnni in (("gpu", False),
                                ("hybrid", True)):
        census = install(cpu_vnni)
        # warm-up pass (prepares/caches expert weights)
        replay(caps[0], 32, True)
        for geo, chunk in (("serial", False), ("chunk", True)):
            for K in KS:
                ts = []
                # census: ACCUMULATE per-replay calls/rows/cpu_s
                # across all replays, maxM across reps, then
                # average. (The previous version zeroed per rep and
                # divided the LAST replay by 40 — call/row stats
                # were ~40x too small.)
                acc = {"calls": 0, "rows": 0, "cpu_s": 0.0,
                       "maxM": 0, "n": 0}
                for c in caps:
                    for rep in range(REPS):
                        census["calls"] = census["rows"] = 0
                        census["maxM"] = 0
                        census["cpu_s"] = 0.0
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        replay(c, K, chunk)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t)
                        acc["calls"] += census["calls"]
                        acc["rows"] += census["rows"]
                        acc["cpu_s"] += census["cpu_s"]
                        acc["maxM"] = max(acc["maxM"],
                                          census["maxM"])
                        acc["n"] += 1
                ts.sort()
                p50 = ts[len(ts) // 2] * 1e3
                n = acc["n"]
                key = f"{placement}_{geo}"
                results["geometries"].setdefault(key, {})[str(K)] = {
                    "wall_ms_p50": round(p50, 2),
                    "expert_calls_per_replay": round(
                        acc["calls"] / n, 1),
                    "expert_rows_per_replay": round(
                        acc["rows"] / n, 1),
                    "expert_maxM": acc["maxM"],
                    "cpu_rail_s_per_replay": round(
                        acc["cpu_s"] / n, 4),
                }
                print(f"{key} K{K}: wall {p50:.2f} ms, calls "
                      f"{acc['calls']/n:.0f}, "
                      f"rows {acc['rows']/n:.0f}, "
                      f"maxM {acc['maxM']} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        uninstall()

    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
