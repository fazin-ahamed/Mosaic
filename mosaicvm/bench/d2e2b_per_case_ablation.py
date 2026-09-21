"""D2-E2b: per-case delta ablation (roadmap S45/S46, closure of the
4 E1 ULP-boundary expert cases).

The aggregate E2 result (all four exact inputs executed while
prompt h42 stayed bitwise) proves only the COMBINED impact washed
out — multiple deltas could cancel. This ablation isolates each
delta:

  For each E1 non-bitwise case (name, M, activation-hash):
    run the GPU reference tail replay of its prompt with ONLY
    that one expert call's output replaced by the CPU VNNI
    output (all other expert calls untouched GPU reference);
    compare final h42 bitwise vs the pure-GPU replay.

  bitwise h42  -> that individual delta provably washes out
  non-bitwise  -> that delta propagates; report max_abs

The substituted output is computed from the replay's actual input
(hash-identical to the E1-captured activation by construction).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from source_tail import SourceTail
from vnni_source import prepare_weights, quantized_gemm

OUT = Path("artifacts/d2e2b_per_case_ablation.json")


def xhash(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


class VNNICompute:
    def __init__(self, km, snap, index):
        self.km, self.snap, self.index = km, snap, index
        self.cache = OrderedDict()

    def __call__(self, name, x):
        w = self.cache.get(name)
        if w is None:
            from safetensors import safe_open
            shard = self.index[name + ".weight"]
            with safe_open(str(self.snap / shard),
                           framework="pt") as f:
                pw = f.get_tensor(name + ".weight")
                ps = f.get_tensor(name + ".scale")
            w = prepare_weights(pw, ps)
            self.cache[name] = w
            if len(self.cache) > 96:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(name)
        cpu = x.detach().to("cpu")
        q, sa = self.km.act_quant(cpu, 128, self.km.scale_fmt,
                                  self.km.scale_dtype)
        k = q.shape[-1]
        out = quantized_gemm(q.reshape(-1, k),
                             sa.reshape(-1, sa.shape[-1]), w)
        return (out.to(x.dtype).to(x.device)
                .reshape(*x.shape[:-1], -1))


class SubstitutingLinear(torch.nn.Module):
    """Pass-through GPU reference except at the target call, where
    the VNNI output is substituted (one-at-a-time ablation)."""

    def __init__(self, inner, name, target_key, vnni, counter):
        super().__init__()
        self.inner, self.name = inner, name
        self.target_key, self.vnni = target_key, vnni
        self.counter = counter

    def forward(self, x):
        y = self.inner(x)
        key = (self.name, x.shape[0], xhash(x.detach().to("cpu")))
        if key == self.target_key:
            self.counter["n"] += 1
            return self.vnni(self.name, x)
        return y


def main():
    t0 = time.time()
    snap = SourceTail.SNAPSHOT if hasattr(SourceTail, "SNAPSHOT") \
        else None
    from source_tail import SNAPSHOT
    snap = SNAPSHOT
    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=512, cache_cap=96)
    print(f"source tail loaded ({time.time()-t0:.0f}s)", flush=True)

    e1 = json.loads(Path("artifacts/d2e1_gpu_cpu_expert.json")
                    .read_text())
    cases = [c for c in e1["cases"]
             if not c["gpu_vs_vnni"]["bitwise"]]
    print(f"{len(cases)} non-bitwise E1 cases", flush=True)

    # originals for restore
    originals = {}
    for L, block in zip((41, 42), tail.blocks):
        for e_idx, expert in enumerate(block.ffn.experts):
            if expert is None:
                continue
            for nm in ("w1", "w3", "w2"):
                name = f"layers.{L}.ffn.experts.{e_idx}.{nm}"
                originals[name] = getattr(expert, nm)

    vnni = VNNICompute(tail.km, snap, tail.index)

    def load_capture(cat):
        return torch.load(Path("artifacts/d3_data_105")
                          / f"{cat}.pt", map_location="cpu",
                          weights_only=True)

    def replay(c):
        S = c["S"]
        K = min(32, c["h40"].shape[0] - S)
        exact = torch.cat([c["h40"][:S], c["h40"][S:S + K]])
        return tail.replay(exact, c["tokens"], S), S, K

    # reference replays per prompt (pure GPU)
    ref_h42 = {}
    prompts_needed = sorted({c["prompt"] for c in cases})
    for cat in prompts_needed:
        c = load_capture(cat)
        h42, S, K = replay(c)
        cap = c["h42"][:S + K]
        assert torch.equal(h42.view(torch.int16),
                           cap.view(torch.int16)), \
            f"{cat}: reference replay not bit-exact vs capture"
        ref_h42[cat] = (h42, c, S, K)
        print(f"{cat}: GPU reference replay bit-exact "
              f"({time.time()-t0:.0f}s)", flush=True)

    # one-at-a-time substitution
    results = []
    for case in cases:
        name = (f"layers.{case['layer']}.ffn.experts."
                f"{case['expert']}.{case['matrix']}")
        target_key = (name, case["M"], case["x_sha256_16"])
        counter = {"n": 0}
        # wrap every expert linear; only the target substitutes
        for L, block in zip((41, 42), tail.blocks):
            for e_idx, expert in enumerate(block.ffn.experts):
                if expert is None:
                    continue
                for nm in ("w1", "w3", "w2"):
                    nm_full = (f"layers.{L}.ffn.experts."
                               f"{e_idx}.{nm}")
                    setattr(expert, nm, SubstitutingLinear(
                        originals[nm_full], nm_full, target_key,
                        vnni, counter))
        cat = case["prompt"]
        h42_ref, c, S, K = ref_h42[cat]
        h42_sub, _, _ = replay(c)
        # restore originals
        for L, block in zip((41, 42), tail.blocks):
            for e_idx, expert in enumerate(block.ffn.experts):
                if expert is None:
                    continue
                for nm in ("w1", "w3", "w2"):
                    nm_full = (f"layers.{L}.ffn.experts."
                               f"{e_idx}.{nm}")
                    setattr(expert, nm, originals[nm_full])
        if torch.equal(h42_sub.view(torch.int16),
                       h42_ref.view(torch.int16)):
            cmpres = {"bitwise": True, "max_abs": 0.0}
        else:
            d = (h42_sub.float() - h42_ref.float()).abs()
            frac = (h42_sub.view(torch.int16)
                    == h42_ref.view(torch.int16)).float().mean()
            cmpres = {"bitwise": False,
                      "max_abs": float(d.max().item()),
                      "frac_equal_bits": float(frac.item())}
        results.append({
            "case": (f"L{case['layer']}.{case['matrix']}"
                     f".e{case['expert']}.M{case['M']}"
                     f".{case['x_sha256_16']}@{case['prompt']}"
                     f"/{case['phase']}"),
            "delta_from_e1": case["gpu_vs_vnni"]["max_abs"],
            "substitutions_triggered": counter["n"],
            "h42_sub_vs_ref": cmpres,
        })
        print(f"{results[-1]['case']}: triggered "
              f"{counter['n']}x -> {cmpres} "
              f"({time.time()-t0:.0f}s)", flush=True)

    n_wash = sum(1 for r in results
                 if r["h42_sub_vs_ref"]["bitwise"])
    result = {
        "classification": "D2E2B_PER_CASE_DELTA_ABLATION",
        "method": "GPU reference tail with exactly ONE expert call "
                  "substituted to its CPU VNNI output per run; "
                  "final h42 compared bitwise",
        "n_cases": len(results),
        "per_case_washout": f"{n_wash}/{len(results)}",
        "results": results,
        "verdict": ("ALL FOUR individual deltas wash out "
                    "(per-case bitwise)" if n_wash == len(results)
                    else "At least one delta propagates — see "
                         "results; individual causality open"),
        "production_timing": None,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "results"}, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
