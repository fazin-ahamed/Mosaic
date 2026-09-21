"""EK0c — long single-stream route capture + temporal-residency confirmation.

Generates N_GEN (default 512) tokens from ONE prompt through the exact source
model, hooking every MoE.gate to log per-position top-6 routed expert IDs
across all 43 layers. Feeds the single long stream through the corrected EK0b
residency simulation (LRU / static-hot / Belady) to test whether sustained
generation develops materially higher expert locality than the 32-token
prompts did (adjacent reuse was only 0.24-0.44 there).

Kill gate: if steady-state best-oracle @10GB miss/cycle stays >> 103, GPU-only
Exact-30 is FINAL closed. Reuses the exact stage1 model-loading path.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, OrderedDict, Counter
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

BYTES = 13369344
H2D = 12.90e9
CORE_MS = 17.5
BUDGET_MS = 125.2
N_LAYERS = 43
W = 6
STRIDE = 4
CAPS_GB = [2, 4, 6, 8, 10]
N_GEN = int(os.environ.get("EK0C_NGEN", "512"))
PROMPT = ("Explain, step by step and in careful technical detail, how a modern "
          "operating system schedules threads across multiple CPU cores, "
          "including run queues, load balancing, priority handling, and "
          "context-switch cost. Then discuss how this interacts with cache "
          "locality and NUMA effects, and give concrete examples.")


def capture_routes():
    """Run source model, generate N_GEN tokens, log per-position top-6 per layer."""
    # Build the source model (stage1_record loader pattern) + gate hooks.
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
        "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    sys.path.insert(0, str(snap / "inference"))
    from source_corpus_gen import build_kernel_module, StreamLinear
    sys.modules["kernel"] = build_kernel_module()
    import types, math as _math
    _fht = types.ModuleType("fast_hadamard_transform")
    _HAD = {}

    def _had(x):
        d = x.shape[-1]; H = _HAD.get(d)
        if H is None or H.device != x.device:
            h = torch.tensor([[1.0]], device=x.device, dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
            _HAD[d] = H = h / _math.sqrt(d)
        return (x.float() @ H).to(x.dtype)
    _fht.hadamard_transform = lambda x, scale=None: _had(x)
    sys.modules["fast_hadamard_transform"] = _fht
    import model as KM
    import dataclasses
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads((snap / "inference/config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0; args.max_batch_size = 1; args.max_seq_len = 1024
    args.dspark_block_size = 0; args.n_mtp_layers = 0
    _orig = torch.empty

    def _patch(*a, **k):
        if k.get("dtype") == torch.float4_e2m1fn_x2:
            k["device"] = "meta"
        return _orig(*a, **k)
    torch.empty = _patch
    model = KM.Transformer(args)
    torch.empty = _orig
    for L, layer in enumerate(model.layers):
        for e_idx, ex in enumerate(layer.ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(ex, nm, StreamLinear(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}"))
    import gc; gc.collect()
    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight", "head.weight") \
            else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()
    import types as _t

    def _embed_fwd(self, x):
        return torch.nn.functional.embedding(x.cpu(), self.weight).to(x.device)

    def _head_fwd(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return torch.nn.functional.linear(x.cpu().float(), self.weight).to("cuda")
    model.embed.forward = _t.MethodType(_embed_fwd, model.embed)
    model.head.forward = _t.MethodType(_head_fwd, model.head)
    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    StreamLinear._snap = snap; StreamLinear._index = index
    params = dict(model.named_parameters())

    def dequant_wo_a(w, s):
        rows, cols = w.shape
        return (w.float() * s.float().repeat_interleave(rows // s.shape[0], 0)
                .repeat_interleave(cols // s.shape[1], 1)).to(torch.bfloat16)
    for name, shard in index.items():
        if ".ffn.experts." in name and ".shared_experts." not in name:
            continue
        if name not in params:
            continue
        with safe_open(str(snap / shard), framework="pt") as f:
            t = f.get_tensor(name)
            if name.endswith("wo_a.weight"):
                t = dequant_wo_a(t, f.get_tensor(name.replace(".weight", ".scale")))
        with torch.no_grad():
            params[name].data.copy_(t)

    # hook every MoE gate: capture (layer -> top6 ids) per forward position
    cur = {}
    routes = []   # per generated position: {layer: [6 ids]}

    def mk(L):
        def h(mod, inp, out):
            w, idx = out
            # idx: [tokens, 6]. Only record the LAST position's route (the
            # token being generated); prefill (P>1) contributes prompt routes
            # we discard by taking the final row.
            flat = idx.detach().cpu()
            last = flat.reshape(-1, flat.shape[-1])[-1].tolist()[:6]
            cur[L] = last
        return h
    for L, layer in enumerate(model.layers):
        layer.ffn.gate.register_forward_hook(mk(L))

    from transformers import AutoTokenizer
    torch.set_default_device("cuda")
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
    from mosaicvm.bench.d1b_tail_robustness import reset_decode_state
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    reset_decode_state(model)
    t0 = time.time()
    seq = list(ids)
    prev = 0
    print(f"gen start: prompt {len(ids)} tokens, N_GEN {N_GEN}", flush=True)
    with torch.inference_mode():
        for step in range(N_GEN):
            cur.clear()
            inp = torch.tensor(seq[prev:], device="cuda").unsqueeze(0)
            out_ids, _, _ = model(inp, prev)
            prev = len(seq)
            nxt = int(out_ids[0]) if out_ids.ndim == 1 else int(out_ids[0, -1])
            if len(cur) == N_LAYERS:
                routes.append(dict(cur))
            seq.append(nxt)
            if (step + 1) % 64 == 0:
                print(f"  gen {step+1}/{N_GEN} ({time.time()-t0:.0f}s)",
                      flush=True)
    return routes


def build_cycles(routes):
    cycles = []
    for start in range(0, len(routes) - W + 1, STRIDE):
        U = set()
        for pm in routes[start:start + W]:
            for L in range(N_LAYERS):
                for e in pm.get(L, []):
                    U.add((L, e))
        cycles.append(sorted(U))
    return cycles


def main():
    from mosaicvm.bench.ek0b_residency import (
        cycles_to_stream, lru_stream, belady_stream, static_hot_stream,
        per_cycle_miss, summ)
    routes = capture_routes()
    print(f"captured {len(routes)} decode positions")
    # persist raw (position -> {layer: [6 ids]}) for reuse (H0 capacity replay)
    torch.save({"routes": routes, "n_layers": N_LAYERS},
               "artifacts/ek0c_routes.pt")
    print("persisted artifacts/ek0c_routes.pt")
    # adjacent reuse
    adj = []
    for i in range(1, len(routes)):
        a = set((L, e) for L in range(N_LAYERS) for e in routes[i].get(L, []))
        b = set((L, e) for L in range(N_LAYERS) for e in routes[i-1].get(L, []))
        adj.append(len(a & b) / len(a))
    print(f"adjacent-position reuse: mean {sum(adj)/len(adj):.3f}")
    cycles = build_cycles(routes)
    stream, spans = cycles_to_stream(cycles)
    max_cold = int((BUDGET_MS - CORE_MS) / 1e3 * H2D / BYTES)
    print(f"cycles {len(cycles)}, gate <= {max_cold} cold/cycle\n")
    print(f"{'capGB':>5} {'policy':>8} {'meanMiss':>9} {'p90':>5} {'tputUB':>7}")
    out = {"classification": "EK0C_LONGSTREAM", "n_positions": len(routes),
           "adjacent_reuse": round(sum(adj)/len(adj), 3),
           "abs_max_cold": max_cold, "by_cap": {}}
    for cap_gb in CAPS_GB:
        cap = int(cap_gb * 1e9 / BYTES)
        rec = {}
        for pol, fn in (("lru", lru_stream), ("hot", static_hot_stream),
                        ("belady", belady_stream)):
            pcm = per_cycle_miss(fn(stream, cap), spans)
            s = summ(pcm); rec[pol] = s
            print(f"{cap_gb:>5} {pol:>8} {s['mean_miss']:>9} {s['p90_miss']:>5} "
                  f"{s['mean_tput_UB']:>7}")
        out["by_cap"][str(cap_gb)] = rec
    best10 = min(out["by_cap"]["10"]["belady"]["mean_miss"],
                 out["by_cap"]["10"]["hot"]["mean_miss"])
    out["best_oracle_10GB_mean_miss"] = best10
    out["verdict"] = "ORACLE_PASS" if best10 <= max_cold else "GPU_ONLY_FINAL_CLOSED"
    print(f"\nbest-oracle @10GB {best10} vs gate {max_cold} -> {out['verdict']}")
    Path("artifacts/ek0c_longstream.json").write_text(json.dumps(out, indent=1))
    print("wrote artifacts/ek0c_longstream.json")


if __name__ == "__main__":
    main()
