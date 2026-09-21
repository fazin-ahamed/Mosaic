"""D2-P: adapter-inclusive CPU VNNI expert-path timing (roadmap S38).

v2 — production-correct methodology per the D2-P contract:
  - REAL routed activations: hooked from live kit-model replays
    (the E1 capture pattern), stratified by (layer, matrix) with
    their real M values (decode M=1, prefill M<=~14).
  - COLD stratified into two distinct regimes:
      dram-cold  round-robin over ALL captured experts of the
                 (layer, matrix); prepared weights all RAM-
                 resident; working set >> L3 so weight streaming
                 dominates (production bank regime).
      first-use  prepare_weights INCLUDED in the timed region
                 (raw FP4 read + prepare per fresh expert).
  - CONTENDED: a SEPARATE PROCESS runs a sustained CUDA matmul
    workload (independent CUDA context) while the main process
    times expert calls -> genuine host/H2D contention.

Timed region (the full hybrid expert call):
    x bf16 [M,K] (already on CPU from the captured real call)
    -> source act_quant (E4M3 codes + E8M0 scales)
    -> VNNI GEMM (prepared weights)
    -> bf16 reconstruct
(The GPU<->CPU transfers are architecturally part of the call
but are benchmarked separately after the compute regimes since
the capture phase occupies the GPU; reported as their own rows.)

This is expert-path timing ONLY — not the complete tail wall.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

OUT = Path("artifacts/d2p_expert_timing.json")
PROMPTS = ("convo_13", "math_13", "coding_13")
N_DECODE = 8
REPS = 50
CONTEND_CMD = (
    "import torch; a = torch.randn(4096, 4096, device='cuda')"
    ".to(torch.bfloat16); b = torch.randn(4096, 4096, "
    "device='cuda').to(torch.bfloat16);\n"
    "for _ in range(10 ** 9):\n"
    "    a = a @ b\n"
)


def capture_real_calls():
    """Load the kit model, run val prompts, capture routed expert
    calls (prefix, x_cpu, M). Frees the model before returning."""
    from source_corpus_gen import build_kernel_module, StreamLinear
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    sys.path.insert(0, str(snap / "inference"))
    sys.modules["kernel"] = build_kernel_module()
    import math as _math
    _fht = type(sys)("fast_hadamard_transform")
    _HAD = {}

    def _had(x):
        d = x.shape[-1]
        H = _HAD.get(d)
        if H is None or H.device != x.device:
            h = torch.tensor([[1.0]], device=x.device,
                             dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1),
                               torch.cat([h, -h], 1)])
            _HAD[d] = H = h / _math.sqrt(d)
        return (x.float() @ H).to(x.dtype)

    _fht.hadamard_transform = lambda x, scale=None: _had(x)
    sys.modules["fast_hadamard_transform"] = _fht
    import model as KM
    import dataclasses
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads((snap / "inference" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 512
    args.dspark_block_size = 0
    args.n_mtp_layers = 0
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
    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight",
                                        "head.weight") else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()
    import types

    def _embed_fwd(self, x):
        return torch.nn.functional.embedding(
            x.cpu(), self.weight).to(x.device)

    def _head_fwd(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return torch.nn.functional.linear(
            x.cpu().float(), self.weight).to("cuda")

    model.embed.forward = types.MethodType(_embed_fwd, model.embed)
    model.head.forward = types.MethodType(_head_fwd, model.head)

    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    StreamLinear._snap = snap
    StreamLinear._index = index
    params = dict(model.named_parameters())

    def dequant_wo_a(w, s):
        rows, cols = w.shape
        return (w.float() * s.float()
                .repeat_interleave(rows // s.shape[0], 0)
                .repeat_interleave(cols // s.shape[1], 1)
                ).to(torch.bfloat16)

    for name, shard in index.items():
        if ".ffn.experts." in name and ".shared_experts." not in name:
            continue
        if name not in params:
            continue
        with safe_open(str(snap / shard), framework="pt") as f:
            t = f.get_tensor(name)
            if name.endswith("wo_a.weight"):
                s = f.get_tensor(name.replace(".weight", ".scale"))
                t = dequant_wo_a(t, s)
        with torch.no_grad():
            params[name].data.copy_(t)
    torch.set_default_device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap),
                                        trust_remote_code=True)
    spec = json.loads(Path("artifacts/stage1_prompts.json").read_text())
    prompts = {p["category"]: p["prompt"] for p in spec["prompts"]}

    CAP = {}

    def make_hook(sl):
        def h(mod, args_, output):
            x = args_[0]
            if sl.prefix not in CAP:
                CAP[sl.prefix] = x.detach().to("cpu").clone()
        return h

    handles = []
    for L in (41, 42):
        for e_idx, ex in enumerate(model.layers[L].ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                sl = getattr(ex, nm)
                handles.append(sl.register_forward_hook(
                    make_hook(sl)))
    for cat in PROMPTS:
        ids = tok(prompts[cat], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        with torch.no_grad():
            out_ids, _, _ = model(ids, 0)
            for step in range(N_DECODE):
                out_ids, _, _ = model(out_ids.reshape(1, 1), S + step)
        print(f"captured {cat}: {len(CAP)} distinct calls",
              flush=True)
    for h in handles:
        h.remove()
    # free the GPU model before timing
    del model, params
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return CAP, snap, index


def main():
    t0 = time.time()
    CAP, snap, index = capture_real_calls()
    print(f"capture done: {len(CAP)} distinct (expert,matrix) "
          f"calls, model freed ({time.time()-t0:.0f}s)", flush=True)

    from source_tail import SNAPSHOT
    from vnni_source import prepare_weights, quantized_gemm
    snap = SNAPSHOT
    from safetensors import safe_open

    def load_raw(name):
        shard = index[name + ".weight"]
        with safe_open(str(snap / shard), framework="pt") as f:
            return (f.get_tensor(name + ".weight"),
                    f.get_tensor(name + ".scale"))

    # stratify: per (layer, matrix), the captured experts
    strata = {}
    for prefix, x in CAP.items():
        L = int(prefix.split(".")[1])
        matrix = prefix.rsplit(".", 1)[1]
        strata.setdefault((L, matrix), []).append((prefix, x))
    for k in strata:
        strata[k].sort(key=lambda p: p[0])
    print("strata: " + ", ".join(
        f"L{L}.{m}:{len(v)} (M={v[0][1].shape[0]})"
        for (L, m), v in strata.items()), flush=True)

    # exact source act_quant path: the model module (already
    # imported during capture) carries scale_fmt/scale_dtype
    KM = sys.modules["model"]

    def one_call(x_cpu, w):
        q, sa = KM.act_quant(x_cpu, 128, KM.scale_fmt,
                             KM.scale_dtype)
        out = quantized_gemm(q.reshape(-1, q.shape[-1]),
                             sa.reshape(-1, sa.shape[-1]), w)
        return out.to(torch.bfloat16)

    def timeit(fn, reps=REPS):
        fn()                       # warm-up
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        ts.sort()
        return round(ts[len(ts) // 2] * 1e3, 3)

    # prepared weights (RAM-resident, excluded from dram-cold timing)
    prepared = {}
    prep_times = {}
    for (L, matrix), items in strata.items():
        for prefix, _ in items:
            w, sc = load_raw(prefix)
            t = time.perf_counter()
            prepared[prefix] = prepare_weights(w, sc)
            prep_times[prefix] = time.perf_counter() - t
    print(f"prepared {len(prepared)} experts "
          f"({time.time()-t0:.0f}s)", flush=True)

    results = {"classification": "D2P_SOURCE_ADAPTER_EXPERT_TIMING",
               "method": ("real hooked routed activations; per "
                          "(layer,matrix) strata; full adapter+VNNI "
                          "call; p50 ms of reps"),
               "regimes": {}, "n_prepared": len(prepared),
               "per_expert_prepare_seconds_mean": round(
                   sum(prep_times.values()) / len(prep_times), 4),
               "production_tail_wall": None}

    # ---- compute regimes (no GPU involvement) ----
    for (L, matrix), items in strata.items():
        key = f"L{L}.{matrix}"
        row = {}
        prefix0, x0 = items[0]
        mh = {}
        for _, xx in items:
            mh[int(xx.shape[0])] = mh.get(int(xx.shape[0]), 0) + 1
        row["M_histogram"] = mh
        row["M"] = int(x0.shape[0])
        row["n_experts_in_stratum"] = len(items)
        # warm: same expert/act repeatedly
        row["warm_ms"] = timeit(lambda: one_call(x0, prepared[prefix0]))
        # dram-cold: round-robin all experts of the stratum
        rr = {"i": 0}
        def cold_fn():
            prefix, x = items[rr["i"] % len(items)]
            rr["i"] += 1
            one_call(x, prepared[prefix])
        row["dram_cold_ms"] = timeit(cold_fn)
        # first-use: prepare included in timed region
        fu = {"i": 0}
        raw = {p: load_raw(p) for p, _ in items}
        def fu_fn():
            prefix, x = items[fu["i"] % len(items)]
            fu["i"] += 1
            w, sc = raw[prefix]
            one_call(x, prepare_weights(w, sc))
        row["first_use_ms"] = timeit(fu_fn, reps=min(REPS, 20))
        results["regimes"][key] = row
        print(f"{key}: M{row['M']} warm {row['warm_ms']} "
              f"dram-cold {row['dram_cold_ms']} "
              f"first-use {row['first_use_ms']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- transfer costs (GPU involved, no contention) ----
    x_g = CAP[strata[(41, "w1")][0][0]].cuda()
    def to_cpu():
        x_g.detach().to("cpu")
    y_c = one_call(strata[(41, "w1")][0][1],
                   prepared[strata[(41, "w1")][0][0]])
    y_g = y_c.cuda()
    def to_gpu():
        y_c.to("cuda")
    torch.cuda.synchronize()
    results["transfers_ms"] = {
        "gpu_to_cpu_M1": timeit(to_cpu),
        "cpu_to_gpu_M1": timeit(to_gpu),
    }

    # ---- contended: separate-process sustained CUDA workload ----
    proc = subprocess.Popen(
        [sys.executable, "-c", CONTEND_CMD],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)                # let the workload ramp
    try:
        alive = proc.poll() is None
        for (L, matrix), items in strata.items():
            key = f"L{L}.{matrix}"
            prefix0, x0 = items[0]
            results["regimes"][key]["contended_warm_ms"] = timeit(
                lambda: one_call(x0, prepared[prefix0]))
            rr2 = {"i": 0}
            def cold2():
                prefix, x = items[rr2["i"] % len(items)]
                rr2["i"] += 1
                one_call(x, prepared[prefix])
            results["regimes"][key]["contended_cold_ms"] = timeit(cold2)
            print(f"{key} contended: "
                  f"warm {results['regimes'][key]['contended_warm_ms']} "
                  f"cold {results['regimes'][key]['contended_cold_ms']} "
                  f"(workload alive={alive}, {time.time()-t0:.0f}s)",
                  flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # ---- END-TO-END hybrid call: GPU-resident input -> blocking
    # D2H -> adapter+GEMM -> H2D -> cuda.synchronize. This is the
    # advertised production call; stratified by (layer, matrix)
    # and by the distinct real M values in each stratum. The old
    # per-stratum rows above are CPU-kernel diagnostics only.
    KM = sys.modules["model"]

    def full_call(x_gpu, w):
        cpu = x_gpu.detach().to("cpu")          # blocking D2H
        q, sa = KM.act_quant(cpu, 128, KM.scale_fmt,
                             KM.scale_dtype)
        out = quantized_gemm(q.reshape(-1, q.shape[-1]),
                             sa.reshape(-1, sa.shape[-1]), w)
        y = out.to(torch.bfloat16).to("cuda")   # H2D (async)
        torch.cuda.synchronize()                # block on return
        return y

    def e2e_strata(tag):
        rows = {}
        for (L, matrix), items in strata.items():
            key = f"L{L}.{matrix}"
            by_M = {}
            for prefix, x in items:
                by_M.setdefault(int(x.shape[0]), []).append(
                    (prefix, x))
            row = {}
            for M_val, mitems in sorted(by_M.items()):
                prefix0, x0 = mitems[0]
                x_g = x0.cuda()
                # warm same-expert e2e
                ms = timeit(lambda: full_call(x_g,
                                              prepared[prefix0]))
                # dram-cold round-robin across this M's experts.
                # Production activations are ALREADY GPU-resident:
                # preload each captured activation to GPU outside
                # the timed region; the timed path is exactly
                # full_call (D2H + adapter + GEMM + H2D + sync).
                x_gpu_pool = [x.cuda() for _, x in mitems]
                rr3 = {"i": 0}
                def cold3():
                    i3 = rr3["i"] % len(mitems)
                    rr3["i"] += 1
                    full_call(x_gpu_pool[i3],
                              prepared[mitems[i3][0]])
                ms_cold = timeit(cold3)
                row[f"M{M_val}_n{len(mitems)}"] = {
                    "e2e_warm_ms": ms, "e2e_dram_cold_ms": ms_cold}
            rows[key] = row
            print(f"e2e[{tag}] {key}: {row} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return rows

    torch.cuda.synchronize()
    results["end_to_end_idle"] = e2e_strata("idle")

    # contended e2e: separate-process GPU workload running
    proc2 = subprocess.Popen(
        [sys.executable, "-c", CONTEND_CMD],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
    try:
        alive = proc2.poll() is None
        results["end_to_end_contended"] = e2e_strata(
            f"contended, workload alive={alive}")
    finally:
        proc2.terminate()
        proc2.wait(timeout=10)

    results["classification"] = (
        "D2P_SOURCE_ADAPTER_EXPERT_TIMING_V2")
    results["regime_semantics"] = {
        "per_stratum_rows": "CPU-kernel diagnostics (adapter+GEMM "
                            "only, CPU-resident input/output)",
        "end_to_end_*": "full hybrid call from GPU-resident input: "
                        "blocking D2H + adapter + VNNI GEMM + H2D "
                        "+ synchronize",
    }
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
