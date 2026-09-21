"""D2-E1: CPU VNNI expert vs source GPU expert parity on REAL
routed activations (roadmap S46-S49).

Method:
  1. Load the full kit model on GPU (source-class path, identical
     to the recorder) and run live forwards on validation prompts.
  2. Forward hooks on L41/L42 routed-expert StreamLinear modules
     capture (prefix, x [M,K] bf16, y_gpu [M,N] bf16) per call.
  3. On CPU, for each captured case:
       a. act_quant code parity: KM.act_quant(x) on GPU vs CPU —
          bitwise E4M3 codes + E8M0 scales.
       b. CPU torch reference: KM.linear(x, w_fp4) (fp4_gemm).
       c. CPU VNNI: quantized_gemm(q, sa, prepare_weights(w, s)).
  4. Compare y_gpu vs y_vnni (the D2-E1 target), plus y_gpu vs
     y_ref and y_vnni vs y_ref for localization.

Coverage: natural routing traffic (prefill M up to ~S, decode M=1),
all three matrices (w1/w3 see raw routed rows; w2 sees
weight-multiplied activations — signed/fractional coverage).

Classification per case: 'bitwise' if y_gpu == y_vnni exactly;
else max_abs / rel reported. Aggregate PASS only if codes are
bitwise AND outputs are bitwise (no tolerance promotion).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from source_corpus_gen import build_kernel_module, StreamLinear

OUT = Path("artifacts/d2e1_gpu_cpu_expert.json")
PROMPTS = ("convo_13", "math_13", "coding_13")   # val split
N_DECODE = 8
MAX_CASES_PER_MATRIX = 6   # distinct (layer,expert) per matrix
LAYERS = (41, 42)


def main():
    t0 = time.time()
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
    import gc
    gc.collect()
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
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)
    torch.set_default_device("cuda")

    # ---- capture hooks on L41/L42 routed experts ----
    CAP = []
    CURRENT = {"prompt": None, "phase": None}

    def make_hook(prefix):
        def h(mod, args_, output):
            x = args_[0]
            CAP.append((prefix, x.detach().to("cpu"),
                        output.detach().to("cpu"),
                        CURRENT["prompt"], CURRENT["phase"]))
        return h

    handles = []
    for L in LAYERS:
        for e_idx, ex in enumerate(model.layers[L].ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                sl = getattr(ex, nm)
                handles.append(sl.register_forward_hook(
                    make_hook(sl.prefix)))
    print(f"hooks: {len(handles)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap),
                                        trust_remote_code=True)
    spec = json.loads(Path("artifacts/stage1_prompts.json").read_text())
    prompts = {p["category"]: p["prompt"] for p in spec["prompts"]}

    for cat in PROMPTS:
        ids = tok(prompts[cat], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        with torch.no_grad():
            CURRENT["prompt"], CURRENT["phase"] = cat, "prefill"
            out_ids, _, _ = model(ids, 0)
            for step in range(N_DECODE):
                CURRENT["phase"] = f"decode{step}"
                cur = out_ids.reshape(1, 1)
                out_ids, _, _ = model(cur, S + step)
        CURRENT["prompt"] = CURRENT["phase"] = None
        print(f"{cat}: {len(CAP)} cumulative expert calls "
              f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()

    # ---- select cases: distinct (layer, expert, matrix) ----
    seen = {}
    for prefix, x, y, cat, phase in CAP:
        key = prefix
        if key not in seen:
            seen[key] = (prefix, x, y, cat, phase)
    print(f"distinct (layer,expert,matrix) cases: {len(seen)}",
          flush=True)

    # ---- CPU-side evaluation ----
    from mosaicvm.bench.vnni_source import (prepare_weights,
                                            quantized_gemm)

    def load_weight(prefix):
        shard = index[prefix + ".weight"]
        with safe_open(str(snap / shard), framework="pt") as f:
            w = f.get_tensor(prefix + ".weight")
            s = f.get_tensor(prefix + ".scale")
        return w.view(torch.float4_e2m1fn_x2), s

    import hashlib

    def xhash(t):
        return hashlib.sha256(
            t.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()[:16]

    rows = []
    saved_cases = {}
    for prefix, x, y_gpu, cat, phase in seen.values():
        L = int(prefix.split(".")[1])
        matrix = prefix.rsplit(".", 1)[1]
        M, K = x.shape
        w_fp4, w_scale = load_weight(prefix)

        # (a) act_quant code parity GPU vs CPU
        q_gpu, s_gpu = KM.act_quant(x.cuda(), 128, KM.scale_fmt,
                                    KM.scale_dtype)
        q_cpu, s_cpu = KM.act_quant(x, 128, KM.scale_fmt,
                                    KM.scale_dtype)
        q_eq = torch.equal(q_gpu.view(torch.uint8).cpu(),
                           q_cpu.view(torch.uint8))
        s_eq = torch.equal(s_gpu.cpu().view(torch.uint8),
                           s_cpu.view(torch.uint8))

        # (b) CPU torch reference (fp4_gemm)
        w_ref = w_fp4.cpu()
        w_ref.scale = w_scale.cpu()
        with torch.no_grad():
            y_ref = KM.linear(x, w_ref, None)

        # (c) CPU VNNI
        prep = prepare_weights(w_fp4, w_scale)
        y_vnni = quantized_gemm(q_cpu.reshape(-1, K),
                                s_cpu.reshape(-1, s_cpu.shape[-1]),
                                prep).reshape(*x.shape[:-1], -1)

        def cmp(a, b):
            if torch.equal(a, b):
                return {"bitwise": True, "max_abs": 0.0,
                        "rel": 0.0}
            d = (a.float() - b.float()).abs()
            denom = b.float().abs().clamp_min(1e-6)
            # unrounded max_abs: rounding to 6 decimals previously
            # produced misleading max_abs=0.0 on non-bitwise cases
            return {"bitwise": False,
                    "max_abs": float(d.max().item()),
                    "rel": float((d / denom).max().item())}

        rows.append({
            "layer": L, "matrix": matrix, "M": int(M),
            "expert": int(prefix.split(".")[4]),
            "prompt": cat, "phase": phase, "x_sha256_16": xhash(x),
            "codes_bitwise": bool(q_eq and s_eq),
            "gpu_vs_vnni": cmp(y_gpu, y_vnni.to(y_gpu.dtype)),
            "gpu_vs_cpuref": cmp(y_gpu, y_ref),
            "vnni_vs_cpuref": cmp(y_vnni.to(y_ref.dtype), y_ref),
            "y_absmax": round(y_gpu.float().abs().max().item(), 3),
        })

    # persist exact activations of non-bitwise cases for the
    # E2 substitution/identity closure
    for (prefix, x, y_gpu, cat, phase), r in zip(seen.values(),
                                                 rows):
        if not r["gpu_vs_vnni"]["bitwise"]:
            saved_cases[f"L{r['layer']}.{r['matrix']}.e{r['expert']}"
                        f".{r['x_sha256_16']}"] = {
                "x": x, "y_gpu": y_gpu, "prompt": cat,
                "phase": phase}
    if saved_cases:
        torch.save(saved_cases,
                   "artifacts/d2e1_nonbitwise_activations.pt")

    n = len(rows)
    codes_ok = sum(r["codes_bitwise"] for r in rows)
    vnni_bit = sum(r["gpu_vs_vnni"]["bitwise"] for r in rows)
    ref_bit = sum(r["gpu_vs_cpuref"]["bitwise"] for r in rows)
    vnni_max = max(r["gpu_vs_vnni"]["max_abs"] for r in rows)
    ref_max = max(r["gpu_vs_cpuref"]["max_abs"] for r in rows)
    Ms = sorted({r["M"] for r in rows})
    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    summary = {L: {
        "cases": len(v),
        "codes_bitwise": sum(x["codes_bitwise"] for x in v),
        "gpu_vs_vnni_bitwise": sum(x["gpu_vs_vnni"]["bitwise"]
                                   for x in v),
        "gpu_vs_vnni_max_abs": max(x["gpu_vs_vnni"]["max_abs"]
                                   for x in v),
    } for L, v in per_layer.items()}

    result = {
        "classification": "D2E1_GPU_SOURCE_EXPERT_VS_CPU_VNNI",
        "method": "live GPU kit-model routed expert calls vs CPU "
                  "VNNI adapter on identical captured activations",
        "prompts": list(PROMPTS), "decode_steps": N_DECODE,
        "n_cases": n, "M_values": Ms,
        "codes_bitwise": f"{codes_ok}/{n}",
        "gpu_vs_vnni_bitwise": f"{vnni_bit}/{n}",
        "gpu_vs_vnni_max_abs": vnni_max,
        "gpu_vs_cpuref_bitwise": f"{ref_bit}/{n}",
        "gpu_vs_cpuref_max_abs": ref_max,
        "per_layer": summary,
        "verdict": ("PASS — codes bitwise; outputs bitwise"
                    if codes_ok == n and vnni_bit == n else
                    "PARTIAL — %d/%d bitwise; ULP-boundary cases "
                    "enumerated; native-source bitwise "
                    "certification remains OPEN; no tolerance "
                    "promotion" % (vnni_bit, n)),
        "cases": rows,
        "production_timing": None,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "cases"}, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
