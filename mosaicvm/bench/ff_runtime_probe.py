"""0xFF closure probe: what does the WORKING transformers path actually hold/run?

Loads the model exactly as the B3-S/RFIX harness did (bf16, device_map=auto,
66GiB GPU cap, trust_remote_code) and inspects:
  1. attention projection module types + weight/scale dtypes at runtime
  2. raw scale bytes (bf16-pattern check) after load
  3. one real forward: per-layer hidden-state finiteness
  4. capture (input, output) of one FP8 attention linear for effective-scale
     reverse engineering
"""
from __future__ import annotations

import json
import time
import torch

OUT = "/marimo/MosaicVM/mosaicvm/bench/rfix_out"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "66GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()
    log(f"loaded ({time.time()-t0:.0f}s)")

    import transformers
    log(f"transformers {transformers.__version__}, torch {torch.__version__}")

    # ---- 1+2: module types + scale state ----
    layer = model.model.layers[2]
    findings = {}
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj", "o_proj"):
        mod = getattr(layer.self_attn, name, None)
        if mod is None:
            # try alternate names
            cand = [n for n in dir(layer.self_attn)
                    if not n.startswith("_") and hasattr(
                        getattr(layer.self_attn, n), "weight")]
            log(f"  {name}: NOT FOUND; candidates: "
                f"{[c for c in cand if isinstance(getattr(layer.self_attn, c), torch.nn.Module)][:12]}")
            continue
        w = mod.weight
        s = getattr(mod, "weight_scale_inv", None)
        entry = {
            "type": type(mod).__name__,
            "weight_dtype": str(w.dtype),
            "weight_shape": tuple(w.shape),
            "scale_dtype": str(s.dtype) if s is not None else None,
            "scale_shape": tuple(s.shape) if s is not None else None,
        }
        if s is not None:
            sb = s.detach().view(torch.uint8).flatten()
            entry["scale_bytes_head"] = sb[:16].tolist()
            entry["scale_nan_f32"] = torch.isnan(
                s.detach().float()).any().item() if s.dtype not in (
                torch.uint8,) else None
        findings[name] = entry
        log(f"  {name}: {entry}")

    # list ALL attention child modules with weights
    log("attention children:")
    for n, m in layer.self_attn.named_children():
        if hasattr(m, "weight"):
            log(f"    {n}: {type(m).__name__} w={m.weight.dtype}")

    # ---- 3: one real forward, finiteness per layer ----
    ids = tok("def fibonacci(n):", return_tensors="pt").input_ids.cuda()
    finite_by_layer = []
    hooks = []

    def mk_hook(idx):
        def h(mod, args, output):
            t = output[0] if isinstance(output, tuple) else output
            finite_by_layer.append(
                (idx, bool(torch.isfinite(t.float()).all().item())))
        return h

    for i, lyr in enumerate(model.model.layers):
        hooks.append(lyr.register_forward_hook(mk_hook(i)))
    with torch.no_grad():
        out = model(ids)
    for h in hooks:
        h.remove()
    log(f"forward logits finite: {torch.isfinite(out.logits.float()).all().item()}")
    bad = [i for i, f in finite_by_layer if not f]
    log(f"non-finite layers: {bad if bad else 'NONE'}")

    # greedy continuation sanity
    with torch.no_grad():
        g = out.logits[0, -1].argmax().item()
    log(f"next-token id: {g} ({tok.decode([g])!r})")

    # ---- 4: capture one FP8 linear's (input, output) ----
    target = None
    for name in ("q_b_proj", "q_a_proj", "kv_a_proj"):
        m = getattr(layer.self_attn, name, None)
        if m is not None and hasattr(m, "weight_scale_inv"):
            target = (name, m)
            break
    if target is not None:
        name, m = target
        cap = {}

        def cap_hook(mod, args, kwargs, output):
            cap["x"] = args[0].detach().float().cpu()
            cap["y"] = output.detach().float().cpu()

        h = m.register_forward_hook(cap_hook, with_kwargs=True)
        with torch.no_grad():
            model(ids)
        h.remove()
        if cap:
            torch.save({
                "name": name, "x": cap["x"].half(),
                "y": cap["y"].half(),
                "w_dtype": str(m.weight.dtype),
                "w_shape": tuple(m.weight.shape),
                "s_dtype": str(m.weight_scale_inv.dtype),
                "s_shape": tuple(m.weight_scale_inv.shape),
                "s_bytes": m.weight_scale_inv.detach().view(torch.uint8)
                .cpu(),
                "w_bytes_head": m.weight.detach().view(torch.uint8).flatten()
                [:4096].cpu(),
            }, f"{OUT}/ff_probe_linear.pt")
            log(f"captured {name} IO: x{tuple(cap['x'].shape)} "
                f"y{tuple(cap['y'].shape)} -> ff_probe_linear.pt")

    # save findings
    with open(f"{OUT}/ff_probe_findings.json", "w") as f:
        json.dump({
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "modules": findings,
            "non_finite_layers": bad,
            "next_token": tok.decode([g]),
        }, f, indent=1, default=str)
    log("=== FF PROBE DONE ===")


if __name__ == "__main__":
    main()
