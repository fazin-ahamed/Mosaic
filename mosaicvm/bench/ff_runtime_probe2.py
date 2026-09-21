"""0xFF closure probe 2: full runtime attention state extraction.

For every layer (0..42) and every attention projection:
  - sha256 of the runtime weight bytes (compare vs core archive)
  - the runtime weight_scale_inv bytes (e8m0)
Plus a full (W, s, x, y) capture of layers.2.q_b_proj for exact
GEMM-semantics verification.

Answers:
  (a) do runtime codes == checkpoint codes?
  (b) what are the runtime scales for ALL projections?
  (c) does y == x @ (codes * 2^(scale_byte-127))?
"""
from __future__ import annotations

import hashlib
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
        max_memory={0: "88GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()
    log(f"loaded ({time.time()-t0:.0f}s)")

    report = {"layers": {}}
    scale_dump = {}

    for L in range(len(model.model.layers)):
        sa = model.model.layers[L].self_attn
        entry = {}
        for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj",
                     "o_b_proj"):
            m = getattr(sa, name, None)
            if m is None or not hasattr(m, "weight_scale_inv"):
                continue
            w = m.weight.detach()
            s = m.weight_scale_inv.detach()
            if w.is_meta or s.is_meta:
                entry[name] = {"status": "META (offloaded)"}
                continue
            wb = w.view(torch.uint8).cpu().contiguous()
            sb = s.view(torch.uint8).cpu().contiguous().flatten()
            entry[name] = {
                "w_shape": tuple(w.shape),
                "w_sha256": hashlib.sha256(
                    wb.numpy().tobytes()).hexdigest()[:16],
                "s_shape": tuple(s.shape),
                "s_uniques": torch.unique(sb).tolist(),
                "s_bytes_b64_len": sb.numel(),
            }
            scale_dump[f"{L}.{name}"] = sb
        report["layers"][str(L)] = entry
        if L % 8 == 0:
            log(f"  layer {L} done")

    torch.save(scale_dump, f"{OUT}/ff_runtime_scales.pt")

    # full capture of layers.2.q_b_proj for GEMM verification
    m = model.model.layers[2].self_attn.q_b_proj
    cap = {}

    def hook(mod, args, kwargs, output):
        cap["x"] = args[0].detach().float().cpu()
        cap["y"] = output.detach().float().cpu()

    h = m.register_forward_hook(hook, with_kwargs=True)
    ids = tok("def fibonacci(n):", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)
    h.remove()
    torch.save({
        "x": cap["x"].half(), "y": cap["y"].half(),
        "w_bytes": m.weight.detach().view(torch.uint8).cpu().contiguous(),
        "s_bytes": m.weight_scale_inv.detach().view(torch.uint8)
        .cpu().contiguous().flatten(),
    }, f"{OUT}/ff_gemm_verify.pt")
    log("captured full q_b_proj IO + weights")

    # per-layer scale-byte histogram summary
    allu = {}
    for k, v in scale_dump.items():
        for b in torch.unique(v).tolist():
            allu[b] = allu.get(b, 0) + 1
    meta_count = sum(1 for L in report["layers"].values()
                     for v in L.values() if v.get("status") == "META (offloaded)")
    report["meta_offloaded_tensors"] = meta_count
    report["scale_byte_histogram"] = dict(
        sorted(allu.items(), key=lambda kv: -kv[1])[:20])
    with open(f"{OUT}/ff_probe2_report.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    log(f"scale byte histogram (top): "
        f"{dict(sorted(allu.items(), key=lambda kv: -kv[1])[:10])}")
    log("=== FF PROBE 2 DONE ===")


if __name__ == "__main__":
    main()
