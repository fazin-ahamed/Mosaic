"""0xFF closure probe 3: loading introspection.

from_pretrained with output_loading_info=True to capture exactly which
keys loaded / were missing / mismatched, plus quantizer state.
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
    model, info = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto", output_loading_info=True,
        max_memory={0: "88GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()
    log(f"loaded ({time.time()-t0:.0f}s)")

    print("loading info keys:", list(info.keys()))
    for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
              "error_msgs"):
        v = info.get(k, [])
        print(f"{k}: {len(v)}")
        att = [x for x in v if "attn" in x or "self_attn" in x][:10]
        for x in att:
            print("   ", x)
        if not att and v:
            print("    sample:", v[:5])

    # quantizer state
    import transformers
    print("\ntransformers:", transformers.__version__)
    q = getattr(model, "hf_quantizer", None)
    if q is not None:
        print("quantizer:", type(q).__name__)
        print("  pre_quantized:", getattr(q, "pre_quantized", None))
        print("  dequantize:",
              getattr(q.quantization_config, "dequantize", "N/A")
              if hasattr(q, "quantization_config") else "N/A")
        print("  modules_to_not_convert:",
              list(getattr(q, "modules_to_not_convert", []))[:10])

    # save info
    with open(f"{OUT}/ff_probe3_info.json", "w") as f:
        json.dump({k: (v if isinstance(v, list) else str(v))
                   for k, v in info.items()}, f, indent=1)
    log("=== FF PROBE 3 DONE ===")


if __name__ == "__main__":
    main()
