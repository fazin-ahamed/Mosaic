"""B4-O prep: capture per-domain PROMPT states (mh_full) for ring prefill.

The 210-position dataset starts at the prompt end; the chain/oracle
prefill needs mx for ALL prompt positions (draft_with_mx(..., 0) fills
rings from the full-prompt projection). This capture runs the target
prefill once per domain with the same hooks as B3-S and saves mh_full.

Output: rfix_out/b4o_prompt_states_{domain}.pt  (fp16 [1, n_prompt, 3d])
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/marimo/MosaicVM")

import torch

from mosaicvm.bench.dspark_wire import (
    N_STAGES, PROMPTS, TARGET_LAYERS,
    DSparkStage, FP4_TABLE_VALUES, load_mtp_tensors,
    precompute_freqs_cis)

OUT = "/marimo/MosaicVM/mosaicvm/bench/rfix_out"
B3_GPU_GIB = 66


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _save_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    os.sync()


def main():
    t00 = time.time()

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{B3_GPU_GIB}GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()
    log(f"target loaded ({time.time()-t00:.0f}s)")

    captured = []

    def hook(mod, args, output):
        captured.append(output.detach().mean(dim=2))

    def target_forward(tokens):
        captured.clear()
        hs = [model.model.layers[li].register_forward_hook(hook)
              for li in TARGET_LAYERS]
        try:
            with torch.no_grad():
                o = model(tokens, use_cache=False)
            mh = torch.cat(captured, dim=-1)
        finally:
            for h in hs:
                h.remove()
            captured.clear()
        return o.logits, mh

    for cat, prompt_text in PROMPTS:
        path = f"{OUT}/b4o_prompt_states_{cat}.pt"
        if os.path.exists(path):
            log(f"  [{cat}] already captured, skipping")
            continue
        ids = tok(prompt_text, return_tensors="pt").input_ids.cuda()
        logits, mh_full = target_forward(ids)
        _save_atomic({
            "mh_full": mh_full.half().cpu(),       # [1, n_prompt, 3*dim]
            "n_prompt": ids.shape[1],
            "last_tok": ids[0, -1].item(),
        }, path)
        log(f"  [{cat}] prompt states: {tuple(mh_full.shape)}")

    log(f"=== prompt-states DONE in {(time.time()-t00)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
