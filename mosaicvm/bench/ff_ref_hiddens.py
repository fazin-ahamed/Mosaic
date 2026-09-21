"""§64 localization — per-layer reference hiddens for the two failing
prompts (math_reasoning, long_generation) from the AUTHORITATIVE
Blackwell transformers runtime. Small tensors, mirrored locally for
layer-by-layer comparison against the exact 3060 runtime.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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

    corpus = json.loads(Path(
        "/marimo/MosaicVM/artifacts/routes_overlap_corpus_3060.json"
    ).read_text())
    targets = [p for p in corpus["prompts"]
               if p["category"] in ("math_reasoning", "long_generation")]

    for p in targets:
        cat = p["category"]
        ids = tok(p["prompt"], return_tensors="pt").input_ids.cuda()
        expected0 = p["expected_generated_ids"][0]

        # hook every layer's OUTPUT (post-block hidden) + embeddings
        caps = {}

        def mk(name):
            def h(mod, args, output):
                t = output[0] if isinstance(output, tuple) else output
                caps[name] = t.detach().float().cpu().half()
            return h

        handles = []
        handles.append(model.model.embed_tokens.register_forward_hook(
            mk("embed")))
        for i, lyr in enumerate(model.model.layers):
            handles.append(lyr.register_forward_hook(mk(f"layer{i}")))
        handles.append(model.model.norm.register_forward_hook(mk("norm")))

        with torch.no_grad():
            out = model(ids)
        for h in handles:
            h.remove()

        greedy0 = out.logits[0, -1].argmax().item()
        # top-5 logits for tie inspection
        topv, topi = out.logits[0, -1].float().topk(5)
        log(f"[{cat}] S={ids.shape[1]} greedy0={greedy0} "
            f"expected0={expected0} "
            f"{'MATCH' if greedy0 == expected0 else 'DIFF'}")
        log(f"  top5: {[(int(i), round(float(v), 4)) "
            f"for i, v in zip(topi, topv)]}")

        save = {"category": cat, "S": ids.shape[1],
                "greedy0": greedy0, "expected0": expected0,
                "top5_ids": topi.tolist(), "top5_vals": topv.tolist(),
                "input_ids": ids[0].tolist(), **caps}
        torch.save(save, f"{OUT}/ref_hiddens_{cat}.pt")
        log(f"[{cat}] saved {len(caps)} tensors")

    log("=== REF HIDDENS CAPTURE DONE ===")


if __name__ == "__main__":
    main()
