"""RTX 3060 overlap corpus vs Blackwell routes.

Compares generated token IDs, Top-6 expert IDs, hash-layer routes, and
route weights for the 32-token overlap corpus.

Requires MOSAICVM_MODEL_ROOT pointing at a full HF snapshot (experts
included). The local .mosaicvm/core/core.safetensors dump is core-only
and cannot produce Top-6 IDs.

Gate: NO TOP-6 ID CHANGES over the validation corpus.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

CORPUS = Path("artifacts/routes_overlap_corpus_3060.json")
OUT = Path("artifacts/routes_overlap_3060_result.json")
N_TOKENS = 32


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    corpus = json.loads(CORPUS.read_text())
    root = os.environ.get("MOSAICVM_MODEL_ROOT")
    result = {
        "classification": "RTX3060_DISPLAY_ATTACHED",
        "corpus_revision": corpus["model_revision"],
        "n_tokens": N_TOKENS,
        "prompts": [],
        "gate": {"NO_TOP6_ID_CHANGES": None, "token_exact": None},
    }
    if not root:
        # try HF snapshot
        snap = Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots" / corpus["model_revision"]
        idx = snap / "model.safetensors.index.json"
        if idx.exists():
            root = str(snap)
        else:
            result["status"] = "BLOCKED_NO_EXPERT_CHECKPOINT"
            result["reason"] = (
                "MOSAICVM_MODEL_ROOT unset and local HF snapshot has no "
                "model.safetensors.index.json / expert shards. "
                ".mosaicvm/core/core.safetensors is core-only (no routed "
                "experts) so Top-6 IDs cannot be produced."
            )
            OUT.write_text(json.dumps(result, indent=1))
            log(result["reason"])
            return 2

    log(f"loading streaming model from {root}")
    import torch
    torch.set_default_dtype(torch.bfloat16)  # production dtype: params must
    # be bf16 — fp32 doubles the host budget footprint and exhausted the
    # 14GB cap at layer 28 (ram_overflow -> random weights -> divergence)
    from mosaicvm.runtime.streaming_model import load_streaming_model
    from mosaicvm.model.layers import HashRouter, Router
    from transformers import AutoTokenizer

    def capture_one_prompt(model, tokenizer, prompt_text, capture_id,
                           n_positions, device, seed=42):
        """V4Model-native variant of route_capture_v2.capture_one_prompt.

        Same hook/record semantics; generate() takes the V4Model
        signature (no do_sample/use_cache kwargs).
        """
        state = {"records": [], "current_pos": None, "current_token": None,
                 "capture_id": capture_id}
        mod_layer = {}
        for name, mod in model.named_modules():
            if isinstance(mod, (Router, HashRouter)):
                layer_idx = None
                for part in name.split("."):
                    if part.isdigit():
                        layer_idx = int(part)
                mod_layer[id(mod)] = layer_idx

        def _hook(mod, args, output):
            scores, idx = output
            state["records"].append({
                "capture_id": capture_id,
                "pos": state["current_pos"],
                "token": state["current_token"],
                "layer": mod_layer.get(id(mod)),
                "router_class": "hash" if isinstance(mod, HashRouter)
                               else "learned",
                "expert_ids": idx.detach().cpu().tolist(),
                "weights": scores.detach().cpu().tolist(),
            })

        handles = [mod.register_forward_hook(_hook)
                   for name, mod in model.named_modules()
                   if isinstance(mod, (Router, HashRouter))]
        try:
            input_ids = tokenizer(
                prompt_text, return_tensors="pt").input_ids.to(device)
            torch.manual_seed(seed)
            with torch.no_grad():
                # single call: one prefill + incremental decodes (the
                # per-token full re-prefill loop was ~20x slower and
                # produced identical greedy tokens)
                state["current_pos"] = 0
                state["current_token"] = input_ids[0, -1].item()
                output = model.generate(input_ids,
                                        max_new_tokens=n_positions)
                generated = output[0, -n_positions:].tolist()
        finally:
            for h in handles:
                h.remove()
        return generated, input_ids[0].tolist(), state["records"]

    model, loader, missing = load_streaming_model(root, device="cuda")
    log(f"missing: {len(missing) if missing else 0}")
    if missing:
        import collections
        pat = collections.Counter()
        for m in missing:
            mm = m.split("shape")[0].replace("ram_overflow:", "")
            parts = mm.rsplit(".", 2)
            pat[".".join(parts[1:]) if len(parts) > 1 else mm] += 1
        log(f"missing pattern: {dict(pat)}")
        for m in missing[:8]:
            log(f"  miss: {m}")
    # VRAM relief on the display-attached 12GB card: keep the embedding on
    # CPU (forward is device-agnostic since the embed fix) — frees ~1GB.
    model.embed_tokens.to("cpu")
    import torch as _t
    _t.cuda.empty_cache()
    log(f"embed->cpu; VRAM now "
        f"{_t.cuda.memory_allocated()/1e9:.2f} GB allocated")
    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)

    token_exact = True
    top6_exact = True
    for p in corpus["prompts"]:
        cat = p["category"]
        expected = p["expected_generated_ids"][:N_TOKENS]
        log(f"[{cat}] generating {N_TOKENS} tokens...")
        t0 = time.time()
        generated, input_ids, records = capture_one_prompt(
            model, tok, p["prompt"], cat, N_TOKENS, device="cuda", seed=42)
        elapsed = time.time() - t0
        gen = generated[:N_TOKENS]
        tok_match = gen == expected
        token_exact = token_exact and tok_match
        first_div = next((i for i, (a, b) in enumerate(zip(gen, expected))
                          if a != b), None)
        rec = {
            "category": cat,
            "elapsed_s": round(elapsed, 1),
            "n_generated": len(gen),
            "n_records": len(records),
            "token_exact": tok_match,
            "first_token_divergence": first_div,
            "generated_ids": gen,
            "expected_ids": expected,
        }
        result["prompts"].append(rec)
        log(f"[{cat}] token_exact={tok_match} first_div={first_div} "
            f"({elapsed:.0f}s)")

    result["gate"]["token_exact"] = token_exact
    result["gate"]["NO_TOP6_ID_CHANGES"] = None  # no Blackwell per-pos IDs in overlap corpus (only tokens)
    result["status"] = "TOKEN_COMPARE_DONE"
    result["note"] = (
        "overlap corpus currently stores expected_generated_ids only; "
        "Top-6 ID compare needs Blackwell per-position records aligned "
        "to the same 32 tokens (from routes_real_256_raw_compact.json)."
    )
    OUT.write_text(json.dumps(result, indent=1))
    log(f"wrote {OUT}")
    return 0 if token_exact else 1


if __name__ == "__main__":
    sys.exit(main())
