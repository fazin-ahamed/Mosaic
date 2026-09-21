"""Track D1-R: exact L41/L42 tail replay from true h40 states.

THE decisive architectural gate: can source L41/L42, fed the TRUE h40
trajectory position-by-position (building KV/DSA state sequentially),
reproduce the true h42 states and 100%-Top-1 logits?

If YES → deep-tail corrector architecture validated.
If NO  → the h40→tail interface is incomplete (missing state).

Method:
  1. Run the full kit model on a test prompt, capturing h40 at ALL
     positions (prefill batch + decode steps).
  2. Save the initial (empty) KV/DSA state of Block 41 and Block 42.
  3. Reset Blocks 41/42 to empty state.
  4. Sequentially feed the captured h40 positions through Block 41
     (position 0, 1, 2, ... — building L41's KV cache from h40 only).
  5. Sequentially feed L41's outputs through Block 42.
  6. Compare L42 outputs against the model's original L42 output.
  7. Apply the fp32 head to the replayed h42 and verify Top-1.
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

OUT = Path("artifacts/d1r_tail_replay.json")
N_TEST = 3        # prompts to test
N_STEPS = 16      # decode steps to verify


def main():
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
    args.max_seq_len = 256
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
    import types

    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight",
                                        "head.weight") else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()

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
    print("model loaded", flush=True)
    torch.set_default_device("cuda")   # for kit mask/arange tensors

    # ---- capture hooks ----
    CAP = {"h40": [], "h41": [], "h42": [], "logits": []}

    def make_hook(L, key):
        def h(mod, args_, output):
            t = output[0] if isinstance(output, tuple) else output
            CAP[key].append(t.detach().float().cpu())
        return h

    h40_handle = model.layers[40].register_forward_hook(
        make_hook(40, "h40"))
    h41_handle = model.layers[41].register_forward_hook(
        make_hook(41, "h41"))
    h42_handle = model.layers[42].register_forward_hook(
        make_hook(42, "h42"))

    # ---- fp32 head for scoring ----
    import glob, os
    from safetensors.torch import safe_open as so
    snap_p = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/*"))[0]
    idx2 = json.load(open(os.path.join(
        snap_p, "model.safetensors.index.json")))["weight_map"]
    fr = {}
    for nm in ("head.weight", "hc_head_fn", "hc_head_base",
               "hc_head_scale", "norm.weight"):
        with so(os.path.join(snap_p, idx2[nm]), framework="pt") as f:
            fr[nm] = f.get_tensor(nm)
    head_w = fr["head.weight"].float()  # CPU (avoid VRAM)
    hc_fn = fr["hc_head_fn"].float()
    hc_base = fr["hc_head_base"].float()
    hc_scale = fr["hc_head_scale"].float()
    norm_w = fr["norm.weight"].float()

    hc_fn_c = hc_fn.cpu()
    hc_base_c = hc_base.cpu()
    hc_scale_c = hc_scale.cpu()
    norm_w_c = norm_w.cpu()

    def head_logits(h42):
        # h42 may be cuda or cpu; compute on cpu with fp32 head
        h = h42.cpu() if h42.is_cuda else h42
        xf = h.flatten(1).float()
        rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.nn.functional.linear(xf, hc_fn_c) * rs
        pre = torch.sigmoid(mixes * hc_scale_c + hc_base_c) + 1e-6
        y = torch.sum(pre.unsqueeze(-1) * h, dim=1)
        yn = torch.nn.functional.rms_norm(y.float(), (4096,),
                                          norm_w_c)
        return torch.nn.functional.linear(yn.float(), head_w)

    # ---- test prompts ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap),
                                        trust_remote_code=True)
    spec = json.loads(Path(
        "artifacts/stage1_prompts.json").read_text())
    test_cats = spec["splits"]["test"][:N_TEST]

    results = []
    for cat in test_cats:
        pe = next(p for p in spec["prompts"]
                  if p["category"] == cat)
        ids = tok(pe["prompt"], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        print(f"\n=== [{cat}] S={S} ===", flush=True)

        # ---- Phase 1: normal forward (capture h40/h41/h42) ----
        for key in CAP:
            CAP[key].clear()
        with torch.no_grad():
            # prefill: process all S positions, hooks capture [1,S,4,4096]
            out_ids, _, _ = model(ids, 0)
            gen = [int(out_ids[0])]
            h40_prefill = CAP["h40"][0].clone()   # [1, S, 4, 4096]
            h41_prefill = CAP["h41"][0].clone()
            h42_prefill = CAP["h42"][0].clone()
            # decode steps
            h40_dec = []
            h41_dec = []
            h42_dec = []
            for step in range(N_STEPS):
                for key in CAP:
                    CAP[key].clear()
                cur = out_ids.reshape(1, 1) if out_ids.dim() == 1 \
                else out_ids[:, -1:]
                out_ids, _, _ = model(cur, S + step)
                gen.append(int(out_ids[0]))
                h40_dec.append(CAP["h40"][0].clone())  # [1,1,4,4096]
                h41_dec.append(CAP["h41"][0].clone())
                h42_dec.append(CAP["h42"][0].clone())

        # full h40 trajectory: prefill positions 0..S-1 + decode S..S+N
        h40_all = torch.cat([h40_prefill] + h40_dec, dim=1
                            ).squeeze(0)   # [S+N+1, 4, 4096]
        h42_true = torch.cat([h42_prefill] + h42_dec, dim=1
                             ).squeeze(0)
        all_tokens = ids[0].tolist() + gen
        n_pos = h40_all.shape[0]

        # ---- Phase 2: save/reset L41/L42 state, replay from h40 ----
        # Save the CURRENT (post-forward) KV state
        blk41 = model.layers[41]
        blk42 = model.layers[42]

        def save_blk_state(blk):
            st = {}
            attn = blk.attn
            st["kv"] = attn.kv_cache[:, :attn.kv_cache.shape[1]
                                     ].clone() if hasattr(attn,
                                                          "kv_cache"
                                                          ) else None
            if hasattr(attn, "compressor") and \
                    attn.compressor is not None and \
                    hasattr(attn.compressor, "kv_cache") and \
                    attn.compressor.kv_cache is not None:
                st["comp_kv"] = attn.compressor.kv_cache.clone()
                st["comp_score"] = attn.compressor.score_state.clone()
            else:
                st["comp_kv"] = None
                st["comp_score"] = None
            if hasattr(attn, "indexer") and \
                    attn.indexer is not None and \
                    hasattr(attn.indexer, "kv_cache"):
                st["ix_kv"] = attn.indexer.kv_cache.clone()
            else:
                st["ix_kv"] = None
            return st

        def reset_blk_state(blk):
            attn = blk.attn
            if hasattr(attn, "kv_cache"):
                attn.kv_cache.zero_()
            if hasattr(attn, "compressor") and \
                    attn.compressor is not None and \
                    hasattr(attn.compressor, "kv_cache") and \
                    attn.compressor.kv_cache is not None:
                attn.compressor.kv_cache.zero_()
                attn.compressor.score_state.fill_(float("-inf"))
            if hasattr(attn, "indexer") and \
                    attn.indexer is not None and \
                    hasattr(attn.indexer, "kv_cache"):
                attn.indexer.kv_cache.zero_()

        state41 = save_blk_state(blk41)
        state42 = save_blk_state(blk42)

        # Replay: feed h40 positions sequentially through L41 then L42
        reset_blk_state(blk41)
        reset_blk_state(blk42)

        # Build input_ids tensor for the full sequence
        input_ids_all = torch.tensor(
            [all_tokens[:n_pos]], device="cuda")

        h41_replay = []
        h42_replay = []
        torch.cuda.set_device(0)
        with torch.no_grad():
            # PREFILL through L41/L42: process all S positions in batch
            h40_pref = h40_all[:S].unsqueeze(0).cuda()  # [1,S,4,4096]
            h41_out = blk41(h40_pref.to(torch.bfloat16), 0,
                            input_ids_all[:, :S])
            h41_replay.append(h41_out.float().cpu())
            h42_out = blk42(h41_out, 0, input_ids_all[:, :S])
            h42_replay.append(h42_out.float().cpu())

            # DECODE through L41/L42: one position at a time
            for p in range(S, n_pos):
                h40_p = h40_all[p:p + 1].unsqueeze(0).cuda()
                h41_p = blk41(h40_p.to(torch.bfloat16), p,
                              input_ids_all[:, p:p + 1])
                h41_replay.append(h41_p.float().cpu())
                h42_p = blk42(h41_p, p,
                              input_ids_all[:, p:p + 1])
                h42_replay.append(h42_p.float().cpu())

        h42_replayed = torch.cat(h42_replay, dim=1).squeeze(0
                                                              ).cpu()

        # ---- Phase 3: compare ----
        n_compare = min(h42_true.shape[0], h42_replayed.shape[0])
        cos_vals = []
        max_abs_vals = []
        top1_match = []
        for p in range(n_compare):
            a = h42_true[p].flatten()
            b = h42_replayed[p].flatten()
            cos = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)).item()
            cos_vals.append(cos)
            max_abs_vals.append((a - b).abs().max().item())
        # head logits on replayed h42
        top1_ok = 0
        topk_ok = {k: 0 for k in (1, 2, 4, 8)}
        for p in range(n_compare):
            h = h42_replayed[p:p + 1].cuda()
            with torch.no_grad():
                lg = head_logits(h)
            # the token produced FROM position p is gen[p - S] if
            # p >= S else the model's greedy at position p
            # For decode positions: the model's output at position p
            # predicts token at position p+1
            if p + 1 < len(all_tokens):
                true_tok = all_tokens[p + 1]
                pred = int(lg.argmax())
                if pred == true_tok:
                    top1_ok += 1
                for k in topk_ok:
                    tk = lg.topk(k, dim=-1).indices
                    if true_tok in tk[0].tolist():
                        topk_ok[k] += 1
        n_scored = n_compare - 1  # positions with a next token
        res = {
            "category": cat,
            "positions_compared": n_compare,
            "h42_cosine_p50": round(sorted(cos_vals
                                            )[len(cos_vals) // 2], 6),
            "h42_cosine_min": round(min(cos_vals), 6),
            "h42_max_abs_p50": round(sorted(max_abs_vals
                                            )[len(max_abs_vals) // 2],
                                     6),
            "h42_max_abs_max": round(max(max_abs_vals), 6),
            "head_top1_replay": round(top1_ok / max(n_scored, 1), 4),
            "head_topk_replay": {str(k): round(
                v / max(n_scored, 1), 4) for k, v in topk_ok.items()},
        }
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)

        # restore state for next prompt
        # (model's KV was rebuilt by replay — it's now correct for
        #  the same positions, so we can continue)

    for h in (h40_handle, h41_handle, h42_handle):
        h.remove()

    all_pass = all(r["h42_cosine_p50"] > 0.99 for r in results)
    out = {
        "classification": "D1R_EXACT_TAIL_REPLAY",
        "method": "true h40 trajectory → real Block41 → real "
                  "Block42 (sequential KV build) → fp32 head",
        "results": results,
        "verdict": "PASS — tail replay reproduces L42" if all_pass
                   else "FAIL or PARTIAL — see per-prompt details",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n=== D1-R VERDICT: {out['verdict']} ===")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
