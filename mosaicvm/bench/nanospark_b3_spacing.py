"""NanoSpark B3: state-spacing race (roadmap §42-47, the decisive experiment).

For each Δ ∈ {1, 2, 3, 5}: measure how functionally degraded the DSpark
draft becomes when the main_x state is Δ tokens stale.

Mode S (isolated sensitivity): at each probe position, compare the draft
produced with the true state at pos+Δ against the draft produced with the
state at pos (hold-Δ). Zero drafter rings before every probe.

Also train rank-64 low-rank at each Δ spacing and evaluate its drafts.

Metrics per Δ:
  state cosine, rel-error (hold vs true at Δ)
  KL (hold-state draft vs true-state draft)
  KL (rank-64 draft vs true-state draft)
  Top-1/2/4/8/16 coverage (true top-1 in hold draft's top-k)
  Top-1/2/4/8/16 coverage (true top-1 in r64 draft's top-k)
  λ_flip (correction fraction for top-1 recovery)
  per-token degradation: KL(Δ)/Δ

Classification: BLACKWELL_SANDBOX_ONLY + DRAFTER_NUMERICS_DEVIATION_
RECORDED.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/marimo/MosaicVM")

import torch

from mosaicvm.bench.dspark_wire import (
    BLOCK_SIZE, N_STAGES, PROMPTS, TARGET_LAYERS,
    DSparkStage, FP4_TABLE_VALUES, load_mtp_tensors,
    precompute_freqs_cis)

OUT = "/marimo/MosaicVM/artifacts"
B3_GPU_GIB = 66              # leave ~20 GiB for expert dequant cycling
SPACINGS = (1, 2, 3, 5)
N_PROBE = 16               # positions per domain (B3-R: 7x16=112/Δ)
EVAL_DOMAINS = [c for c, _ in PROMPTS]  # all 7 domains (B3-R)
TRAIN_DOMAINS = {"structured_json", "long_context_retrieval",
                 "long_generation", "agent_tool_use", "general_conversation"}
TOPKS = (1, 2, 4, 8, 16)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t00 = time.time()
    os.makedirs(OUT, exist_ok=True)

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

    from safetensors import safe_open
    idx = json.load(open(os.path.join(
        model_dir, "model.safetensors.index.json")))

    def ckpt(key):
        with safe_open(os.path.join(model_dir, idx["weight_map"][key]),
                       framework="pt", device="cpu") as f:
            return f.get_tensor(key).to("cuda")

    head_w = ckpt("head.weight")
    embed_weight = model.model.embed_tokens.weight.data
    if embed_weight.device.type != "cuda":
        embed_weight = ckpt("embed.weight")
    dev = head_w.device
    embed_f = embed_weight.to(torch.float32)

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"loaded ({time.time()-t00:.0f}s)")

    dataset = torch.load(f"{OUT}/nanospark_states_210.pt")
    N_CAP = dataset[PROMPTS[0][0]]["mx"].shape[0]
    log(f"dataset: {len(dataset)} domains × {N_CAP} positions")

    def draft_with_mx(input_ids, main_x, start_pos):
        ids = input_ids.reshape(-1)
        draft_ids = ids.new_full((ids.size(0), BLOCK_SIZE), 128799)
        draft_ids[:, 0] = ids
        x = torch.nn.functional.embedding(
            draft_ids, embed_weight.to(torch.bfloat16))
        x = x.unsqueeze(2).repeat(1, 1, 4, 1)
        h = x
        for st in stages:
            h = st(h, start_pos, main_x)
        return stages[-1].forward_head(h, input_ids, head_w)

    def draft_topk(s, tok_tensor, sp, k=16):
        """Draft and return (topk_ids, topk_scores, logits)."""
        out = draft_with_mx(tok_tensor, s[None, None, :], sp)
        lg = out["base_logits"][0, 0].float()
        topv, topi = lg.topk(min(k, lg.numel()))
        return topi.tolist(), topv.tolist(), lg

    # ---- train rank-64 at each spacing ----
    log("training rank-64 at each spacing...")
    predictors = {}
    for delta in SPACINGS:
        S_tr, TOK_tr, T_tr = [], [], []
        for d in TRAIN_DOMAINS:
            v = dataset[d]
            mx = v["mx"].float().cuda()
            for i in range(0, N_CAP - delta, delta):
                S_tr.append(mx[i])
                TOK_tr.append(v["tokens"][i])
                T_tr.append(mx[i + delta])
        S_tr = torch.stack(S_tr)
        T_tr = torch.stack(T_tr)
        D_tr = T_tr - S_tr
        X_tr = torch.stack([
            torch.cat([s, embed_f[t]]) for s, t in zip(S_tr, TOK_tr)])
        Xc = X_tr - X_tr.mean(0)
        _, sig, Vt = torch.linalg.svd(Xc, full_matrices=False)
        r = 64
        V_r = Vt[:r].T
        Z = (X_tr - X_tr.mean(0)) @ V_r
        Z_std = Z.std(0).clamp_min(1e-6)
        Zn = (Z - Z.mean(0)) / Z_std
        lam = 1.0
        A = Zn.T @ Zn + lam * torch.eye(r, device=dev)
        U_coef = torch.linalg.solve(A, Zn.T @ (D_tr - D_tr.mean(0)))
        predictors[delta] = {
            "V": V_r, "U": U_coef,
            "Z_raw_mean": Z.mean(0), "D_mean": D_tr.mean(0),
            "x_mean": X_tr.mean(0), "Z_std": Z_std,
            "n_pairs": S_tr.shape[0],
        }
        log(f"  Δ={delta}: {S_tr.shape[0]} training pairs")

    def rank64_pred(s, tok, delta):
        p = predictors[delta]
        x = torch.cat([s, embed_f[tok]])
        z = (x - p["x_mean"]) @ p["V"]
        z = (z - p["Z_raw_mean"]) / p["Z_std"]
        return s + z @ p["U"] + p["D_mean"]

    # ---- B3.2: functional sensitivity per spacing ----
    log(f"B3.2: functional sensitivity ({N_PROBE} probes × "
        f"{len(EVAL_DOMAINS)} domains × {len(SPACINGS)} spacings)")

    results = {"per_spacing": {}, "per_domain": {}}

    for delta in SPACINGS:
        d_key = str(delta)
        results["per_spacing"][d_key] = {"per_domain": {}}
        all_top1_h, all_top1_r = [], []
        all_topk_h = {k: [] for k in TOPKS}
        all_topk_r = {k: [] for k in TOPKS}
        all_kl_h, all_kl_r = [], []
        all_cos, all_relerr = [], []

        for cat in EVAL_DOMAINS:
            v = dataset[cat]
            mx_all = v["mx"].float().cuda()
            tokens = v["tokens"]
            dom_top1_h, dom_top1_r = [], []
            dom_topk_h = {k: [] for k in TOPKS}
            dom_topk_r = {k: [] for k in TOPKS}
            dom_kl_h, dom_kl_r = [], []
            dom_cos, dom_relerr = [], []

            for pi in range(N_PROBE):
                pos = delta + pi * ((N_CAP - delta) // N_PROBE)
                if pos >= N_CAP or pos - delta < 0:
                    continue

                s_true = mx_all[pos]         # state at pos
                s_hold = mx_all[pos - delta] # state at pos-delta
                tok = tokens[pos]
                tok_tensor = torch.tensor([[tok]], device=dev)
                sp = pos

                def zero_rings():
                    for st in stages:
                        st.attn.kv_cache.zero_()
                        st.ffn.cache.clear()

                # true-state draft (fresh ring)
                zero_rings()
                t_ids, t_scores, t_logits = draft_topk(
                    s_true, tok_tensor, sp)
                t_top1 = t_ids[0]

                # hold-state draft (fresh ring)
                zero_rings()
                h_ids, h_scores, h_logits = draft_topk(
                    s_hold, tok_tensor, sp)

                # rank-64 draft (fresh ring)
                s_r64 = rank64_pred(s_hold, tokens[pos - delta], delta)
                zero_rings()
                r_ids, r_scores, r_logits = draft_topk(
                    s_r64, tok_tensor, sp)

                # state metrics
                cos = torch.nn.functional.cosine_similarity(
                    s_hold.unsqueeze(0), s_true.unsqueeze(0)).item()
                relerr = ((s_hold - s_true).norm()
                          / s_true.norm()).item()

                # KL
                t_logp = torch.log_softmax(t_logits, dim=-1)
                h_logp = torch.log_softmax(h_logits, dim=-1)
                kl_h = torch.nn.functional.kl_div(
                    h_logp, t_logp, reduction="sum", log_target=True).item()
                r_logp = torch.log_softmax(r_logits, dim=-1)
                kl_r = torch.nn.functional.kl_div(
                    r_logp, t_logp, reduction="sum", log_target=True).item()

                # top-k coverage: is the true-state top-1 in the
                # hold/r64 draft's top-k?
                top1_h = h_ids[0] == t_top1
                top1_r = r_ids[0] == t_top1
                for k in TOPKS:
                    topk_h = t_top1 in h_ids[:k]
                    topk_r = t_top1 in r_ids[:k]
                    dom_topk_h[k].append(topk_h)
                    dom_topk_r[k].append(topk_r)
                dom_top1_h.append(top1_h)
                dom_top1_r.append(top1_r)
                dom_kl_h.append(kl_h)
                dom_kl_r.append(kl_r)
                dom_cos.append(cos)
                dom_relerr.append(relerr)

                # periodic VRAM cleanup (once per probe, not per draft)
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()

            # domain summary
            def _rate(lst):
                return round(sum(lst) / len(lst), 3) if lst else None

            def _mean(lst):
                return round(sum(lst) / len(lst), 4) if lst else None

            dom_summary = {
                "n_probes": len(dom_top1_h),
                "top1_hold": _rate(dom_top1_h),
                "top1_r64": _rate(dom_top1_r),
                "kl_hold": _mean(dom_kl_h),
                "kl_r64": _mean(dom_kl_r),
                "cosine": _mean(dom_cos),
                "rel_err": _mean(dom_relerr),
                    "topk_hold": {str(k): _rate(dom_topk_h[k])
                              for k in TOPKS},
                "topk_r64": {str(k): _rate(dom_topk_r[k])
                             for k in TOPKS},
            }
            results["per_spacing"][d_key]["per_domain"][cat] = dom_summary
            # accumulate
            all_top1_h.extend(dom_top1_h)
            all_top1_r.extend(dom_top1_r)
            for k in TOPKS:
                all_topk_h[k].extend(dom_topk_h[k])
                all_topk_r[k].extend(dom_topk_r[k])
            all_kl_h.extend(dom_kl_h)
            all_kl_r.extend(dom_kl_r)
            all_cos.extend(dom_cos)
            all_relerr.extend(dom_relerr)
            log(f"  Δ={delta} [{cat}] top1_hold={dom_summary['top1_hold']} "
                f"top1_r64={dom_summary['top1_r64']} "
                f"top4_hold={dom_summary['topk_hold']['4']} "
                f"top16_hold={dom_summary['topk_hold']['16']} "
                f"KL={dom_summary['kl_hold']}")

        # aggregate per spacing
        def _rate(lst):
            return round(sum(lst) / len(lst), 3) if lst else None

        def _mean(lst):
            return round(sum(lst) / len(lst), 4) if lst else None

        def _count(lst):
            return f"{sum(lst)}/{len(lst)}" if lst else "0/0"

        def _wilson_ci(lst, z=1.96):
            if not lst:
                return (None, None)
            n = len(lst)
            p = sum(lst) / n
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            margin = z * ((p * (1 - p) / n
                           + z * z / (4 * n * n)) ** 0.5) / denom
            return (round(max(0, center - margin), 3),
                    round(min(1, center + margin), 3))

        kl_mean = _mean(all_kl_h)
        agg = {
            "n_probes": len(all_top1_h),
            "top1_hold": _rate(all_top1_h),
            "top1_hold_count": _count(all_top1_h),
            "top1_hold_ci": _wilson_ci(all_top1_h),
            "top1_r64": _rate(all_top1_r),
            "kl_hold": kl_mean,
            "kl_r64": _mean(all_kl_r),
            "cosine": _mean(all_cos),
            "rel_err": _mean(all_relerr),
            "kl_per_token": round(kl_mean / delta, 4) if kl_mean else None,
        }
        for k in TOPKS:
            agg["topk_hold"] = agg.get("topk_hold", {})
            agg["topk_hold"][str(k)] = _rate(all_topk_h[k])
            agg["topk_hold_counts"] = agg.get("topk_hold_counts", {})
            agg["topk_hold_counts"][str(k)] = _count(all_topk_h[k])
            agg["topk_hold_ci"] = agg.get("topk_hold_ci", {})
            agg["topk_hold_ci"][str(k)] = _wilson_ci(all_topk_h[k])
            agg["topk_r64"] = agg.get("topk_r64", {})
            agg["topk_r64"][str(k)] = _rate(all_topk_r[k])
        results["per_spacing"][d_key]["aggregate"] = agg
        agg = results["per_spacing"][d_key]["aggregate"]
        log(f"  Δ={delta} AGG: top1={agg['top1_hold']} "
            f"top4={agg['topk_hold']['4']} "
            f"top16={agg['topk_hold']['16']} "
            f"KL={agg['kl_hold']} KL/Δ={agg['kl_per_token']}")

    # ---- summary table ----
    log("\n=== B3 SPACING RACE RESULTS ===")
    log(f"{'Δ':>3} {'top1':>6} {'top2':>6} {'top4':>6} {'top8':>6} "
        f"{'top16':>6} {'KL':>7} {'KL/Δ':>6} {'cos':>6}")
    log("-" * 60)
    for delta in SPACINGS:
        a = results["per_spacing"][str(delta)]["aggregate"]
        log(f"{delta:>3} {a['top1_hold']:>6} "
            f"{a['topk_hold']['2']:>6} {a['topk_hold']['4']:>6} "
            f"{a['topk_hold']['8']:>6} {a['topk_hold']['16']:>6} "
            f"{a['kl_hold']:>7} {a['kl_per_token']:>6} "
            f"{a['cosine']:>6}")

    # save
    out = {
        "phase": "NanoSpark-B3-spacing-race",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "design": {
            "spacings": list(SPACINGS),
            "mode": "isolated sensitivity (Mode S; rings zeroed before EVERY draft call)",
            "note": "Mode R (sequential recursion) is B3.3, a separate experiment",
            "n_probes": N_PROBE,
            "eval_domains": EVAL_DOMAINS,
            "train_domains": sorted(TRAIN_DOMAINS),
            "predictor": "rank-64 PCA+ridge",
        },
        "results": results,
    }
    with open(f"{OUT}/nanospark_b3_spacing.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {OUT}/nanospark_b3_spacing.json")
    log(f"=== DONE in {(time.time()-t00)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
