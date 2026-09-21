"""MosaicVM Modal entrypoints.

All remote jobs live here; the only place Modal-specific config exists.
"""
from __future__ import annotations

import json
import os
import modal

app = modal.App("mosaicvm")


MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
VOLUME_NAME = "mosaicvm-model"

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]", "hf_transfer")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=download_image,
    cpu=4,
    memory=16384,
    volumes={"/vol/model": model_volume},
    timeout=60 * 60 * 24,
)
def download_model() -> dict:
    """Stage the full HF snapshot onto the Modal volume (resumable)."""
    from huggingface_hub import snapshot_download

    target = "/vol/model/hub"
    path = snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=target,
        max_workers=16,
    )
    cfg_path = os.path.join(target, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print("=== authoritative config.json ===")
    print(json.dumps(cfg, indent=2))
    return {"snapshot_path": str(path), "config": cfg}


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "triton",
        "pynvml",
        "numpy",
        "safetensors",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "kernels==0.16.0",
    )
    .add_local_python_source("mosaicvm")
)


def _write_artifact(name: str, payload: dict) -> str:
    os.makedirs("/vol/model/artifacts", exist_ok=True)
    path = f"/vol/model/artifacts/{name}"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {path}")
    model_volume.commit()
    return path


@app.function(
    image=gpu_image,
    gpu="A10",
    cpu=8,
    memory=49152,
    volumes={"/vol/model": model_volume},
    timeout=60 * 30,
)
def hw_profile() -> dict:
    import sys

    sys.path.insert(0, "/root")
    from mosaicvm.profiler.hw_profile import run

    profile = run(volume_path="/vol/model/hub")
    _write_artifact("hw_profile_A10.json", profile)
    print(json.dumps({k: profile[k] for k in ("gpu", "vram_bandwidth") if k in profile}, indent=2, default=str))
    return json.loads(json.dumps(profile, default=str))


@app.function(
    image=gpu_image,
    gpu="A10",
    cpu=8,
    memory=49152,
    volumes={"/vol/model": model_volume},
    timeout=60 * 30,
)
def compute_floor() -> dict:
    import sys

    sys.path.insert(0, "/root")
    from mosaicvm.bench.compute_floor import run

    out = run()
    _write_artifact("compute_floor.json", out)
    return json.loads(json.dumps(out, default=str))


@app.function(
    image=gpu_image,
    gpu="A10",
    cpu=4,
    memory=32768,
    volumes={"/vol/model": model_volume},
    timeout=60 * 20,
)
def dsa_proof() -> dict:
    import sys

    sys.path.insert(0, "/root")
    import torch
    from mosaicvm.kernels.dsa_attention import dsa_decode_attention, reference_topk_attention

    torch.manual_seed(0)
    H, d, ctx, topk = 64, 128, 4096, 256
    q = torch.randn(H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(ctx, H, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(ctx, H, d, dtype=torch.bfloat16, device="cuda")
    out_ref, idx_ref = reference_topk_attention(q, k, v, topk)
    out, idx = dsa_decode_attention(q, k, v, topk)
    idx_match = bool(torch.equal(idx.sort(dim=-1).values, idx_ref.sort(dim=-1).values.long()))
    err = (out.float() - out_ref.float()).abs().max().item()
    print(f"parity max abs err={err:.4e} idx_match={idx_match}")

    # 32k-ctx decode: workspace is H x topk, not H x ctx — must fit in 24 GB
    ctx32 = 32768
    k32 = torch.randn(ctx32, H, d, dtype=torch.bfloat16, device="cuda")
    v32 = torch.randn(ctx32, H, d, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    o32, _ = dsa_decode_attention(q, k32, v32, topk)
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    print(f"32k-ctx decode {ms:.2f} ms, peak allocated {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    result = {
        "max_abs_err": err,
        "idx_match": idx_match,
        "pass": bool(err < 2e-2 and idx_match),
        "ctx32k_ms": ms,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    _write_artifact("dsa_proof.json", result)
    return result


from mosaicvm.bench.prompts import GOLDEN_PROMPTS



@app.function(
    image=gpu_image,
    gpu="H100:4",
    cpu=16,
    memory=131072,
    volumes={"/vol/model": model_volume},
    timeout=60 * 60 * 3,
)
def golden_reference() -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    _orig = FineGrainedFP8HfQuantizer.update_tp_plan

    def _safe(self, config):
        try:
            return _orig(self, config)
        except Exception as e:
            print("update_tp_plan skipped:", type(e).__name__, e)
            if getattr(config, "base_model_tp_plan", None) is None:
                config.base_model_tp_plan = {}
            return config

    FineGrainedFP8HfQuantizer.update_tp_plan = _safe

    model_dir = "/vol/model/hub"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(config, "base_model_tp_plan", None) is None:
        config.base_model_tp_plan = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    records = []
    for p in GOLDEN_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=32, do_sample=False)
            logits = model(out).logits[:, -1].float().cpu()
        records.append({
            "prompt": p,
            "token_ids": out[0].tolist(),
            "final_logits": logits[0].tolist(),
        })
        print(p, "->", tok.decode(out[0], skip_special_tokens=True)[:120])
    payload = {"records": records}
    os.makedirs("/vol/model/validation", exist_ok=True)
    with open("/vol/model/validation/golden.json", "w") as f:
        json.dump(payload, f)
    model_volume.commit()
    print("wrote /vol/model/validation/golden.json")
    return {"n_prompts": len(records)}


@app.function(
    image=gpu_image,
    gpu="A10",
    cpu=8,
    memory=49152,
    volumes={"/vol/model": model_volume},
    timeout=60 * 60 * 3,
)
def v4_forward_check() -> dict:
    import sys

    sys.path.insert(0, "/root")
    import torch
    from mosaicvm.runtime.memory_budget import budget
    from mosaicvm.runtime.streaming_model import load_streaming_model

    golden_path = "/vol/model/validation/golden.json"
    if not os.path.exists(golden_path):
        raise FileNotFoundError("run golden_reference first")
    with open(golden_path) as f:
        golden = json.load(f)

    model, loader, missing = load_streaming_model("/vol/model/hub", device="cuda")
    print(f"weight-map misses: {len(missing)}")
    for m in missing[:20]:
        print("  miss", m)
    b = budget()
    matches, cosines, notes = [], [], []
    for rec in golden["records"]:
        prompt_ids = rec["token_ids"][:-32]
        ids = torch.tensor([prompt_ids], device="cuda")
        tok_match = 0.0
        cos = 0.0
        note = ""
        try:
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=4)
                logits = model(gen).float()[:, -1]
            gold_ids = rec["token_ids"]
            got_ids = gen[0].tolist()
            n = min(len(gold_ids), len(got_ids))
            tok_match = sum(int(a == b) for a, b in zip(gold_ids[:n], got_ids[:n])) / max(n, 1)
            gold_logits = torch.tensor(rec["final_logits"], device="cuda")
            mm = min(gold_logits.numel(), logits.numel())
            cos = torch.nn.functional.cosine_similarity(gold_logits[:mm], logits[0, :mm], dim=0).item()
        except Exception as e:
            note = f"{type(e).__name__}: {str(e)[:160]}"
            print("prompt failed:", note)
        matches.append(tok_match)
        cosines.append(cos)
        notes.append(note)
        print(f"match={tok_match:.3f} cosine={cos:.4f} {note}")

    mean_match = sum(matches) / len(matches)
    mean_cos = sum(cosines) / len(cosines)
    result = {
        "mean_token_match": mean_match,
        "mean_logit_cosine": mean_cos,
        "pass": bool(mean_match >= 0.95 and mean_cos >= 0.99),
        "n_weight_misses": len(missing),
        "weight_misses_head": missing[:40],
        "budget": b.peak_summary(),
        "per_prompt": [{"match": m, "cosine": c} for m, c in zip(matches, cosines)],
    }
    _write_artifact("v4_forward_check.json", result)
    print(json.dumps({k: result[k] for k in ("mean_token_match", "mean_logit_cosine", "pass", "budget")}, indent=2))
    return result


@app.function(
    image=gpu_image,
    gpu="H100:4",
    cpu=16,
    memory=131072,
    volumes={"/vol/model": model_volume},
    timeout=60 * 60 * 3,
)
def capture_routes(max_new_tokens: int = 64) -> dict:
    """Run V4 on Modal and capture per-layer top-6 routes for consecutive decode positions.

    Hooks every routed ``ffn.gate``; keeps only T==1 (decode) calls so each
    layer records an ordered (ids, weights) sequence. Writes the payload to
    the volume and returns it for local persistence.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    from mosaicvm.bench.prompts import ROUTE_CAPTURE_PROMPTS

    _orig = FineGrainedFP8HfQuantizer.update_tp_plan

    def _safe(self, config):
        try:
            return _orig(self, config)
        except Exception as e:
            print("update_tp_plan skipped:", type(e).__name__, e)
            if getattr(config, "base_model_tp_plan", None) is None:
                config.base_model_tp_plan = {}
            return config

    FineGrainedFP8HfQuantizer.update_tp_plan = _safe

    model_dir = "/vol/model/hub"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(config, "base_model_tp_plan", None) is None:
        config.base_model_tp_plan = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device

    # --- exact route capture ------------------------------------------------
    # The FP8 fused MoE path bypasses ``mlp.gate``'s module forward, so output
    # hooks see nothing. Pre-hook the MoE block instead (its input IS the
    # router input) and recompute the exact noaux_tc route from the checkpoint
    # gate weight + bias with the V4 semantics (sqrtsoftplus scores; bias
    # affects selection only; norm_topk_prob; routed_scaling_factor) — the same
    # math as mosaicvm.model.layers.Router.forward.
    routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)
    n_experts_per_tok = int(getattr(config, "n_experts_per_tok", None)
                            or getattr(config, "num_experts_per_tok", 6))
    captures = {}  # gate_name -> {"ids": [[k], ...], "weights": [[k], ...]}
    gate_order = []
    hooks = []

    moe_gates = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp") and hasattr(mod, "gate"):
            g = mod.gate
            w = getattr(g, "weight", None)
            if w is None or not torch.is_tensor(w) or w.dim() != 2:
                continue
            b = getattr(g, "e_score_correction_bias", None)
            moe_gates.append((name + ".gate", mod, w, b))

    def _make_prehook(gate_name, w, b):
        wf = w.detach().float()
        bf = b.detach().float() if b is not None else None

        def prehook(mod, args, kwargs):
            try:
                x = args[0] if args else kwargs.get("hidden_states")
                if x is None or not torch.is_tensor(x):
                    return
                if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1:
                    return  # keep only decode steps (B=1, T=1)
                logits = torch.nn.functional.linear(x[0, 0].float(), wf)
                orig = torch.nn.functional.softplus(logits).sqrt()
                scores = orig if bf is None else orig + bf
                idx = torch.topk(scores, n_experts_per_tok, dim=-1).indices
                wts = orig.gather(-1, idx)
                wts = wts / wts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                wts = wts * routed_scaling_factor
                rec = captures.setdefault(gate_name, {"ids": [], "weights": []})
                rec["ids"].append(idx.cpu().tolist())
                rec["weights"].append([round(float(v), 6) for v in wts.cpu().tolist()])
            except Exception as e:
                print(f"prehook {gate_name} failed: {type(e).__name__}: {e}")
        return prehook

    for gate_name, mod, w, b in moe_gates:
        gate_order.append(gate_name)
        hooks.append(mod.register_forward_pre_hook(_make_prehook(gate_name, w, b), with_kwargs=True))
    print(f"hooked {len(hooks)} MoE gates (routed_scaling_factor={routed_scaling_factor}, "
          f"top_k={n_experts_per_tok}): {gate_order[:3]} ...")

    out_captures = []
    for pi, (category, p) in enumerate(ROUTE_CAPTURE_PROMPTS):
        for rec in captures.values():
            rec["ids"].clear()
            rec["weights"].clear()
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        layers = []
        for name in gate_order:
            rec = captures.get(name)
            if rec and rec["ids"]:
                layers.append({
                    "layer_name": name,
                    "expert_ids": rec["ids"],
                    "weights": rec["weights"],
                })
        n_pos = max((len(l["expert_ids"]) for l in layers), default=0)
        out_captures.append({
            "prompt_idx": pi,
            "category": category,
            "prompt": p,
            "n_decode_positions": n_pos,
            "layers": layers,
        })
        print(f"[{category}] {p[:48]} -> {n_pos} decode positions, {len(layers)} routed layers")

    for h in hooks:
        h.remove()

    # floor_b-compatible windows: one per (capture, layer)
    windows = []
    for c in out_captures:
        for l in c["layers"]:
            windows.append({
                "prompt_idx": c["prompt_idx"],
                "category": c["category"],
                "layer_name": l["layer_name"],
                "expert_ids": l["expert_ids"],
                "weights": l["weights"],
            })

    payload = {
        "source": "modal_v4_capture",
        "model": MODEL_REPO,
        "max_new_tokens": max_new_tokens,
        "top_k": 6,
        "n_prompts": len(out_captures),
        "captures": out_captures,
        "windows": windows,
    }
    _write_artifact("golden_routes_raw.json", payload)
    return json.loads(json.dumps(payload, default=str))


@app.local_entrypoint()
def capture_routes_local(out: str = "artifacts/golden_routes.json", max_new_tokens: int = 64):
    """Dispatch capture_routes on Modal and persist the payload locally."""
    from pathlib import Path

    payload = capture_routes.remote(max_new_tokens=max_new_tokens)
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}: {payload['n_prompts']} prompts, {len(payload['windows'])} windows")


@app.function(
    image=gpu_image,
    gpu="H100:4",
    cpu=16,
    memory=131072,
    volumes={"/vol/model": model_volume},
    timeout=60 * 60 * 3,
)
def topk_quality_sweep() -> dict:
    """Stock transformers on H100:4; mean token-id match vs k=6 golden."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    from mosaicvm.bench.prompts import GOLDEN_PROMPTS
    from mosaicvm.bench.topk_quality import SWEEP

    golden_path = "/vol/model/validation/golden.json"
    with open(golden_path) as f:
        golden = json.load(f)
    gold_ids = {r["prompt"]: r["token_ids"] for r in golden["records"]}

    _orig = FineGrainedFP8HfQuantizer.update_tp_plan

    def _safe(self, config):
        try:
            return _orig(self, config)
        except Exception as e:
            print("update_tp_plan skipped:", type(e).__name__, e)
            if getattr(config, "base_model_tp_plan", None) is None:
                config.base_model_tp_plan = {}
            return config

    FineGrainedFP8HfQuantizer.update_tp_plan = _safe

    model_dir = "/vol/model/hub"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rows = []
    per_k = {}
    for k in SWEEP:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if getattr(config, "base_model_tp_plan", None) is None:
            config.base_model_tp_plan = {}
        config.num_experts_per_tok = k
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        device = next(model.parameters()).device
        matches = []
        prompt_rows = []
        for p in GOLDEN_PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=32, do_sample=False)
            got = out[0].tolist()
            want = gold_ids[p]
            n = min(len(got), len(want))
            match = sum(int(got[i] == want[i]) for i in range(n)) / max(n, 1)
            matches.append(match)
            prompt_rows.append({"prompt": p, "token_match": match})
        mean = float(sum(matches) / len(matches))
        per_k[str(k)] = {"mean_token_match": mean, "prompts": prompt_rows}
        rows.append({"k": k, "mean_token_match": mean})
        print("k", k, "mean", mean)
        del model
        torch.cuda.empty_cache()

    turbo = 6
    if per_k.get("4", {}).get("mean_token_match", 0) >= 0.90:
        turbo = 4
    elif per_k.get("5", {}).get("mean_token_match", 0) >= 0.90:
        turbo = 5
    payload = {
        "status": "ok",
        "per_k": per_k,
        "rows": rows,
        "turbo_default": turbo,
        "fidelity_k": 6,
        "pass_criterion": "mean_token_match >= 0.90 vs k=6 golden",
    }
    os.makedirs("/vol/model/artifacts", exist_ok=True)
    with open("/vol/model/artifacts/topk_quality.json", "w") as f:
        json.dump(payload, f, indent=2)
    model_volume.commit()
    return payload


@app.function(
    image=gpu_image,
    gpu="A10",
    cpu=8,
    memory=49152,
    volumes={"/vol/model": model_volume},
    timeout=60 * 30,
)
def resident_core_check() -> dict:
    import sys
    sys.path.insert(0, "/root")
    from mosaicvm.runtime.resident_core import resident_core_check as _check

    result = _check("/vol/model/hub", device="cuda")
    _write_artifact("resident_core_check.json", result)
    return result


@app.function(
    image=download_image,
    cpu=2,
    memory=4096,
    volumes={"/vol/model": model_volume},
    timeout=120,
)
def phase01_report() -> dict:
    art = "/vol/model/artifacts"
    names = ["hw_profile_A10.json", "compute_floor.json", "dsa_proof.json", "v4_forward_check.json"]
    out = {}
    for n in names:
        p = os.path.join(art, n)
        if os.path.exists(p):
            with open(p) as f:
                out[n] = json.load(f)
        else:
            out[n] = None
            print("MISSING", n)
    print("=== Phase 0/1 go/no-go ===")
    cf = out.get("compute_floor.json") or {}
    if cf:
        print("compute floor:", json.dumps(cf.get("gate_eval"), indent=2))
        print("rows:", json.dumps(cf.get("projection_rows"), indent=2, default=str)[:2000])
    dsa = out.get("dsa_proof.json") or {}
    print("dsa:", json.dumps(dsa, indent=2, default=str)[:800])
    v4 = out.get("v4_forward_check.json") or {}
    print("v4:", json.dumps({k: v4[k] for k in v4 if k in ("pass", "mean_token_match", "mean_logit_cosine", "budget", "n_weight_misses")}, indent=2, default=str))
    return {k: (v is not None) for k, v in out.items()}


@app.local_entrypoint()
def main():
    print("entrypoints: download_model, hw_profile, compute_floor, dsa_proof, golden_reference, v4_forward_check, topk_quality_sweep, resident_core_check, phase01_report")


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=196608,
    volumes={"/vol/model": model_volume},
    timeout=60 * 45,
)
def ref_hiddens() -> dict:
    """§64 localization: per-layer reference hiddens for the failing
    prompts (math_reasoning, long_generation) from the authoritative
    transformers runtime. Saves to the volume artifacts."""
    import json
    import time
    from pathlib import Path

    import torch

    t0 = time.time()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = "/vol/model/hub"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "70GiB", "cpu": "170GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

    corpus = json.loads(
        (Path(model_dir).parent / "validation"
         / "routes_overlap_corpus_3060.json").read_text()
        if (Path(model_dir).parent / "validation"
           / "routes_overlap_corpus_3060.json").exists() else "null")
    if corpus is None:
        # fall back to inline prompts from the local corpus
        corpus = {"prompts": [
            {"category": "math_reasoning",
             "prompt": "What is 17 * 23? "
                       "Please reason step by step and give the answer.",
             "expected_generated_ids": []},
            {"category": "long_generation",
             "prompt": "Write a detailed essay about the history "
                       "of computing.",
             "expected_generated_ids": []},
        ]}
    targets = [p for p in corpus["prompts"]
               if p["category"] in ("math_reasoning", "long_generation")]

    for p in targets:
        cat = p["category"]
        ids = tok(p["prompt"], return_tensors="pt").input_ids.cuda()
        expected0 = (p["expected_generated_ids"] or [None])[0]
        caps = {}

        def mk(name):
            def h(mod, args, output):
                t = output[0] if isinstance(output, tuple) else output
                caps[name] = t.detach().float().cpu().half()
            return h

        handles = [model.model.embed_tokens.register_forward_hook(
            mk("embed"))]
        for i, lyr in enumerate(model.model.layers):
            handles.append(lyr.register_forward_hook(mk(f"layer{i}")))
        handles.append(model.model.norm.register_forward_hook(mk("norm")))
        with torch.no_grad():
            out = model(ids)
        for h in handles:
            h.remove()
        greedy0 = out.logits[0, -1].argmax().item()
        topv, topi = out.logits[0, -1].float().topk(5)
        print(f"[{cat}] S={ids.shape[1]} greedy0={greedy0} "
              f"expected0={expected0}", flush=True)
        save = {"category": cat, "S": ids.shape[1],
                "greedy0": greedy0, "expected0": expected0,
                "top5_ids": topi.tolist(), "top5_vals": topv.tolist(),
                "input_ids": ids[0].tolist(),
                "prompt": p["prompt"], **caps}
        out_path = f"/vol/model/artifacts/ref_hiddens_{cat}.pt"
        torch.save(save, out_path)
        print(f"[{cat}] saved -> {out_path}", flush=True)
    model_volume.commit()
    return {"status": "done", "elapsed_s": time.time() - t0}
