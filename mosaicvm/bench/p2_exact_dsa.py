"""§84 NEXT 1 — post-DSA P2 rerun → P2_COREBLOCK_EXACT_DSA (reconstructed).

Co-residency verdict first: banks (4.4 GB) + core-no-head (7.0 GB) +
exact-DSA weights (1.95 GB fp32 compressor/indexer math) = 13.4 GB > 11.6 GB
usable — the fully interleaved exact-DSA block CANNOT reside on the 12-GB
card beside Mixed-A. Per §60 this is therefore P2_COREBLOCK_EXACT_DSA with
explicit omission accounting (the same pattern the original P2 used for the
LM head):

  Phase A (eager baseline): mosaicvm/bench/true_p2_mixed.py rerun in the
      same session/thermal window — banks + core + eager attention.
  Phase B (attention-swap delta, THIS SCRIPT): per-layer
      compute_layer_fused with (a) the eager stand-in and (b) the exact-DSA
      core via ExactAttention.attention_core — identical FP8 projections,
      wo_a/wo_b path, shared experts, gate and HC in both; ONLY the
      attention mechanism differs. Measured WITHOUT expert banks resident
      (core 7.0 GB + DSA weights 1.95 GB fits; MoE cost is identical in
      both configurations and is carried by phase A).
  Phase C (reconstruction): T_exact(K) = T_block_eager(K) + Σ_L Δ_L(K);
      R_oracle = E[A] / (T_exact + t_head); §62 band classification.

  Validation anchor: Σ_L t_eager(L) must reproduce phase A's Δ_core
      (= T_block − MoE_ref) within a few percent — that checks the
      per-layer decomposition is faithful before the delta is trusted.

  DSA state: per layer, 4096-token context populated by real chunked
  prefill (8×512, untimed), then timed K-token verification chunks at
  advancing positions (consumptive state — positions never replay).
  Thermal: warmup_steady + snapshot labeling, as the original P2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

sys.path.insert(0, ".")
from mosaicvm.bench.floors import HIDDEN, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.fp8_core import (
    CORE_PATH, FP8_TENSORS, HEAD_DIM, N_HEADS, O_GROUPS, O_LORA, Q_LORA,
    fp8_linear,
)
from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention

K_VALUES = (56, 80, 112)
CTX = 4096                    # verification context (roadmap §58: 4K)
MAXSEQ = 8192                 # ctx + all timed chunks
OUT = Path("artifacts/p2_exact_dsa.json")
BASE = Path("artifacts/true_p2_mixed.json")
ORACLE = Path("artifacts/branchpack_oracle.json")
MOE_REF = {"56": 0.8379, "80": 1.0046, "112": 1.1959}  # prodfit_frontier


def load_core_no_head(f) -> dict:
    core = {"layers": []}
    core["norm_w"] = f.get_tensor("norm.weight").to("cuda")
    core["hc_head_scale"] = f.get_tensor("hc_head_scale").to("cuda")
    for layer in range(N_LAYERS):
        pre = f"layers.{layer}."
        lp = {"fp8": {}, "scales": {}, "small": {}}
        for name in FP8_TENSORS:
            lp["fp8"][name] = f.get_tensor(pre + name + ".weight").to("cuda")
            lp["scales"][name] = f.get_tensor(pre + name + ".scale").to("cuda")
        lp["small"]["gate_w"] = f.get_tensor(pre + "ffn.gate.weight").to(torch.bfloat16).to("cuda")
        bias_name = pre + "ffn.gate.bias"
        gb = f.get_tensor(bias_name) if bias_name in f.keys() else None
        lp["small"]["gate_bias"] = (gb.to("cuda") if gb is not None
                                    else torch.zeros(256, device="cuda"))
        for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            lp["small"][nm] = f.get_tensor(pre + nm).to("cuda")
        for src, dst in (("attn_norm.weight", "attn_norm"),
                         ("ffn_norm.weight", "ffn_norm"),
                         ("attn.q_norm.weight", "q_norm"),
                         ("attn.kv_norm.weight", "kv_norm"),
                         ("attn.attn_sink", "attn_sink")):
            lp["small"][dst] = f.get_tensor(pre + src).to("cuda")
        core["layers"].append(lp)
    return core


def build_dsa_modules(f, bf16_storage: bool = False,
                      max_seq: int | None = None) -> list:
    """43 exact-DSA attention cores (compressor/indexer/norms/sink only —
    the projection linears are stripped: projections run on the production
    FP8 path inside compute_layer_fused, so the module's own q_a/q_b/
    kv_proj/wo_a/wo_b would be 9 GB of dead weight)."""
    cfg = V4Config()
    mods = []
    for L in range(N_LAYERS):
        # construct/load on CPU, then move: keeps the GPU peak at the final
        # bf16/fp32 weight bytes (no 210-MB random-projection transient per
        # module beside the resident banks+core stack)
        m = ExactAttention(cfg, layer_id=L, max_seq=max_seq or MAXSEQ,
                           device="cpu")
        for proj in ("q_a", "q_b", "kv_proj", "wo_a", "wo_b"):
            delattr(m, proj)
        cast_exact_attention(m, "cpu")
        pre = f"layers.{L}.attn"
        # bf16_storage: checkpoint-bf16-native tensors stored bf16 (fp32 math
        # via lazy widen — bitwise-identical, tests/test_dsa_bf16_storage.py);
        # halves the DSA footprint for the 12-GB residency budget
        bf16 = {"compressor.wkv.weight", "compressor.wgate.weight",
                "compressor.norm.weight", "q_norm.weight", "kv_norm.weight"}
        with torch.no_grad():
            def put(param, t, dt):
                leaf = pre + "."  # unused; storage decided by caller dt
                param.data = t.to(device=param.device, dtype=dt)
            qn_dt = torch.bfloat16 if bf16_storage else torch.float32
            put(m.q_norm.weight, f.get_tensor(f"{pre}.q_norm.weight"), qn_dt)
            put(m.kv_norm.weight, f.get_tensor(f"{pre}.kv_norm.weight"), qn_dt)
            put(m.attn_sink, f.get_tensor(f"{pre}.attn_sink"), torch.float32)
            if m.compressor is not None:
                put(m.compressor.ape, f.get_tensor(f"{pre}.compressor.ape"), torch.float32)
                c_dt = torch.bfloat16 if bf16_storage else torch.float32
                put(m.compressor.wkv.weight,
                    f.get_tensor(f"{pre}.compressor.wkv.weight"), c_dt)
                put(m.compressor.wgate.weight,
                    f.get_tensor(f"{pre}.compressor.wgate.weight"), c_dt)
                put(m.compressor.norm.weight,
                    f.get_tensor(f"{pre}.compressor.norm.weight"), c_dt)
            if m.indexer is not None:
                ix = m.indexer
                from mosaicvm.runtime.fp8 import dequant_fp8
                put(ix.wq_b.weight, dequant_fp8(
                    f.get_tensor(f"{pre}.indexer.wq_b.weight"),
                    f.get_tensor(f"{pre}.indexer.wq_b.scale")), torch.bfloat16)
                put(ix.weights_proj.weight,
                    f.get_tensor(f"{pre}.indexer.weights_proj.weight"), torch.bfloat16)
                isc = ix.compressor
                put(isc.ape, f.get_tensor(f"{pre}.indexer.compressor.ape"), torch.float32)
                put(isc.wkv.weight,
                    f.get_tensor(f"{pre}.indexer.compressor.wkv.weight"), c_dt)
                put(isc.wgate.weight,
                    f.get_tensor(f"{pre}.indexer.compressor.wgate.weight"), c_dt)
                put(isc.norm.weight,
                    f.get_tensor(f"{pre}.indexer.compressor.norm.weight"), c_dt)
        m.to("cuda")
        mods.append(m)
    return mods


def compute_layer_eager(x, lp, k, kv_cache):
    """compute_layer_fused (production FP8 path) — eager attention stand-in.
    Identical to fp8_core.compute_layer_fused."""
    fp8, scales, small = lp["fp8"], lp["scales"], lp["small"]
    q_lora = fp8_linear(x, fp8["attn.wq_a"], scales["attn.wq_a"])
    q_lora = F.rms_norm(q_lora, (Q_LORA,), small["q_norm"])
    q = fp8_linear(q_lora, fp8["attn.wq_b"], scales["attn.wq_b"])
    kv = fp8_linear(x, fp8["attn.wkv"], scales["attn.wkv"])
    kv = F.rms_norm(kv, (HEAD_DIM,), small["kv_norm"])

    q3 = q.view(k, N_HEADS, HEAD_DIM)
    k_exp = kv_cache.expand(kv_cache.shape[0], N_HEADS, HEAD_DIM)
    scores = torch.einsum("khd,chd->kch", q3, k_exp) / (HEAD_DIM ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("kch,chd->khd", attn, k_exp)
    return _tail(x, lp, attn_out, k)


def compute_layer_exact(x, lp, k, dsa: ExactAttention, start_pos: int):
    """Same production FP8 path with the attention mechanism swapped for
    the exact-DSA core (chunked verification at start_pos)."""
    fp8, scales, small = lp["fp8"], lp["scales"], lp["small"]
    q_lora_raw = fp8_linear(x, fp8["attn.wq_a"], scales["attn.wq_a"])
    q_lora = F.rms_norm(q_lora_raw, (Q_LORA,), small["q_norm"])
    q = fp8_linear(q_lora, fp8["attn.wq_b"], scales["attn.wq_b"])
    kv_raw = fp8_linear(x, fp8["attn.wkv"], scales["attn.wkv"])
    o = dsa.attention_core(x, q_lora_raw, q, kv_raw, start_pos)
    return _tail(x, lp, o.view(k, N_HEADS, HEAD_DIM), k)


def _tail(x, lp, attn_out, k):
    """Grouped wo_a + wo_b + shared experts + gate + HC — identical in both
    variants (from compute_layer_fused)."""
    fp8, scales, small = lp["fp8"], lp["scales"], lp["small"]
    attn_grouped = attn_out.reshape(k, O_GROUPS, 4096).transpose(0, 1).contiguous()
    wo_a_out = torch.empty(O_GROUPS, k, O_LORA, dtype=torch.bfloat16, device=x.device)
    s_5 = scales["attn.wo_a"]
    nb = s_5.shape[0] // O_GROUPS
    for g in range(O_GROUPS):
        fp8_linear(attn_grouped[g],
                   fp8["attn.wo_a"].view(O_GROUPS, O_LORA, 4096)[g],
                   s_5.view(O_GROUPS, nb, s_5.shape[1])[g],
                   out=wo_a_out[g])
    wo_a_out = wo_a_out.transpose(0, 1).reshape(k, O_GROUPS * O_LORA).contiguous()
    y = fp8_linear(wo_a_out, fp8["attn.wo_b"], scales["attn.wo_b"])

    g1 = fp8_linear(x, fp8["ffn.shared_experts.w1"],
                    scales["ffn.shared_experts.w1"]).float().clamp(max=10.0)
    u1 = fp8_linear(x, fp8["ffn.shared_experts.w3"],
                    scales["ffn.shared_experts.w3"]).float().clamp(-10.0, 10.0)
    z = (F.silu(g1) * u1).to(torch.bfloat16)
    y = y + fp8_linear(z, fp8["ffn.shared_experts.w2"],
                       scales["ffn.shared_experts.w2"])

    logits = F.linear(x.float(), small["gate_w"].float())
    sr = F.softplus(logits).sqrt() + small["gate_bias"].float()
    torch.topk(sr, 6, dim=-1)

    hc_in = torch.randn(k, 4 * HIDDEN, dtype=torch.float32, device=x.device)
    linf = F.linear
    mixes = linf(hc_in, small["hc_attn_fn"].float())
    torch.sigmoid(mixes * small["hc_attn_scale"].float().mean()
                  + small["hc_attn_base"].float()) + 1e-6
    mixes = linf(hc_in, small["hc_ffn_fn"].float())
    torch.sigmoid(mixes * small["hc_ffn_scale"].float().mean()
                  + small["hc_ffn_base"].float()) + 1e-6

    F.rms_norm(x, (HIDDEN,), small["attn_norm"])
    F.rms_norm(x, (HIDDEN,), small["ffn_norm"])
    return y


def populate_state(dsa: ExactAttention, gen):
    """Chunked prefill of a CTX-token context (8x512, untimed) through the
    projection-free module: attention_core with random projections. State
    VALUES are timing-irrelevant; the paths (prefill + 7 chunks) and cache
    shapes are the real ones."""
    dsa.reset_state()
    CH = 256                                       # smaller prefill chunks:
    x0 = torch.randn(CH, HIDDEN, dtype=torch.bfloat16, device="cuda",
                     generator=gen) * 0.5           # transient headroom
    q_lora = torch.randn(CH, Q_LORA, dtype=torch.bfloat16,
                         device="cuda", generator=gen)
    q = torch.randn(CH, N_HEADS * HEAD_DIM, dtype=torch.bfloat16,
                    device="cuda", generator=gen)
    kv = torch.randn(CH, HEAD_DIM, dtype=torch.bfloat16,
                     device="cuda", generator=gen)
    # no_grad is REQUIRED here: the module's compressor weights are
    # nn.Parameters, so grad-mode calls retain autograd graphs (~70 MB per
    # ratio-4 layer per call — OOMed the 12-GB card before this fix)
    with torch.no_grad():
        dsa.attention_core(x0, q_lora, q, kv, 0)
        for pos in range(CH, CTX, CH):
            dsa.attention_core(x0, q_lora, q, kv, pos)


def time_layer(fn, reps=8, warm=3):
    with torch.no_grad():                    # see populate_state note
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def certify(snaps, start, end):
    temps = [s.get("temp_c") for s in snaps if s.get("temp_c") is not None]
    if not temps:
        return "UNKNOWN"
    return "UNKNOWN"  # temp telemetry unavailable this session (pynvml)


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    base = json.loads(BASE.read_text())
    oracle = json.loads(ORACLE.read_text())["oracle"]

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_no_head(f)
        torch.cuda.synchronize()
        core_gb = torch.cuda.memory_allocated() / 1e9
        print(f"core-no-head: {core_gb:.2f} GB")
        dsa_mods = build_dsa_modules(f)
    torch.cuda.synchronize()
    dsa_gb = torch.cuda.memory_allocated() / 1e9 - core_gb
    print(f"+DSA weights/state: {dsa_gb:.2f} GB "
          f"(total {torch.cuda.memory_allocated()/1e9:.2f} GB, banks-free)")

    warmup_steady(120.0)
    t_start = snapshot()
    snaps = [t_start]
    gen = torch.Generator(device="cuda").manual_seed(9100)

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "config": "P2_COREBLOCK_EXACT_DSA (reconstructed): phase-A eager "
                  "block (banks+core) + per-layer attention-swap delta "
                  "(core+DSA only, banks-free) + head; only the attention "
                  "mechanism differs between variants",
        "core_gb": core_gb, "dsa_weights_state_gb": dsa_gb,
        "co_residency": {
            "banks_plus_core_plus_dsa_fp32_gb": 4.4 + core_gb + 1.95,
            "verdict": "DOES NOT FIT on 12 GB (11.6 usable) — "
                       "P2_COREBLOCK_EXACT_DSA with omission accounting "
                       "(LM head measured separately as in phase A; DSA "
                       "weights require streaming/tiering beside Mixed-A)"},
        "ctx": CTX,
        "rows": {},
        "validation": {},
    }

    for K in K_VALUES:
        kv_cache = torch.randn(1024, 1, HEAD_DIM, dtype=torch.bfloat16,
                               device="cuda")
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        sum_eager = 0.0
        sum_dsa = 0.0
        per_layer = {}
        for L in range(N_LAYERS):
            lp = core["layers"][L]
            dsa = dsa_mods[L]
            t_eager = time_layer(lambda: compute_layer_eager(
                xs[0], lp, K, kv_cache))
            populate_state(dsa, gen)
            pos = [CTX]

            def step():
                p = pos[0]
                compute_layer_exact(xs[pos[0] % 4], lp, K, dsa, p)
                pos[0] += K
            t_dsa = time_layer(step)
            sum_eager += t_eager
            sum_dsa += t_dsa
            per_layer[L] = {"ratio": CFG_RATIO(L),
                            "eager_ms": t_eager * 1e3,
                            "dsa_ms": t_dsa * 1e3,
                            "delta_ms": (t_dsa - t_eager) * 1e3}
        snaps.append(snapshot())
        # validation anchor: Σ eager-layer times vs phase-A Δ_core
        delta_core_A = base["rows"][str(K)]["t_coreblock_s"] - MOE_REF[str(K)]
        sum_delta = sum_dsa - sum_eager
        results["validation"][str(K)] = {
            "sum_eager_layer_s": round(sum_eager, 4),
            "phaseA_delta_core_s": round(delta_core_A, 4),
            "decomposition_match_pct": round(
                100 * sum_eager / delta_core_A, 1),
            "sum_dsa_delta_s": round(sum_delta, 4)}
        # reconstruction
        t_exact = base["rows"][str(K)]["t_coreblock_s"] + sum_delta
        t_head = base["rows"][str(K)]["t_head_s"]
        t_rec = t_exact + t_head
        ea = oracle.get(str(K), {}).get("expected_accepted_depth")
        r_or = ea / t_rec if ea else None
        if r_or and r_or >= 40: cls = "P2 STRONG"
        elif r_or and r_or >= 35: cls = "P2 GOOD"
        elif r_or and r_or >= 33: cls = "P2 VIABLE"
        elif r_or and r_or >= 30: cls = "P2 MINIMUM"
        else: cls = "P2 MISS"
        results["rows"][str(K)] = {
            "t_eager_block_s": base["rows"][str(K)]["t_coreblock_s"],
            "sum_delta_s": round(sum_delta, 4),
            "t_exact_coreblock_s": round(t_exact, 4),
            "t_head_s": t_head, "t_reconstructed_s": round(t_rec, 4),
            "E_A": ea, "R_oracle_per_s": round(r_or, 1) if r_or else None,
            "p2_class": cls,
            "route_auth": base["rows"][str(K)]["route_auth"],
            "per_layer": per_layer,
        }
        print(f"K={K:3d}: ΣΔ_dsa={sum_delta*1e3:6.1f}ms  "
              f"T_exact={t_exact:.4f}s +head={t_head*1e3:.1f}ms  "
              f"R_oracle={r_or and round(r_or,1)}/s [{cls}]  "
              f"(Σeager {sum_eager*1e3:.0f}ms vs phaseA Δcore "
              f"{delta_core_A*1e3:.0f}ms)")
        del kv_cache, xs
        torch.cuda.empty_cache()

    t_end = snapshot()
    results["thermal"] = {"label": certify(snaps, t_start, t_end)}
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"thermal [{results['thermal']['label']}] — wrote {OUT}")


def CFG_RATIO(L):
    from mosaicvm.model.config import V4Config
    return V4Config().compress_ratios[L]


if __name__ == "__main__":
    main()
