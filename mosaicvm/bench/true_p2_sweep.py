"""TRUE P2 (roadmap §42-46) — full source-class verifier-block sweep.

Measures the TRUE-P2 verification block on the RTX 3060 with EVERY
mandatory target component in the production numeric class:

  ExactAttention (via DSA delta reconstruction — cannot co-fit with the
    expert bank on 12 GB; identical reconstruction to the validated
    verifier_floor_sweep)
  source FP8 core (workspace weight-swap architecture)
  real HyperConnections (hc_pre + sinkhorn + hc_post, fp32)
  real router (fp32 gate GEMM + sqrtsoftplus + bias + top-6 +
    normalize + x1.5)
  MXIE routed experts (fused_moe_mxint, packed4 bank, golden routes,
    swiglu_limit=10)
  source-class shared expert (act_quant_ref E4M3 per-128 + bf16 GEMMs
    with fp32 accumulation + source clamp + w2 requant)
  HC head + RMSNorm + fp32 LM head

Phases:
  FULL   core + bank + head resident; full block minus DSA -> T_full(N)
  ABLATE per-component ablations at every N (routed, shared, head, hc,
         router, attn-proxy, swap) -> §45 component table
  B      exact-DSA vs eager, banks-free (validated reconstruction)
         -> T_dsa_delta(N)

  T_true(N) = T_full(N) + T_dsa_delta(N)

Then: fit T(N) = T0 + alpha*N, E_req(N) = 30*T(N), component
decomposition of the intercept, thermal snapshots.

Classification: TRUE_P2_PRODUCTION_NUMERIC_SWEEP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from mosaicvm.bench.floors import HIDDEN, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.fp8_core import (
    CORE_PATH, HEAD_DIM, N_HEADS, Q_LORA, O_GROUPS, O_LORA,
    CTX as KV_CTX, load_core_resident, alloc_workspace, decode_layer,
)
from mosaicvm.bench.phase29r import _allocate_packed4_experts, N_EXPERTS
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.kernels.fp4_source import act_quant_ref
from mosaicvm.bench.wide_verify import _golden_windows
from mosaicvm.bench.wideverify_bootstrap import (
    layer_token_pools, bootstrap_window)
from mosaicvm.bench.p2_exact_dsa import (
    build_dsa_modules, compute_layer_eager, compute_layer_exact,
    populate_state, time_layer, CTX as DSA_CTX,
)

from safetensors.torch import safe_open

N_VALUES = (1, 2, 4, 8, 16, 32, 56, 80, 112)
OUT = Path("artifacts/true_p2_sweep.json")
GOLDEN = Path("artifacts/golden_routes.json")
HC = 4
EPS = 1e-6
IMMA_K = 4   # GU race winner (gu_sweep.json): -14% N16/24, -7.5% N32


def fit_linear(ns, ts):
    n = len(ns)
    sx, sy = sum(ns), sum(ts)
    sxx = sum(x * x for x in ns)
    sxy = sum(x * y for x, y in zip(ns, ts))
    d = n * sxx - sx * sx
    if d == 0:
        return 0.0, ts[0] if ts else 0.0
    a = (n * sxy - sx * sy) / d
    b = (sy - a * sx) / n
    return a, b


# ------------------------------------------------- source-class pieces ----
def hc_split_sinkhorn(mixes, scale, base):
    """Kit hc_split_sinkhorn (20 sinkhorn iters) on [K, 24] mixes."""
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + EPS
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1]
                             + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:].view(-1, HC, HC) * scale[2]
            + base[2 * HC:].view(HC, HC)).softmax(-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    return pre, post, comb


def hc_pre(x_hc, fn, base, scale):
    """[K, HC, D] -> merged [K, D] + (post, comb). fp32 math."""
    shape, dtype = x_hc.size(), x_hc.dtype
    xf = x_hc.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = torch.nn.functional.linear(xf, fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base)
    y = torch.sum(pre.unsqueeze(-1) * x_hc.view(shape), dim=1)
    return y.to(dtype), post, comb


def hc_post(x, residual, post, comb):
    """[K, D] + [K, HC, D] -> [K, HC, D]."""
    y = post.unsqueeze(-1) * x.unsqueeze(1) \
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(1), dim=2)
    return y.type_as(x)


def router_source(x, gate_w, gate_bias):
    """Full source router: fp32 gate GEMM, sqrtsoftplus, bias-shifted
    top-6, normalize, x1.5. Returns (weights, idx)."""
    logits = torch.nn.functional.linear(x.float(), gate_w.float())
    original = torch.nn.functional.softplus(logits).sqrt()
    scores = original + gate_bias.float()
    idx = torch.topk(scores, 6, dim=-1).indices
    w = original.gather(-1, idx)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return w * 1.5, idx


def shared_source(x, ws):
    """Source-class shared expert: dynamic E4M3 per-128 act quant on x
    and on the SwiGLU input to w2; bf16 GEMMs (fp32 accumulation);
    source clamp."""
    k = x.shape[0]

    def _dq(t):
        q, s = act_quant_ref(t, 128)
        n = t.shape[-1]
        return (q.float().view(k, n // 128, 128) * s.unsqueeze(-1)) \
            .reshape(k, n).to(torch.bfloat16)

    xd = _dq(x)
    g = (xd @ ws["ffn.shared_experts.w1"].t()).float().clamp(max=10.0)
    u = (xd @ ws["ffn.shared_experts.w3"].t()).float().clamp(-10.0, 10.0)
    z = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
    zd = _dq(z)
    return zd @ ws["ffn.shared_experts.w2"].t()


def attn_proxy(x, ws, small, k, kv_cache):
    """Workspace-core attention stand-in (identical op sequence to
    fp8_core.compute_layer's attention part). The exact-DSA delta
    replaces this cost in reconstruction."""
    lin = torch.nn.functional.linear
    q_lora = lin(x, ws["attn.wq_a"])
    q_lora = torch.nn.functional.rms_norm(
        q_lora, (Q_LORA,), small["q_norm"])
    q = lin(q_lora, ws["attn.wq_b"])
    kv = lin(x, ws["attn.wkv"])
    kv = torch.nn.functional.rms_norm(kv, (HEAD_DIM,), small["kv_norm"])
    q3 = q.view(k, N_HEADS, HEAD_DIM)
    k_exp = kv_cache.expand(kv_cache.shape[0], N_HEADS, HEAD_DIM)
    scores = torch.einsum("khd,chd->kch", q3, k_exp) / (HEAD_DIM ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("kch,chd->khd", attn, k_exp)
    wo_a_w = ws["attn.wo_a"].view(O_GROUPS, O_LORA, 4096)
    wo_a_out = torch.einsum(
        "kgd,grd->kgr", attn_out.reshape(k, O_GROUPS, 4096), wo_a_w)
    return lin(wo_a_out.reshape(k, O_GROUPS * O_LORA), ws["attn.wo_b"])


def hc_head_apply(x_hc, fn, base, scale):
    shape, dtype = x_hc.size(), x_hc.dtype
    xf = x_hc.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = torch.nn.functional.linear(xf, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + EPS
    return torch.sum(pre.unsqueeze(-1) * x_hc.view(shape), dim=1).to(dtype)


# ------------------------------------------------------------- block -------
def make_block(core, ws, bank, K, routes, head_w, hc_head, ablate=(),
               bank_e=256):
    """Full TRUE-P2 block (minus DSA). ablate: component names to stub."""
    x_hc = torch.randn(K, HC, HIDDEN, dtype=torch.bfloat16,
                       device="cuda")
    kv_cache = torch.randn(KV_CTX, 1, HEAD_DIM, dtype=torch.bfloat16,
                           device="cuda")
    state = {"i": 0}

    def step():
        nonlocal x_hc
        i = state["i"]
        for L in range(N_LAYERS):
            lp = core["layers"][L]
            small = lp["small"]
            if "swap" not in ablate:
                decode_layer(lp["fp8"], lp["scales"], ws)

            # ---- attention half ----
            residual = x_hc
            if "hc" in ablate:
                x = x_hc.mean(1)
            else:
                x, post, comb = hc_pre(
                    x_hc, small["hc_attn_fn"].float(),
                    small["hc_attn_base"].float(),
                    small["hc_attn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["attn_norm"])
            if "attn" not in ablate:
                a = attn_proxy(x, ws, small, K, kv_cache)
            else:
                a = x
            if "hc" in ablate:
                x_hc = x_hc + a.unsqueeze(1)
            else:
                x_hc = hc_post(a, residual, post, comb)

            # ---- ffn half ----
            residual = x_hc
            if "hc" in ablate:
                x = x_hc.mean(1)
            else:
                x, post, comb = hc_pre(
                    x_hc, small["hc_ffn_fn"].float(),
                    small["hc_ffn_base"].float(),
                    small["hc_ffn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["ffn_norm"])
            if "router" not in ablate:
                router_source(x, small["gate_w"], small["gate_bias"])
            y = torch.zeros_like(x)
            if "routed" not in ablate:
                r_idx, r_w = routes[(i * N_LAYERS + L) % len(routes)]
                y = y + fused_moe_mxint(
                    x, r_idx % bank_e, r_w, bank=bank,
                    weight_repr="packed4", crossover=0, imma_k=IMMA_K,
                    swiglu_limit=10.0)
            if "shared" not in ablate:
                y = y + shared_source(x, ws)
            if "hc" in ablate:
                x_hc = x_hc + y.unsqueeze(1)
            else:
                x_hc = hc_post(y, residual, post, comb)
        if "head" not in ablate:
            yh = hc_head_apply(
                x_hc, hc_head["fn"], hc_head["base"], hc_head["scale"])
            yn = torch.nn.functional.rms_norm(
                yh.float(), (HIDDEN,), core["norm_w"].float())
            torch.nn.functional.linear(yn.to(torch.bfloat16), head_w)
        state["i"] += 1

    return step


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    t_start = snapshot()
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "classification": "TRUE_P2_PRODUCTION_NUMERIC_SWEEP (imma_k=4 GU)",
        "components": "ExactAttention (DSA delta reconstruction), "
                      "source FP8 core, real HC+sinkhorn, real router, "
                      "MXIE routed experts (packed4, swiglu_limit=10), "
                      "source-class shared expert (E4M3 per-128 act "
                      "quant, bf16 GEMM fp32-accum), fp32 LM head",
        "n_values": list(N_VALUES),
        "phases": {},
        "components_by_n": {},
        "rows": {},
        "vram": {},
    }
    warmup_steady(120.0)

    golden = json.loads(GOLDEN.read_text())

    # bank FIRST on an empty GPU (packing transients), then core.
    # 128-expert bank (half): identical GEMM shapes/timing per selected
    # expert; frees 1.7 GB for the real-HC fp32 intermediates that the
    # 12 GB display GPU cannot hold alongside the full 256 bank.
    BANK_E = N_EXPERTS // 2
    bank = _allocate_packed4_experts(BANK_E)
    torch.cuda.empty_cache()
    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_resident(f)
        hc_head = {
            "fn": f.get_tensor("hc_head_fn").float().to("cuda"),
            "base": f.get_tensor("hc_head_base").float().to("cuda"),
            "scale": f.get_tensor("hc_head_scale").float().to("cuda"),
        }
        # bf16-resident head (checkpoint stores bf16; bf16 GEMM has
        # exact same products with fp32 accumulation — 1 GB not 2.1)
        head_w = f.get_tensor("head.weight").to("cuda")
    ws0 = alloc_workspace("cuda")
    torch.cuda.synchronize()
    results["vram"]["full_resident_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)
    print(f"full resident: {results['vram']['full_resident_gb']} GB")

    pools = layer_token_pools(golden)
    gen_r = torch.Generator().manual_seed(909)

    def build_routes(K):
        wins = _golden_windows(K, golden)
        if len(wins) >= 16 * N_LAYERS:
            step_ = max(1, len(wins) // (16 * N_LAYERS))
            routes = [
                (torch.tensor(w[0], device="cuda", dtype=torch.int64),
                 torch.tensor(w[1], device="cuda", dtype=torch.float32))
                for w in wins[::step_][:16 * N_LAYERS]]
            auth = "real_golden"
        else:
            routes = [
                (lambda p: (p[0].to("cuda"), p[1].to("cuda")))(
                    bootstrap_window(pools[L % N_LAYERS], K, gen_r))
                for L in range(16 * N_LAYERS)]
            auth = "per_layer_bootstrap"
        return routes, auth

    # ================= FULL phase =================
    t_full = {}
    for K in N_VALUES:
        routes, auth = build_routes(K)
        step = make_block(core, ws0, bank, K, routes, head_w, hc_head,
                          bank_e=BANK_E)
        t = time_layer(step, reps=8, warm=3)
        t_full[K] = t
        print(f"  FULL N={K:3d} [{auth}]: {t:.4f}s "
              f"({K / t:.1f} raw tok/s)")
        del routes
        torch.cuda.empty_cache()
    results["phases"]["T_full_block"] = {str(k): round(v, 4)
                                         for k, v in t_full.items()}

    # ================= component ablations =================
    ABLATIONS = ("routed", "shared", "head", "hc", "router", "attn",
                 "swap")
    comp = {a: {} for a in ABLATIONS}
    for K in N_VALUES:
        routes, _ = build_routes(K)
        base_t = t_full[K]
        row = {}
        for a in ABLATIONS:
            step = make_block(core, ws0, bank, K, routes, head_w,
                              hc_head, ablate=(a,), bank_e=BANK_E)
            t = time_layer(step, reps=5, warm=2)
            comp[a][K] = base_t - t
            row[a] = round(base_t - t, 4)
        # swap-only floor (everything else stubbed)
        step = make_block(core, ws0, bank, K, routes, head_w, hc_head,
                          ablate=("routed", "shared", "head", "hc",
                                  "router", "attn"), bank_e=BANK_E)
        t_swap = time_layer(step, reps=5, warm=2)
        row["swap_only_floor"] = round(t_swap, 4)
        results["components_by_n"][str(K)] = row
        print(f"  ablations N={K:3d}: " + " ".join(
            f"{a}={row[a]:.3f}" for a in ABLATIONS))
        del routes
        torch.cuda.empty_cache()
    for a in ABLATIONS:
        results["phases"][f"component_{a}"] = {
            str(k): round(v, 4) for k, v in comp[a].items()}

    # ================= phase B: DSA (banks-free) =================
    step = None          # drop the last make_block closure (holds bank)
    import gc
    del bank, ws0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    results["vram"]["after_bank_free_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)

    with safe_open(str(CORE_PATH), framework="pt") as f:
        dsa_mods = build_dsa_modules(f)
    gen = torch.Generator(device="cuda").manual_seed(9100)
    for dsa in dsa_mods:
        populate_state(dsa, gen)
    torch.cuda.synchronize()
    results["vram"]["after_dsa_load_gb"] = round(
        torch.cuda.memory_allocated() / 1e9, 2)

    phase_be, phase_bd = {}, {}
    for K in N_VALUES:
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16,
                          device="cuda") for _ in range(4)]
        kv_cache = torch.randn(KV_CTX, 1, HEAD_DIM,
                               dtype=torch.bfloat16, device="cuda")

        def eager_block():
            for L in range(N_LAYERS):
                compute_layer_eager(xs[0], core["layers"][L], K, kv_cache)

        pos = [DSA_CTX]

        def exact_block():
            p = pos[0]
            for L in range(N_LAYERS):
                compute_layer_exact(xs[p % 4], core["layers"][L], K,
                                    dsa_mods[L], p)
            pos[0] += K

        t_be = time_layer(eager_block, reps=8, warm=3)
        t_bd = time_layer(exact_block, reps=8, warm=3)
        phase_be[K], phase_bd[K] = t_be, t_bd
        print(f"  phase B N={K:3d}: eager={t_be:.4f}s "
              f"exact={t_bd:.4f}s "
              f"(Δ_dsa={((t_bd - t_be) * 1e3):+.1f} ms)")
        del xs, kv_cache
        torch.cuda.empty_cache()
    results["phases"]["B_eager_banks_free"] = {
        str(k): round(v, 4) for k, v in phase_be.items()}
    results["phases"]["B_exact_dsa_banks_free"] = {
        str(k): round(v, 4) for k, v in phase_bd.items()}

    # ================= reconstruction =================
    for K in N_VALUES:
        t_true = t_full[K] + phase_bd[K] - phase_be[K]
        row = results["components_by_n"][str(K)]
        results["rows"][str(K)] = {
            "T_full_block_s": round(t_full[K], 4),
            "T_dsa_delta_s": round(phase_bd[K] - phase_be[K], 4),
            "T_true_s": round(t_true, 4),
            "true_verify_tok_s": round(K / t_true, 1),
            "E_req_for_30": round(30.0 * t_true, 1),
            "components": row,
        }

    ns = [float(x) for x in N_VALUES]
    t_true_list = [t_full[k] + phase_bd[k] - phase_be[k]
                   for k in N_VALUES]
    a_v, b_v = fit_linear(ns, t_true_list)
    a_f, b_f = fit_linear(ns, [t_full[k] for k in N_VALUES])
    a_d, b_d = fit_linear(ns, [phase_bd[k] - phase_be[k]
                               for k in N_VALUES])
    results["fits"] = {
        "T_true": {"T0_s": round(b_v, 4),
                   "alpha_s_per_tok": round(a_v, 5)},
        "T_full_block": {"T0_s": round(b_f, 4),
                         "alpha_s_per_tok": round(a_f, 5)},
        "T_dsa_delta": {"T0_s": round(b_d, 4),
                        "alpha_s_per_tok": round(a_d, 5)},
    }
    # intercept decomposition (fit-level): swap floor + components
    a_s, b_s = fit_linear(ns, [results["components_by_n"][str(k)]
                               ["swap_only_floor"] for k in N_VALUES])
    comp_t0 = {}
    for a in ABLATIONS:
        ca, cb = fit_linear(
            ns, [comp[a][k] for k in N_VALUES])
        comp_t0[a] = round(cb, 4)
    results["intercept_decomposition"] = {
        "T0_true": round(b_v, 4),
        "T0_swap_only_floor": round(b_s, 4),
        "T0_components": comp_t0,
        "T0_dsa_delta": round(b_d, 4),
    }

    t_end = snapshot()
    results["thermal"] = {"start": t_start, "end": t_end}

    # prior provisional fit for comparison
    prior = Path("artifacts/verifier_floor_sweep.json")
    if prior.exists():
        p = json.loads(prior.read_text())
        results["prior_provisional_fit"] = p.get("fits", {}).get(
            "T_verify", {})

    print("\n=== TRUE P2 SWEEP (reconstructed) ===")
    print(f"{'N':>4}{'T_true':>9}{'tok/s':>8}{'E_req30':>9}{'routed':>8}"
          f"{'shared':>8}{'head':>8}{'hc':>8}{'router':>8}{'attn':>8}"
          f"{'Δ_dsa':>8}")
    for K in N_VALUES:
        r = results["rows"][str(K)]
        c = r["components"]
        print(f"{K:>4}{r['T_true_s']:>9.3f}{r['true_verify_tok_s']:>8.1f}"
              f"{r['E_req_for_30']:>9.1f}{c['routed']:>8.3f}"
              f"{c['shared']:>8.3f}{c['head']:>8.3f}{c['hc']:>8.3f}"
              f"{c['router']:>8.3f}{c['attn']:>8.3f}"
              f"{r['T_dsa_delta_s']:>8.3f}")
    print(f"\nfit: T_true(N) = {b_v:.3f}s + {a_v * 1e3:.2f}ms * N")
    print(f"intercept decomposition: {json.dumps(results['intercept_decomposition'])}")
    print(f"E[A]_req at T0-only: {30 * b_v:.1f}")

    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
