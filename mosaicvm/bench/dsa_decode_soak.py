"""§74 NEXT 1 — full-model exact-DSA decode soak (AGENTS §51–52).

State-lifetime / integration gate. All 43 layers, layer-major (layers hold
independent attention state; sequential execution = the model's own order),
each driven through the full session lifecycle:

    prefill → decode (ring wraps, compressions fire) → chunked verify →
    decode → DEVICE MOVE (cuda→cpu→cuda mid-session) → decode →
    reset_state → re-prefill (different length) → chunk → decode

Correctness per §52: reference = a FRESH module with identical weights
running full-prefix recomputation. The soak compares INCREMENTAL state
(prefill+decode+chunk) against RECOMPUTE — the module's cached rows come
from different GEMM batchings (M=S vs M=1/M=K), which differs ~3e-6 in
fp32 and lands at 1–3 bf16 ulps (~2e-2 at |out|~2) once caches round to
bf16. Classification:
  * bf16 production mode — NUMERIC_PARITY, gate <= 0.06 (~4 bf16 ulps at
    |out|~4; the ring-overwrite bug class shows at 1.0+, 30x above gate)
  * fp32 boundary-free control (passthrough quants, representative layers)
    — proves the residual is GEMM-batching + bf16 rounding, not semantics:
    gate <= 1e-4
Bitwise == 0.0 is only asserted where shapes are identical on BOTH sides
(the module's own prefill determinism check).

Level: P1 integration gate, RTX 3060, thermal UNKNOWN; wall time recorded
as information only, never a performance claim.
"""
import json
import time
from pathlib import Path

import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention

DEV = "cuda"
CFG = V4Config()


def _clone_weights(src_mod: ExactAttention, dst_mod: ExactAttention):
    dst_mod.load_state_dict(src_mod.state_dict())   # caches are persistent=False


def _full_recompute(ref: ExactAttention, xs: torch.Tensor, upto: int):
    """Fresh-module full-prefix recompute of positions [0, upto)."""
    ref.reset_state()
    return ref(xs[:upto], 0)


def soak_layer(layer: int, max_seq: int = 4096, fp32_control: bool = False):
    """Single-processing lifecycle: every position is consumed by the soak
    module EXACTLY ONCE (decode steps advance a one-shot state machine —
    re-running an emit step double-shifts the compressor state, so the
    comparison harness must never replay positions). Reference: fresh
    module, one full prefill per checkpoint."""
    if fp32_control:
        import mosaicvm.model.dsa as D
        import mosaicvm.model.dsa_layers as DL
        D.act_quant_sim, D.fp4_act_quant_sim = (
            (lambda x, block=64: x), (lambda x, block=32: x))
        DL.act_quant_sim, DL.fp4_act_quant_sim = (
            D.act_quant_sim, D.fp4_act_quant_sim)
    torch.manual_seed(1000 + layer)
    mod = ExactAttention(CFG, layer_id=layer, max_seq=max_seq, device=DEV)
    cast_exact_attention(mod, DEV)
    ref = ExactAttention(CFG, layer_id=layer, max_seq=max_seq, device=DEV)
    cast_exact_attention(ref, DEV)
    if fp32_control:
        for m in (mod, ref):
            m.float()
            m.rebuild_freqs()
            if m.compressor is not None:
                m.compressor.rebuild_freqs()
                if m.indexer is not None:
                    m.indexer.compressor.rebuild_freqs()
            m._wire_caches()
    _clone_weights(mod, ref)

    r = {"ratio": CFG.compress_ratios[layer],
         "bitwise_max": 0.0, "parity_max": 0.0, "checks": 0, "events": {}}
    dt = torch.float32 if fp32_control else torch.bfloat16

    def recompute_ref(x, upto):
        ref.reset_state()
        return ref(x[:upto], 0).float()

    with torch.no_grad():
        # ---------------- session A ----------------
        torch.manual_seed(5000 + layer)
        S_A, DEC_A, K, DEC_B = 200, 60, 48, 5
        xs = torch.randn(S_A + DEC_A + K + DEC_B + 1, CFG.hidden_size,
                         dtype=dt, device=DEV) * 0.5
        mod(xs[:S_A], 0)
        r["events"]["prefill_A"] = True
        # decode phase — each position once
        y_inc = []
        for i in range(DEC_A):
            sp = S_A + i
            y_inc.append(mod(xs[sp:sp + 1], sp).float())
        y_ref = recompute_ref(xs, S_A + DEC_A)
        d = max((y_inc[i] - y_ref[S_A + i]).abs().max().item()
                for i in range(DEC_A))
        r["bitwise_max"] = max(r["bitwise_max"], d)
        r["checks"] += 1
        r["events"]["decode_A_wrap"] = True

        # chunked verification block at pos = S_A + DEC_A
        pos = S_A + DEC_A
        y_chunk = mod(xs[pos:pos + K], pos).float()
        y_ref = recompute_ref(xs, pos + K)
        r["parity_max"] = max(r["parity_max"],
                              (y_chunk - y_ref[pos:]).abs().max().item())
        r["checks"] += 1
        r["events"]["chunk_verify"] = True

        # decode after chunk
        y_inc = []
        for i in range(DEC_B):
            sp = pos + K + i
            y_inc.append(mod(xs[sp:sp + 1], sp).float())
        end = pos + K + DEC_B
        y_ref = recompute_ref(xs, end)
        d = max((y_inc[i] - y_ref[pos + K + i]).abs().max().item()
                for i in range(DEC_B))
        r["bitwise_max"] = max(r["bitwise_max"], d)
        r["checks"] += 1

        # ---------------- device move mid-session ----------------
        mod.to("cpu").to(DEV)
        sp = end
        y_move = mod(xs[sp:sp + 1], sp).float()
        y_ref = recompute_ref(xs, sp + 1)
        r["bitwise_max"] = max(r["bitwise_max"],
                               (y_move - y_ref[sp]).abs().max().item())
        r["checks"] += 1
        r["events"]["device_move"] = True

        # ---------------- reset + re-prefill (session B) ----------------
        mod.reset_state()
        torch.manual_seed(7000 + layer)
        S_B, K_B, DEC_C = 130, 32, 8
        xs_b = torch.randn(S_B + K_B + DEC_C, CFG.hidden_size,
                           dtype=dt, device=DEV) * 0.5
        mod(xs_b[:S_B], 0)
        y_chunk_b = mod(xs_b[S_B:S_B + K_B], S_B).float()
        y_ref_b = recompute_ref(xs_b, S_B + K_B)
        r["parity_max"] = max(r["parity_max"],
                              (y_chunk_b - y_ref_b[S_B:]).abs().max().item())
        r["checks"] += 1
        r["events"]["reset_reprefill"] = True

        y_inc = []
        for i in range(DEC_C):
            sp = S_B + K_B + i
            y_inc.append(mod(xs_b[sp:sp + 1], sp).float())
        y_ref_b = recompute_ref(xs_b, S_B + K_B + DEC_C)
        d = max((y_inc[i] - y_ref_b[S_B + K_B + i]).abs().max().item()
                for i in range(DEC_C))
        r["bitwise_max"] = max(r["bitwise_max"], d)
        r["checks"] += 1
    return r


def main():
    t0 = time.time()
    results = {"level": "P1-integration-gate", "gpu": torch.cuda.get_device_name(0),
               "thermal": "UNKNOWN", "layers": {}, "wall_s_info_only": None}
    fails = []
    for layer in range(43):
        r = soak_layer(layer)
        results["layers"][layer] = r
        ok = (r["bitwise_max"] <= 0.06 and r["parity_max"] <= 0.06
              and r["checks"] == 6)
        if not ok:
            fails.append((layer, r["bitwise_max"], r["parity_max"], r["checks"]))
        print(f"L{layer:>2} (r{r['ratio']:>3}): bitwise={r['bitwise_max']:.1e} "
              f"parity={r['parity_max']:.1e} checks={r['checks']} "
              f"{'OK' if ok else 'FAIL'}")
        torch.cuda.empty_cache()

    # fp32 boundary-free control on representative layers: proves the bf16
    # residual is GEMM-batching + cache rounding, not semantics
    results["fp32_control"] = {}
    for layer in (0, 2, 3, 20, 42):
        rc = soak_layer(layer, fp32_control=True)
        results["fp32_control"][layer] = {
            "bitwise_max": rc["bitwise_max"], "parity_max": rc["parity_max"]}
        okc = rc["bitwise_max"] <= 1e-4 and rc["parity_max"] <= 1e-4
        if not okc:
            fails.append((f"fp32-control-L{layer}",
                          rc["bitwise_max"], rc["parity_max"], rc["checks"]))
        print(f"L{layer:>2} fp32-control: bitwise={rc['bitwise_max']:.1e} "
              f"parity={rc['parity_max']:.1e} {'OK' if okc else 'FAIL'}")
        torch.cuda.empty_cache()

    results["wall_s_info_only"] = round(time.time() - t0, 1)
    agg_b = max(r["bitwise_max"] for r in results["layers"].values())
    agg_p = max(r["parity_max"] for r in results["layers"].values())
    agg_b32 = max(v["bitwise_max"] for v in results["fp32_control"].values())
    agg_p32 = max(v["parity_max"] for v in results["fp32_control"].values())
    results["summary"] = {
        "layers_passed": 43 - len([f for f in fails if not str(f[0]).startswith("fp32")]),
        "layers_failed": len(fails),
        "max_bitwise_drift_bf16": agg_b, "max_parity_drift_bf16": agg_p,
        "max_bitwise_drift_fp32_control": agg_b32,
        "max_parity_drift_fp32_control": agg_p32,
        "gate": "bf16 NUMERIC_PARITY <= 0.06 (~4 bf16 ulps); fp32 control "
                "<= 1e-4 (isolates GEMM-batching + bf16 cache rounding); "
                "semantic-bug class (ring overwrite) shows at 1.0+"}
    if fails:
        print("FAILURES:", fails)
    out = Path("artifacts/dsa_decode_soak.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"{'SOAK PASS' if not fails else 'SOAK FAIL'} — "
          f"max bitwise {agg_b:.1e}, max parity {agg_p:.1e}, "
          f"wall {results['wall_s_info_only']}s — written {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
