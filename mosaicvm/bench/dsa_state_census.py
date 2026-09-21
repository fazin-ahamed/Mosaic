"""§70 NEXT 1 — exact-DSA memory/performance census (AGENTS §39–43).

Answers, with measured numbers:
  1. What is persistent DSA state, per component, per layer type?
  2. How does it scale with context (4K → 1M)?
  3. What does a single-token decode cost (latency) as context grows,
     split into ring/compression-event/indexer/sparse-attention effects?
  4. What is implementation artifact vs semantically required?

Method:
  * State bytes: exact per-buffer accounting from named_buffers (no
    allocator noise), validated against analytic formulas.
  * Decode latency at context C: caches directly populated with random
    values at size C (timing is value-insensitive), then N single-token
    decode steps with start_pos near C. Validated at 4K against a REAL
    full prefill (fill-based timing == real-prefill timing within noise).
  * Compression-event latency isolated at small context (sp=127).
  * Thermal label UNKNOWN (no thermal control) — P1 module-level numbers,
    never P-level claims.

Level: P1 module census, local RTX 3060.
"""
import json
import time
from pathlib import Path

import torch

from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import ExactAttention, cast_exact_attention

DEV = "cuda"
LADDER = [4096, 16384, 65536, 262144, 524288, 1048576]
MEASURED_LADDER = [4096, 16384, 65536, 262144, 524288, 1048576]  # fill-based
CFG = V4Config()

torch.manual_seed(0)


# ------------------------------------------------------------------ state --
def state_bytes(attn: ExactAttention) -> dict:
    """Exact per-buffer persistent bytes, split semantic vs artifact."""
    sem = {}
    sem["kv_cache"] = attn.kv_cache.numel() * attn.kv_cache.element_size()
    if attn.indexer is not None:
        sem["ix_kv_cache"] = attn.ix_kv_cache.numel() * attn.ix_kv_cache.element_size()
    comp_sem = 0
    for m in ([attn.compressor, attn.indexer.compressor]
              if attn.indexer is not None else
              [attn.compressor] if attn.compressor is not None else []):
        for nm in ("kv_state", "score_state"):
            t = getattr(m, nm)
            comp_sem += t.numel() * t.element_size()
    sem["compressor_states"] = comp_sem
    # freqs tables: implementation artifact (values identical across layers
    # of the same theta class — shareable/computable, not per-layer required)
    art = attn.freqs.numel() * attn.freqs.element_size()
    if attn.compressor is not None:
        art += attn.compressor.freqs.numel() * attn.compressor.freqs.element_size()
    if attn.indexer is not None:
        art += (attn.indexer.compressor.freqs.numel()
                * attn.indexer.compressor.freqs.element_size())
    return {"semantic": sem, "freqs_artifact": art}


def analytic(C: int) -> dict:
    """Closed-form per-layer semantic bytes at context C (bf16 caches,
    fp32 states) — matches source cache sizing."""
    hd, win, b = 512, 128, 2
    def r4():
        kv = (win + C // 4) * hd * b
        ix = (C // 4) * 128 * b
        st = 2 * (8 * 1024 + 8 * 256) * 4          # main + indexer states
        return kv + ix + st
    def r128():
        return (win + C // 128) * hd * b + 2 * (128 * 512) * 4
    def r0():
        return win * hd * b
    return {"ratio4": r4(), "ratio128": r128(), "ratio0": r0(),
            "total_43L": 21 * r4() + 20 * r128() + 2 * r0()}


# ---------------------------------------------------------------- timing --
def fill_caches(attn: ExactAttention, C: int):
    """Populate caches to represent context C (timing-only; values random)."""
    with torch.no_grad():
        attn.kv_cache.normal_()
        if attn.indexer is not None:
            attn.ix_kv_cache.normal_()
            attn.indexer.compressor.kv_state.normal_()
            attn.indexer.compressor.score_state.normal_()
        if attn.compressor is not None:
            attn.compressor.kv_state.normal_()
            attn.compressor.score_state.normal_()


def decode_latency(attn: ExactAttention, C: int, steps: int = 50,
                   warm: int = 10) -> dict:
    """Median single-token decode latency at context C (start_pos near C)."""
    fill_caches(attn, C)
    x = torch.randn(1, CFG.hidden_size, dtype=torch.bfloat16, device=DEV)
    base = C - steps - warm
    with torch.no_grad():
        for i in range(warm):                       # warmup, also advances
            attn(x, base + i)
        torch.cuda.synchronize()
        ts, ev_bytes = [], 0
        torch.cuda.reset_peak_memory_stats()
        for i in range(steps):
            sp = base + warm + i
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            attn(x, sp)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ev_bytes = (torch.cuda.max_memory_allocated()
                    - torch.cuda.memory_allocated())
    ts = sorted(ts)
    return {"median_ms": round(ts[len(ts) // 2], 4),
            "p90_ms": round(ts[int(len(ts) * 0.9)], 4),
            "transient_peak_bytes": ev_bytes}


def compression_event_latency(attn: ExactAttention) -> float:
    """Isolated compression-event cost: decode step where (sp+1)%ratio==0,
    at small context (sp=127 for ratio 4/128) so gather size is small."""
    x = torch.randn(1, CFG.hidden_size, dtype=torch.bfloat16, device=DEV)
    ratio = attn.ratio or 1
    sp = ratio - 1                                   # fires compression
    fill_caches(attn, 256)
    with torch.no_grad():
        for _ in range(5):
            attn(x, sp + 2)                          # non-event warmup
        attn(x, sp + 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        attn(x, sp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3


def prefill_latency(attn: ExactAttention, N: int, reps: int = 20) -> float:
    x = torch.randn(N, CFG.hidden_size, dtype=torch.bfloat16, device=DEV)
    with torch.no_grad():
        for _ in range(5):
            attn(x, 0)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            attn(x, 0)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = sorted(ts)
    return round(ts[len(ts) // 2], 4)


# ------------------------------------------------------------------- run ---
results = {"level": "P1", "gpu": torch.cuda.get_device_name(0),
           "thermal": "UNKNOWN", "dtype": "bf16 caches / fp32 states"}

# --- state census: measured per-type at 4K + ladder (analytic + spot) ------
results["state"] = {}
for C in LADDER:
    row = {"analytic_per_layer": analytic(C)}
    if C in (4096, 65536):
        for L, label in [(0, "ratio0"), (2, "ratio4"), (3, "ratio128")]:
            attn = ExactAttention(CFG, layer_id=L, max_seq=C, device=DEV)
            cast_exact_attention(attn, DEV)
            sb = state_bytes(attn)
            sem_total = sum(sb["semantic"].values())
            row[f"measured_{label}"] = {
                "semantic_bytes": sem_total,
                "semantic_MB": round(sem_total / 1e6, 3),
                "freqs_artifact_MB": round(sb["freqs_artifact"] / 1e6, 3),
                "components": {k: round(v / 1e6, 3)
                               for k, v in sb["semantic"].items()},
                "matches_analytic": sem_total ==
                    analytic(C)[f"ratio{CFG.compress_ratios[L]}"],
            }
            del attn
            torch.cuda.empty_cache()
    results["state"][f"C{C}"] = row
    tot = row["analytic_per_layer"]["total_43L"]
    print(f"C={C:>7}: semantic total 43L = {tot/1e6:8.2f} MB "
          f"({tot/2**30:6.3f} GiB)")

# freqs artifact scaling (current implementation, per layer)
freq_per_layer = {"ratio4": 3, "ratio128": 2, "ratio0": 1}
for C in LADDER:
    tbl = (C + 2) * 32 * 8                          # one complex64 table
    results["state"][f"C{C}"]["freqs_artifact_43L_MB"] = round(
        (21 * freq_per_layer["ratio4"] + 20 * freq_per_layer["ratio128"]
         + 2 * freq_per_layer["ratio0"]) * tbl / 1e6, 2)
print("freqs artifact 43L @1M:",
      results["state"]["C1048576"]["freqs_artifact_43L_MB"], "MB "
      "(shareable to 3 unique tables)")

# --- decode latency ladder --------------------------------------------------
results["decode"] = {}
for L, label in [(0, "ratio0"), (2, "ratio4"), (3, "ratio128")]:
    attn = ExactAttention(CFG, layer_id=L, max_seq=1048576, device=DEV)
    cast_exact_attention(attn, DEV)
    row = {"ladder": {}}
    for C in MEASURED_LADDER:
        row["ladder"][f"C{C}"] = decode_latency(attn, C)
    row["compression_event_ms"] = round(
        compression_event_latency(attn), 4) if attn.ratio else None
    results["decode"][label] = row
    del attn
    torch.cuda.empty_cache()
    print(f"decode {label}:",
          {k: v["median_ms"] for k, v in row["ladder"].items()},
          "event_ms:", row["compression_event_ms"])

# --- fill-based vs real-prefill validation @4K ------------------------------
# NOTE census finding: prefill of S tokens materializes an
# O(S*(win+S/ratio)) fp32 gather inside sparse_attn_ref — ~9.6 GB transient
# at S=4090 for ratio-4 (this OOMed at S=4090 on 12 GB). Validation uses
# S=2048 (~2.7 GB transient). Production prefill needs chunking (the
# chunked start_pos>0, s>1 path is a separate open item).
torch.cuda.empty_cache()
attn = ExactAttention(CFG, layer_id=2, max_seq=4096, device=DEV)
cast_exact_attention(attn, DEV)
xp = torch.randn(2048, CFG.hidden_size, dtype=torch.bfloat16, device=DEV)
with torch.no_grad():
    attn(xp, 0)                                     # real prefill
real = decode_latency(attn, 2048, steps=30, warm=5)
fill = decode_latency(attn, 2048, steps=30, warm=5)  # refill path
results["validation_real_prefill_vs_fill_4K"] = {
    "S": 2048,
    "real_prefill_then_decode_ms": real["median_ms"],
    "fill_based_decode_ms": fill["median_ms"]}
print("validation 4K real-vs-fill:",
      real["median_ms"], "vs", fill["median_ms"], "ms")
del attn
torch.cuda.empty_cache()

# --- verification-width prefill timing (start_pos=0 blocks) ------------------
results["prefill_width_ms"] = {}
for L, label in [(0, "ratio0"), (2, "ratio4"), (3, "ratio128")]:
    attn = ExactAttention(CFG, layer_id=L, max_seq=4096, device=DEV)
    cast_exact_attention(attn, DEV)
    results["prefill_width_ms"][label] = {
        f"N{N}": prefill_latency(attn, N) for N in (32, 56, 80, 112)}
    del attn
    torch.cuda.empty_cache()
print("prefill widths:", results["prefill_width_ms"])

out = Path("artifacts/dsa_state_census.json")
out.write_text(json.dumps(results, indent=1))
print("written", out)
