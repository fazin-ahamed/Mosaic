# 0xFF / attention-scale semantics — CLOSED (RESOLVED: stale-CDN acquisition)

Date: 2026-09-02 · Phase: Track A 0xFF closure
Status: **CLOSED — ROOT CAUSE WAS LOCAL, NOT UPSTREAM.**

## Resolution

The 0xFF bytes, the noise-like scale tensors, the bf16 wgate NaNs, and the
garbage `hc_head_*` values never existed in the authoritative checkpoint.
They existed only in a **STALE CDN copy of the shards** served to plain
HTTP-range requests:

| acquisition path | `layers.2.attn.wq_b.scale` bytes | weights |
|---|---|---|
| plain HTTP range (`huggingface.co/resolve/main` + `Range`) | `[159, 60, 22, ...]` garbage, 0xFF present | decoy/stale codes (sha `6d51b65f…`) |
| `hf_hub_download` (xet-backed) | `[115, 115, ...]` clean e8m0 | authoritative codes (sha `65d1cf8c…`) |

Both paths resolve the SAME revision `7872f01b` — the CDN's HTTP-range
edge served a stale file version while the xet content-addressed path
returned the current object. Verified from the same sandbox, same URL,
same offsets, same revision: the two methods return different bytes.

## Chain of evidence

1. `ff_runtime_probe.py` (sandbox, transformers 5.14.1): the working
   reference model holds clean e8m0 scales `{114..118}` and its
   `q_b_proj` weight hash `65d1cf8c02b74b16` — different from our
   archive's `wq_b` (`6d51b65f9dea1427`).
2. `ff_runtime_probe2.py`: 0/125 attention weight tensors match the
   archive; per-element sign agreement 0.50 (independent data); value
   histograms statistically similar (both are e4m3 weight dumps).
3. Direct plain-HTTP range fetch of the same tensor at the same
   revision reproduces the ARCHIVE bytes (garbage) — so the archive was
   a faithful copy of what range-requests return.
4. The same plain-HTTP range fetch FROM THE SANDBOX ITSELF also returns
   the garbage — while the sandbox's `hf_hub_download`-cached file has
   the clean bytes. Method, not location, determines the result.
5. `ff_gemm_verify.pt` (captured x/W/s/y of `layers.2.q_b_proj`):
   `y ≈ x @ (codes × 2^(scale−127))` at 2.2% residual (fp8 noise) —
   the runtime semantics are exactly `W_eff = codes × e8m0`, with NO
   0xFF ever entering the working path.

## Archive rebuild (the fix)

`mosaicvm/bench/rebuild_core_hub.py` rebuilt `.mosaicvm/core/` from
authoritative hub bytes (sha `a9089f25fa85bac9`, 9.44 GB). Closure
verification (in `manifest.json`):

```text
attention scale tensors        390
  with 0xFF bytes              0
  failing smoothness           0
layers.2.attn.wq_b sha16       65d1cf8c02b74b16  (== runtime)
bf16 wgate tensors with NaN    0
hc_head_fn/base/scale          all finite
real-weight dequant (pow2)     finite, amax 0.22–0.34, std ~0.025,
                               scales 2^-13..2^-10
```

The stale copy is preserved at `.mosaicvm/core/core.stale-cdn.safetensors`
for forensic reference. Tests: 269/269 green after the swap.

## Frozen semantics (unchanged, now moot)

`dequant_scale(ff=...)`: `pow2` (2^(E−127), the hardware/vLLM decode) is
correct for all bytes that exist in the authoritative files; `saturate`
and `nan` policies remain available but are no longer load-bearing. The
production loader (`runtime.fp8.dequant_fp8`, `s.float()`) matches.

## Consequences

- The **checkpoint-integrity deviation is dissolved** (see
  `deviation_checkpoint_integrity.md` addendum): real-weight DSA numeric
  validation is UNBLOCKED.
- The **upstream issue must NOT be filed** (`upstream_issue_draft.md`
  withdrawn): upstream data was never defective.
- Any future re-acquisition MUST use `hf_hub_download` (xet), never raw
  HTTP ranges against `resolve/main`.
- All sandbox-validated work (B3-S/RFIX chains, route captures, DSpark
  dataset) already used hub-downloaded bytes — the true model — and
  stands unchanged.
