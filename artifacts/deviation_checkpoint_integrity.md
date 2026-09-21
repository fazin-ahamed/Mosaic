# UNPLANNED_DEVIATION — checkpoint attention-weight integrity

phase: 2.11 exact core / §60-1 exact DSA integration
commit: (no git; working tree @ 2026-08-29, 134/134 tests green)
expected: `.mosaicvm/core/core.safetensors` attention FP8 weights
  (e8m3 codes + e8m0 128×128-block scales) decode to finite, sane bf16
  weights, enabling real-weight numeric validation of the exact DSA port
  and of the 0xFF OPEN_SEMANTICS_CASE.
observed: attention scale tensors are not coherent e8m0 scales; several
  bf16 tensors ship literal NaN entries. Both verified byte-identical to
  upstream `deepseek-ai/DeepSeek-V4-Flash-0731` @ 7872f01b (HTTP range
  fetch of the same tensor bytes).

minimal_reproducer:
  - `layers.2.attn.wkv.scale` (F8_E8M0 [4,32]): adjacent-byte |Δ| mean
    87.2 (noise-like); a real 128×512-block quantizer yields smooth scales
    (|Δ|≈0–4). Median byte 119 while codes imply ~127. Reading the 128 raw
    bytes as little-endian fp32 at offset 0 yields 32 "plausible-looking"
    but SIGNED floats (±0.006–0.06) — negative scales are meaningless, so
    this is coincidence, not a hidden fp32 format.
  - `layers.2.attn.compressor.wgate.weight` (BF16 [1024,4096]): 155 stored
    NaN entries; `layers.3.*` 178; spread across wgate tensors archive-wide
    (wkv tensors are clean).
  - `hc_head_fn` / `hc_head_scale` / `hc_head_base` (F32): garbage
    magnitudes (std up to 8e34) or all-NaN.
  - wq_a/wq_b/wo_a/wo_b/wkv code tensors: 0 NaNs, sane magnitudes — codes
    are fine; only scales/select tensors are corrupt.

raw_benchmark: artifacts/dsa_bench.json (P1 timing, NaN-value-insensitive);
  smoothness/NaN scans inline in session log 2026-08-29.

correctness_impact:
  - Real-weight DSA numeric validation BLOCKED (mutual-finite masking finds
    0% finite overlap under every frozen decode — NaN spreads through
    RMSNorm to whole hidden rows).
  - Port correctness proven INDEPENDENTLY of real weights: fp32 boundary-
    free comparison against the checkpoint's own reference implementation
    (scripts/validate_dsa_fp32.py) is BITWISE EXACT on L0 (ratio 0),
    L2 (ratio 4 + indexer), L3 (ratio 128) at s=256. Production bf16 mode
    (scripts/validate_dsa_source.py): 71% of outputs within rtol 2%
    (max |abs| 0.055), residual = bf16 kernel-order noise at the source's
    own cast boundaries.
  - 0xFF OPEN_SEMANTICS_CASE cannot be closed by decode choice (see
    artifacts/ff_semantics.md) — subsumed by this deviation.

performance_impact: none for timing (NaN computes at full speed);
  dsa_bench numbers valid as P1 timing.

memory_impact: none.

suspected_causes:
  1. Upstream publish bug: attention-scale tensors scrambled/corrupted
     during export (e.g. wrong tensor bytes, layout transform written
     without the inverse, or garbage-filled allocation).
  2. Deliberate obfuscation/red-herring of non-expert weights (codes fine,
     scales destroyed — a targeted pattern).
  3. Undocumented scale container format (no reader in the kit, vLLM, or
     converter decodes it; all three treat these bytes as plain e8m0).

confidence: high (0.95) that the tensors as published cannot serve the
  published inference kit: the kit's own `inference_convert.py` performs no
  repair, and its `generate.py` loads with `strict=False` — silently
  leaving whatever the checkpoint contains.

possible_solutions:
  a. Ask upstream (HF discussion/issue) for the attention-scale format or a
     corrected revision. [preferred]
  b. Obtain DeepSeek's converted `model{i}-mp{mp}.safetensors` (the format
     the kit actually consumes) and rebuild the core archive from it.
  c. Re-derive scales locally by requantizing the (clean) e4m3 codes
     against a calibration set — NOT exact; forbidden by the exactness
     contract as anything but a documented approximation.
  d. If (a)/(b) fail: quantify end-to-end impact under the exactness
     contract before any further core work assumes real weights.

recommended_next_experiment: (a) upstream issue with the
  minimal_reproducer evidence; in parallel try (b) — check whether the HF
  repo or DeepSeek's release page ships kit-format weights.

## What is NOT blocked

- Exact DSA integration itself (done; bitwise-proven port, P1 benchmarked).
- P2 rerun timing (NaN-insensitive); P2 *correctness* outputs will carry
  NaN regions until this deviation resolves.
- All work that does not consume attention scale tensors or hc_head_*:
  experts (FP4 tensors untouched by this finding), router, LM head, norms,
  DSpark (own tensors — spot-check before reliance).

---

## ADDENDUM 2026-09-02 — RESOLVED (stale-CDN acquisition; upstream exonerated)

Every finding in this deviation was an artifact of the acquisition
method, not of the published checkpoint:

- `fetch_core.py` used plain HTTP `Range` requests against
  `huggingface.co/…/resolve/main/…`. The CDN's range-serving path
  returned a **stale file version** (garbage attention scales with
  0xFF, different weight codes, NaN wgates, garbage hc_head) while the
  authoritative object (served via `hf_hub_download`'s xet path, same
  revision `7872f01b`) is clean.
- The "byte-identical to upstream" verification above re-fetched via
  the same plain-HTTP-range method — a self-consistent check that
  could not detect the staleness.
- Evidence chain and archive rebuild: `artifacts/ff_semantics.md`
  (CLOSED). New archive sha `a9089f25fa85bac9`; closure verification:
  0/390 scales with 0xFF, 0 smoothness failures, wq_b(L2) sha16
  `65d1cf8c02b74b16` == transformers-runtime hash, 0 NaN wgates,
  hc_head finite, real-weight pow2 dequant finite with amax 0.22–0.34.

Status changes:

- **Real-weight DSA numeric validation: UNBLOCKED.**
- **Suspected causes 1–3: withdrawn** (no upstream publish bug, no
  obfuscation, no undocumented format).
- **Solution (a) [upstream issue]: withdrawn** — do not file
  (`upstream_issue_draft.md` is void).
- **Solution (b) [kit-format weights]: obsolete** — the HF-format
  shards ARE the authoritative weights.
