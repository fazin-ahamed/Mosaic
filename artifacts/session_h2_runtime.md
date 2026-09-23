# Session Report: H2 — integrated exact runtime scaffold (end-to-end UNVERIFIED)

**Date:** 2026-09-23
**Status:** H2 **constructs and passes one local op-parity check**. End-to-end
throughput is **UNMEASURED**: every decode attempt aborted under GPU memory
pressure on this shared 12 GB card. This report separates what is measured
from what is not, and corrects two overstatements from the in-session summary.

Evidence classes: MEASURED (wall/counter on this box) · SIMULATED · CONDITIONAL
· PROJECTED. Nothing here is a throughput result.

## What was built

`mosaicvm/bench/h2_runtime.py` — the full 43-layer source model (its own
`inference/model.py`, kernel shims, exact FP4×FP8 semantics, the same path
EK0c/source_corpus_gen use) with routed experts served from a measured
three-tier store:

    VRAM (GPU) -> RAM (host) -> NVMe (checkpoint shards)

- `TieredStore`: demand-paged LRU per tier, byte + wall-time accounting on
  every tier crossing (`nvme_reads/bytes/ms`, `ram_hits`, `vram_hits`,
  `h2d_bytes/ms`, `evictions`), pinned hot-set support, 4.25 MiB/op sizing
  verified against the checkpoint (w1/w3 [2048,2048] int8 + [2048,128] e8m0;
  w2 [4096,1024] int8 + [4096,64] e8m0).
- `static_hot` policy: hot set from EK0c's persisted route frequencies
  (`ek0c_routes.pt`, a *different* prompt — an honest online prior), pinned
  in VRAM, demand LRU on the rest.
- Route capture across all 43 gates; per-decode-position snapshots.
- Prefill/decode split with a memory diagnostic (peak reset immediately
  before prefill; allocated/reserved/device-free recorded on a common GiB
  basis; OOM path captures phase + requested size + failing op).

## MEASURED

- **Op-level parity:** `micro_parity: True` — for one expert op
  (`layers.41.ffn.experts.0.w1`), the TieredStore GEMM output is `torch.equal`
  to the unmodified source `StreamLinear` path. This establishes parity **for
  that one operation**, not full-runtime exactness, and `torch.equal` is
  value equality of the BF16 output, not a storage-byte comparison. A strict
  bitwise full-runtime claim would need token parity vs a reference decode
  (which did not complete) and/or byte comparison of stored tensors.
- **Core resident VRAM = 8.255 GiB** (`memory_allocated` after construction;
  embed 1.06 + head 2.12 GiB kept on CPU).
- **First-`_had` peak, 66-token prompt = 10.32 GiB; 9-token prompt = 9.16 GiB**
  (`max_memory_allocated`, peak reset before the measured phase in the final
  build). Prompt length moves the peak, but 9 tokens still reaches 9.16 GiB.
- **Device free ≈ 10.6 GiB** of the 11.63 GiB card at session start;
  ~1.3 GiB is held by desktop processes (sunshine 108, slack 175, nautilus
  22 MiB + driver baseline), measured via `nvidia-smi`.
- **Every decode attempt aborted** — one early run raised a catchable
  `torch.OutOfMemoryError` ("Tried to allocate 1024 MiB … 10.6 GiB free");
  later runs aborted hard without flushing the diagnostic triple.

## Corrections to the in-session summary

1. **Peak − core is NOT an activation measurement.** `max_memory_allocated`
   is historical unless reset immediately before the phase; the difference can
   include construction temporaries, cache/decode buffers, and allocator
   reserve, and 43-layer activations are not necessarily all live at once. The
   final build resets peak before prefill, but the aborting runs did not
   capture the allocated/reserved/free triple at the failure point — so
   **prefill activations are NOT isolated as the cause.** What is established:
   the full forward exceeds available memory on this contended card.
2. **The Hadamard input is ~1.03 MiB BF16 / 2.06 MiB FP32**, not 64 KiB
   (input `(1,66,64,128)` = 540,672 elements). Still not gigabytes; the
   earlier "64 KiB" conflated the H matrix with the activation. The `_had`
   print marks the last flush before abort, not the failing allocation.
3. **Memory figures now share a basis** (GiB, 1024³) and are captured
   together; "device free" is start-of-session unless labeled otherwise.

## What is NOT established

- No tok/s, latency, prefill-vs-decode split, or traffic totals — no decode
  completed.
- No token-level parity vs a reference greedy decode.
- No residency-policy comparison (needs a completed baseline first).
- Exact OOM attribution (phase, requested bytes, failing op) — instrumented
  but not captured, because the aborting runs killed the process before the
  diagnostic flushed.

## Next experiment (next session; does not resume now)

1. Keep the scaffold as **experimental, end-to-end unverified**.
2. Memory diagnostic that survives failure. A pre-prefill snapshot
   (`synchronize()`, `reset_peak_memory_stats()`, allocated/reserved/
   `mem_get_info()` free in bytes — already coded) preserves only the
   *start* state; it CANNOT reveal memory at the failing op. So: write a
   phase checkpoint to disk after each completed step (embed, per-layer or
   per-N-layers, prefill-done), and catch the CUDA `OutOfMemoryError` when
   Python can (record its message + requested bytes + failing op). Accept
   that a hard process abort may still leave the final failing allocation
   unknown — the last flushed checkpoint bounds it.
3. One completed baseline is the goal. The 9-token prompt also OOMed, so an
   even shorter prompt is a diagnostic attempt, not an assured fix. Measure
   prefill separately from decode, keep tier counters on, decode long enough
   to reveal cache churn, and label any result with its (prompt len, gen
   len) — a short prompt changes the workload and cannot stand in for the
   66-token case.
4. Compare residency policies only after a baseline completes.
5. Chunked prefill is a later engineering task (needs its own
   continuation/KV correctness + numerical parity), not a prerequisite for a
   first measurement.

## Bottom line

H2 constructs and passes one local op-parity check; **end-to-end throughput
remains unmeasured because every run OOMed** on a 12 GB card whose core alone
holds 8.26 GiB and ~1.3 GiB is occupied by desktop apps. This is enough to
pick the next experiment without overstating the failure (not proven to be
prefill activations specifically) or the verification (one op, not the runtime).

## Artifacts

- `mosaicvm/bench/h2_runtime.py` — runtime scaffold (experimental).
- No `h2_runtime.json` — no run completed.
