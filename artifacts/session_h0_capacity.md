# Session Report: H0-A — CPU-Hybrid Combined-Memory Capacity Bound

**Date:** 2026-09-20
**Status:** INCONCLUSIVE by design margin. Whether the exact model fits the
combined VRAM+RAM store depends on tight reserves: the single-session expert
working set (32-37 GB) straddles the usable combined store (30-38 GB). This is
the roadmap's "inconclusive -> proceed to H0-B" case, NOT a whole-machine close.

---

## 1. Provenance reconciliation (fixed)

- **LC1/EK0c mismatch resolved:** LC1 reused the SHORT multi-domain corpus
  (`ek0b_residency.load_cycles`), giving 384.1 miss @10GB — NOT the EK0c
  256-token long stream (396.7). EK0c never persisted raw routes. The two agree
  within 3% (short vs long regime); H0-A uses the short corpus, labeled as proxy.
- **349.6 plateau explained:** it is COMPULSORY first-touches. Unlimited-cache
  Belady per-cycle miss == compulsory first-touch count EXACTLY (verified both
  test domains). On short reset-separated traces, most misses are cold-start,
  not capacity.
- **LC0 correction:** 105 tensors (not 175); aggregate 1.0910x. The 1.04x figure
  is the MARGINAL-nibble entropy bound, NOT a universal lossless ceiling —
  restated as "tested zstd-19 gives 1.091x; no demonstrated route to 4x;
  codec engineering deferred," per your correction.

## 2. Measured inputs

- H2D (LINK0): 12.84 GB/s. **NVMe cold random expert-size reads: 0.55 GB/s
  aggregate** (1.36 GB/s p50), measured with `posix_fadvise(DONTNEED)`. NVMe is
  ~10-23x slower than H2D — reads that reach storage dominate any cycle.
- Combined store (usable): VRAM 12 − reserve, + RAM 31 − reserve. Reserves for
  core/KV/shared/activations/staging (VRAM) and OS/runtime/proposer/pagecache
  (RAM). Conservative: 8+22=30 GB. Aggressive: 10+25=35 GB.

## 3. The decisive number: single-session working set

Per-domain (one generation session) DISTINCT experts touched:

```
agent 31.8  coding 32.6  convo 30.8  retrieval 33.9
longgen 36.8  math 32.0  json 31.3   GB
ALL-domain union: 103.1 GB  (multi-session serving, not one session)
```

## 4. Optimistic oracle result (combined store, zero compute, Belady)

On the SHORT trace the simulator reports ZERO capacity misses at a 30 GB store
(all 349.6 misses/cycle are compulsory first-touches). This does NOT establish
steady state: the 7-cycle-per-domain trace is too short for evict-then-
re-request churn to appear, and future first-touches still require storage
reads. Any per-cycle churn / tok-s number derived from it would be unsupported.

What the JSON DOES support: single-session observed footprint 32-37 GB vs
usable combined store 30-38 GB — the two straddle. Whether a SUSTAINED session
fits is `INCONCLUSIVE_SHORT_TRACE` and is the object of the long-trace recovery
(section 6) and H0-B.

Capacity SCENARIO (not a measured sustained result): if effective store
(aggressive 35 GB x 1.091 = 38 GB) >= session footprint, weight reads from NVMe
are eliminated UNDER THE FIT ASSUMPTION, and CPU expert compute (H0-B) becomes
the gate. If store < footprint, unavoidable NVMe reads at 0.55-1.36 GB/s
dominate. The exact crossover needs the long trace.

## 5. Verdict: INCONCLUSIVE (proceed to H0-B)

The outcome flips on ~5 GB of memory-reserve assumption:
- **Conservative reserves (30 GB store):** session WS exceeds store, ~2-48
  experts/cycle churn to NVMe at 0.55-1.36 GB/s -> **1-8 tok/s. Hybrid fails.**
- **Aggressive reserves + 1.091x (38 GB store):** session WS fits -> zero steady
  NVMe -> movement no longer the bottleneck -> **CPU expert COMPUTE becomes the
  gate (H0-B decides).**

So H0-A does NOT close the machine. It says: **if the single-session working set
can be made to fit VRAM+RAM (plausible at 32-37 GB vs ~30-38 GB usable), the
bottleneck moves from movement to CPU expert execution throughput.** That is
exactly the H0-B question.

## 6. Honest limitations

- **Short-trace proxy:** 7 domains x ~7 cycles. Capacity misses (evict-then-
  re-request) barely materialize in 7 cycles, so the sim's "capacity_only=0" is
  an artifact of trace length, corrected here by the working-set-vs-store
  analysis. A long single-session trace with real eviction churn would tighten
  the 32-37 GB WS number.
- **Session WS may grow with context length** (more distinct experts over more
  tokens). 256-token EK0c already showed the union near ~900/cycle; a 1M-context
  session WS is unmeasured and could exceed any store -> re-open at long context.
- W6/stride-4 remains a proxy for real speculative verification requests.

## 7. State / next

```text
GPU-only Exact-30              CLOSED (EK0c oracle 3.85x over)
lossless compression rail      CLOSED (LC0 1.091x; codec engineering deferred)
CPU-hybrid capacity (H0-A)     INCONCLUSIVE — session WS 32-37GB vs store 30-38GB
                               conservative: fails (NVMe-bound 1-8 tok/s)
                               aggressive:   fits -> compute-bound (H0-B decides)
NEXT                           H0-B: measured exact CPU VNNI expert wall at M=1
                               (65% of calls), then M distribution, on the
                               per-layer critical path. Decides if fits-case
                               reaches 30 tok/s or is CPU-compute-bound.
proposer / BLIND               frozen / sealed; exactness intact
```

## 8. Bottom line

H0-A refutes the premature "104 GB >> memory so hybrid is dead" reasoning: the
relevant quantity is the SINGLE-SESSION working set (32-37 GB), which plausibly
fits the combined VRAM+RAM store. If it fits, movement stops being the wall and
CPU expert compute throughput (H0-B) becomes decisive. If reserves are tight or
long-context WS grows, NVMe churn caps throughput at single-digit tok/s. The
honest next measurement is H0-B: can exact M=1 VNNI experts execute fast enough
on the 43-layer critical path to approach 30 tok/s when the working set fits?

Artifacts: link0.json, lc0_entropy.json, lc1_compressed_replay.json, h0_capacity.json.
