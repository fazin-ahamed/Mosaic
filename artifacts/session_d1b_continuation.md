# D1-B continuation and integrity audit — 2026-09-06

The future-h40 + source-tail architecture remains the leading research
candidate. The handoff was read against the harness, checkpoint source,
D1-R/D1-B artifacts and H1/VNNI implementation. The new handoff supersedes
the workspace's obsolete B3-S AGENTS.md; both roadmaps are preserved.
No completion percentage or production throughput is newly certified.

## Corrections made before extending the experiment

1. **K and root accounting.** Old `EA_natural` started at S-1 (the exact
   root) and counted only K-1 uncertain decisions. It then added one for
   prefix+1. V2 scores K uncertain decisions S..S+K-1, reports root separately,
   and uses 1 + accepted speculative prefix. A first uncertain miss gives 1,
   an all-correct K gives K+1. Target verification width N must still be
   established by integrated-cycle semantics; it is not implicitly K.
2. **Sliding windows were not refreshed cycles.** Old `EA_allwin` cropped a
   single perturbed trajectory without making each crop's prior tail history
   exact. V2 reports only the actual committed boundary. Old reported values
   remain historical probe statistics, not certified production E[A]. The
   artifact has 39 overlapping K8 windows per configuration, not ~180
   independent windows; its independent sample is three coding prompts.
3. **Terminal label was dropped.** The generator already records the next
   token for the final h40 state. V2 includes it rather than losing one
   speculative decision. K8/16/32 use matched prefixes of the same trajectory.
4. **Incomplete resets.** The old reset omitted compressor kv_state and the
   indexer's compressor state. V2 resets nested kv_cache, kv_state and
   score_state before reference prompt generation and every independent
   tail replay. Constants and weights remain intact.
5. **Source head casts.** A hand-coded FP32 HC/norm formula skipped source
   BF16 casts. V2 calls the actual source HC head and RMSNorm, followed by
   CPU FP32 head projection. Exact h42 equality, greedy decisions and route
   matches are hard gates; an exact-replay failure raises a nonzero error.
6. **Reproducibility.** Noise uses FP32 and matched random directions across
   levels. V2 records configuration, GPU, torch version, revision, startup
   script hash, raw per-position correctness and per-prompt E[A]. CI resamples
   prompts, never overlapping windows; one prompt receives no invented CI.
   Default output is new, preserving both historical JSON curves.
7. **Full h40 interface.** Captured source h40 is [4,4096] per position,
   i.e. 16,384 values. D1-R did not establish that an arbitrary 4,096-value
   compression of this four-stream state suffices. Budget D3 accordingly.

Historical synthetic Top-k observations remain encouraging. Their exact
production-head values require a v2 rerun. Changing tail routes does not
prove tail MoE redundancy. Relative L2 does not imply a universal cosine:
cos=1-e²/2 assumes equal norms; pure scaling can retain cosine=1 at large L2.
Synthetic state errors also cannot substitute for measured student residuals.

## D2 and H1

The complete local checkpoint is available and all 48 shard headers were
read. Measured source tail payload is **7,158,447,536 bytes**. VNNI v4-B's
expanded routed representation requires **14,495,514,624 bytes**. An FP32
head requires another **2,118,123,520 bytes**. Packed+expanded+head alone
is approximately **23.46 GB**, before remaining runtime allocations.
See `d2_tail_inventory.json` and `d2_readiness.md` for scope and details.

Production D2 wall remains unmeasured. The current torch source shims and
serial continuation cannot be presented as a production chunked tail.
Source attention's nonzero-position path writes one ring slot; K>1 batch
continuation needs a separate causal implementation and parity proof.

H1's shared request/result queue allowed self-consumption. It now uses
separate queues. CPU FLOPs now credit only actual GU work, removing the
1.5x full-expert overcredit. New H1 output is explicitly synthetic and
writes `track_h_hybrid_v2.json`; the old result is unchanged. The historical
~1.113 s wall is not a production H2 measurement or sufficient evidence for
an Exact-30 projection.

## Checks and execution

Four CPU regression checks pass: acceptance endpoints/first/last rejection,
nested state reset under different poison values, prompt-level CI behavior,
and H1 request/completion transport with actual GU accounting. Changed
Python files compile. These tests do not replace full source-tail isolation
checks at compressor boundaries and ring wrap.

A local RTX 3060 K8 smoke run passed bit-exact h42 replay, exact source-head
Top1 and exact routes. On coding_16, scale error 0.3 gave E[A]=9. This is
one synthetic boundary, not a student benchmark. `d1b_v2_smoke.json` was
produced during development; its end-of-run script hash is not a reliable
hash of the loaded version. Prefer the subsequent frozen pilot and its
startup script hash for evidence.

The matched K8/K16/K32 pilot completed on coding_16. Exact replay produced
bit-exact h42, exact routes, and ceilings 9/17/33. At rel-L2 0.1 and 0.3,
all four synthetic families gave K32 E[A]=12 except Gaussian 0.3, which
gave 7. K32 Top1 was 90.625–96.875%, Top4 96.875–100%, and Top16 100%.
This is ONE independent boundary; no generalization or student PASS follows.
The result specifically shows that high marginal token accuracy does not
ensure a long accepted prefix. The run took about 290 s in the reference
harness; that is not production tail/cycle latency.

The exact test-only state capture is held-out validation material and must
not enter D3 training. A subsequent seven-domain run uses one distinct
held-out prompt per domain, matched seeds and K8/K16/K32. Completed prompts
are saved in a clearly incomplete partial artifact until the whole run ends.

## Reproduce and extend

From the workspace, the small K32 pilot command is:

```bash
D1B_N_TEST=1 D1B_HORIZONS=8,16,32 D1B_N_STEPS=32 \
D1B_LEVELS=0.1,0.3 D1B_FAMILIES=gauss,direction,scale,ar1 \
D1B_CAPTURE_DIR=artifacts/d1b32_test_captures \
D1B_OUT=artifacts/d1b32_v2_pilot.json SCG_CACHE_CAP=24 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-cu/bin/python -u mosaicvm/bench/d1b_tail_robustness.py
```

After the pilot, set D1B_N_TEST=21 and use a distinct output filename to
cover all seven existing held-out domains. This remains just three prompts
per domain, well below the required large DATA corpus. The serial source
reference runs slowly; its elapsed time is not production tail latency.

Then implement production D2 with complete expert execution and realistic
representation residency; collect prompt-split training data separately;
evaluate D3 residuals through this corrected tail; and finish real H2.
The final decision remains measured held-out E[A] divided by complete cycle
wall, followed by sustained production P4 and long-context qualification.
