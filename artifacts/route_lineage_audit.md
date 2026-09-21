# Route-Corpus Lineage Audit (CORRUPT_ROUTE_CORPUS = golden_routes.json)

**Corruption**: all 7 captures carry byte-identical route data
(md5 a548194e) — one trajectory replicated at capture time. Corpus
marked DO-NOT-USE in-file.

## Classification

### UNAFFECTED (timing-only consumers — route CONTENT irrelevant)
The following use routes purely as GEMM inputs (indices/weights with
realistic shapes and select counts); timing conclusions do not depend
on which specific experts the single trajectory selected:

```text
true_p2_sweep.py        TRUE-P2 fits (both imma8 and imma4)
gu_sweep.py             GU parameter race (imma4 winner)
dispatch_decomposition.py  launch-gap split (4-9%)
routed_decomposition.py GU/DOWN split (1.85:1)
verifier_floor_sweep.py provisional T(N) (superseded anyway)
true_p2.py / true_p2_mixed.py
floor_b_moe.py, mask_sweep.py, matrix_hybrid.py, hybrid_bank_sweep.py,
phase29l_trackb.py, phase29r.py (bank alloc only), wide_verify*.py,
c192_stream_bench.py (timing overlay; its route-STRUCTURE use is
  separately invalid — see below)
```

### TAINTED (distributional inputs from n=1 trajectory)
```text
wideverify_bootstrap.py   synthetic-route generation from n=1 stats
single_stack_fit.py       cold-set selection by n=1 frequency
prodfit_frontier.py       if route stats entered the frontier
```
Impact: synthetic route *distributions* may under-represent diversity;
select counts/shapes remain realistic (same source shape). Marginal
timing impact; flag for rerun when a genuine expanded corpus exists.

### INVALIDATED (route-statistics conclusions)
```text
hotbank_study.json        bank hit rates (superseded by genuine
                          G1-corpus analysis: cross-prompt 0.13-0.30)
p3_deadline_sim.json      (also had its own no-eviction bug; the
                          per-layer universe correction kills it
                          independently)
residency_oracle.py       any output consuming corpus stats
routing_traces.py         corpus-derived traces
```

### Note on c192_stream_bench.py
Its TIMING machinery is validated (zero-movement C256 run). Its route
use had TWO errors now fixed/mooted: (a) single-layer pool framing
(invalid universe), (b) the "realistic routes" rerun used the corrupt
corpus's replicated captures. The genuine-route movement numbers in
`session_stage1_closed_report.md` supersede all of it.

## Genuine route authority (current)

`artifacts/source_corpus/*.pt` gates — 7 prompts × 43 layers × 32
tokens of real kit-model routes (distinct per prompt, verified by
generation divergence). Findings derived from them: 34.1GB/block at
N=32, within-prompt reuse 0.47-0.56, adjacent-token 0.37,
cross-prompt 0.13-0.30. Expanded corpus (Track R) should reuse the
126 Stage-1 prompt set with per-capture hash assertions.
