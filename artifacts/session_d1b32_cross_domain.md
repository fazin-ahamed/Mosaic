# D1-B32 cross-domain continuation — 2026-09-07

## Scope

Seven distinct held-out prompts: coding_16, math_16, agent_16, convo_16,
retrieval_16, longform_16, json_16. One boundary per domain; this is an
initial cross-domain probe, not the larger 21-prompt or 1000+ prompt corpus.
The same committed boundaries and seeded noise directions are scored at
K8/K16/K32. Four families (Gaussian, direction, scale, AR1), rel-L2 0.1/0.3.
Reference source tail, full h40 [4,4096], source head casts and FP32
projection. Exact h42/route/token replay must pass before perturbations.

Each completed prompt is checkpointed to a `.partial.json` marked incomplete;
the final aggregate is written only if all seven complete. New source snapshot:
`d1b32_v2_seven_domain.source.py`. Results: `d1b32_v2_seven_domain.json`;
log: `d1b32_v2_seven_domain.log`. Captures remain test-only and excluded from
student training. This run's source-reference time is not D2/P4 timing.

## Pilot already complete

On coding_16, exact replay reached K8/16/32 ceilings 9/17/33. For perturbed
states, K32 prefix+1 acceptance was 7 (Gaussian 0.3) or 12 (all other tested
family/level pairs). Top16 stayed 100%; AR1 0.3 Top4 was 96.875%, other Top4
100%. These are measured synthetic conditional probe results on one prompt,
not real-student acceptance or proof of an optimal geometry.

## Other work during this run

D2's real CPU expert probe found that the historical VNNI v4-B unsigned
integer kernel needs a source-activation adapter. Historical 0..15 activation
microbenchmarks did not exercise negative/fractional/full-range E4M3 codes.
The unsigned dot kernel itself is not invalidated. The new adapter passed bounded integer-oracle and one real-expert CPU
reference checks. No full-tail production CPU exactness or performance is
being assumed from these checks or from the old microbenchmarks.

See `d2_cpu_expert.md` for current evidence, and `d3_data_readiness.md` for
available training-corpus provenance and remaining data work.

## Completed results (reviewed 2026-09-08)

All seven exact replays passed; exact K32 E[A]=33. Each row below has seven
independent prompts and one boundary per domain. Synthetic perturbations are
not real student residuals. Confidence intervals are exploratory prompt
bootstrap intervals; this sample cannot establish within-domain variability.

| Family | Relative L2 | K8 E[A] | K16 E[A] | K32 E[A] | K32 exploratory 95% CI | K32 Top16 |
|---|---:|---:|---:|---:|---|---:|
| gauss | 0.1 | 9.00 | 16.00 | 25.57 | 19.14–31.14 | 100.0% |
| gauss | 0.3 | 5.71 | 8.00 | 11.00 | 3.71–19.71 | 100.0% |
| direction | 0.1 | 7.29 | 12.29 | 18.14 | 9.14–26.29 | 100.0% |
| direction | 0.3 | 5.29 | 8.00 | 8.71 | 3.00–14.57 | 100.0% |
| scale | 0.1 | 8.00 | 14.14 | 22.29 | 13.43–29.71 | 100.0% |
| scale | 0.3 | 6.43 | 10.00 | 13.00 | 5.71–21.43 | 100.0% |
| ar1 | 0.1 | 7.29 | 11.71 | 15.14 | 8.00–22.43 | 100.0% |
| ar1 | 0.3 | 6.43 | 9.14 | 11.00 | 5.00–17.57 | 100.0% |

At 30% error, K32 Top1 is 91.96–95.54%, Top4 99.11–100%, and Top16 100%.
Early mistakes sharply limit prefix length despite strong marginal coverage.
Do not freeze rel-L2<=0.3 as a sufficient D3 success gate. Select the student
using real residuals and downstream acceptance. Neither a K winner nor a
production tree is established; BranchPack still requires actual student logits.

The full run completed in approximately 1,589 seconds using source-reference
shims. That number is experiment runtime, not tail, verifier, or output latency.
Seven test-only captures exist; no test data was used for training.

## Completed parallel work — 2026-09-08

- CPU source adapter implemented and checked: all finite E4M3 codes in bounded
  integer-oracle cases; one complete real L41 expert matches the ordered CPU
  reference bitwise. No native-source/full-tail/timing certification. Details:
  `d2_cpu_adapter_v2.md`, `d2_cpu_expert_v2.json`.
- DATA recorder now has an explicit source-class BF16 train/validation mode.
  Captured coding_01 (train) and coding_13 (validation), nine generated-state
  rows each with full prefix h40/h42 and terminal labels. Both pass saved
  alignment/shape/finiteness checks. Details: `d3_data_readiness.md` and
  `d3_source_pilot_validation.json`. No student was trained on these pilots.
- Five targeted CPU regressions pass; modified Python modules compile.
  The complete historical test suite was not rerun. GPU source-reference
  replay and data-capture jobs finished successfully; no job remains running.

Next: expand independent train/validation data across domains, validate real
student residuals through the tail, and measure a production-correct complete
CPU/GPU tail and verifier. Seven-domain synthetic results do not establish
an architecture freeze, final output rate, or Exact-30.
