# Session Report — SpecCapture shadow (SCALE1000 recipe
# unchanged). Prefix audit + gold hashes only.

## Fail-closed GPU launcher (this drop)

`d3_speccapture_killgate.py --run-gpu` now checks for **any**
`stage1_record.py` (not a hardcoded PID) **before importing
torch**. Verified 2026-09-16: exit 2, `pids=['3723149']`,
recorder untouched, captures still 23/371.

`--prepare-gold-only` forces `CUDA_VISIBLE_DEVICES=""` and
`torch.set_default_device("cpu")`. Never imports
`stage1_record`.

**Date:** 2026-09-16
**Constraint:** SCALE1000 capture stays on the 3060. No
recipe change. No GPU steal.

## SCALE1000 still running

At gold-hash time: 22/371 captures, recorder alive, GPU
~11 GB. Engine/orch not restarted.

## Prefix audit (CPU tokenizer, same snapshot)

371 SCALE1000 prompts, `add_special_tokens=True`:

```text
mean prompt length  17.09 tokens
R_prefix            0.1746  (1107 shared occurrences / 6342)
per-domain R        convo .277  coding .240  longform .235
                    json .172   retrieval .121  agent .103
                    math .075
val-only R          0.0955
```

Verdict: **do not build Radix-KV for SCALE1000 wall-clock.**
Prompt tokens are ~17 vs N_GEN=40 decode steps. Even a
perfect prefix hit saves prefill only. Decode dominates
the measured 274–324 s/prompt. Keep the audit; revisit if
a later corpus has long shared system prefixes.

`artifacts/d3_scale1000/prefix_audit.json`
sha256 `bf5b19875c5d2f13…`

## SpecCapture-Exact kill gate (not launched)

`mosaicvm/bench/d3_speccapture_killgate.py`

- `--prepare-gold-only`: CPU hashes of 20 SCALE700 captures
  (tokens / h40 / h42), snapshot `7872f01b…`
- `--run-gpu`: **refuses with exit 2** while
  `stage1_record.py` is alive (verified live)
- GPU replay loop not implemented in this drop. After the
  3060 is free: K=2,4,8 vs those hashes; require
  `tokens` + `torch.equal(h40)` + `torch.equal(h42)`

`artifacts/d3_speccapture/gold_hashes.json`  n=20
sha256 `f538adeba2c07cb8…`

## What this does not do

- Does not change SCALE1000 authoring, capture, teacher,
  trainer, or ledger
- Does not enable CaptureWave, fused loss, compile, BDE,
  ETNA, EMA, or KD
- Does not claim a 1.5–2.2× capture speedup. That needs
  the bitwise gate first

## After SCALE1000 GPU idle

1. Implement SpecCapture loop in the kill-gate
2. Run K=2,4,8 on the 20 gold hashes
3. Only if bitwise PASS: consider a *future* capture
   pipeline. Never splice into the in-flight 371.
