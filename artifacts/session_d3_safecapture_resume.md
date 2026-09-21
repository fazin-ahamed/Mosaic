# Session Report — SafeCapture Resume after Host-RAM OOM

**Date:** 2026-09-17

## What happened

Host-RAM OOM at 10:46 killed the `omp` wrapper and the
SCALE1000 recorder pid 3888476 (cgroup members). This was
**not** a CUDA OOM. The capture recipe is unchanged.

At the kill point: 155/371 files (70/70 val + 85 train).
All 155 passed compile/load/alignment checks — **schema
valid**, not bitwise. Bitwise proof applies only to the
two recaptures below.

## Bitwise replay (2 prompts)

- math_127, math_137 re-captured to `/tmp/safecap_replay`
- both match original: tokens, h40, h42 bitwise identical

Only these two were re-captured. no claim about the other
153 files.

## Soak — 8-prompt epoch

Worker pid 4142925 = healthy soak worker under supervisor
4147329. 8/8 landed (json_179, retrieval_137–143). This
proves the epoch loop + atomic saves, not a 20-prompt soak.

RSS 5.41 → 5.45 GB across 7 telemetry rows; frag pinned
242 MB; dark ~2.8–3.0 GB: no drift. kill 4142925 →
entire epoch already saved atomically.

## Repairs

- Stall logic: stall only when **both** `recorder_age`
  and `newest .pt age` exceed 2700 s. Previously an old
  json_178 triggered a kill every 30 s on a healthy worker
  mid-load.
- engines ended at 10:46 and were not restarted. Supervisor
  alone recycles workers now.
- Atomic save: `*.pt.tmp` → file-hash check → `mv`.
- Telemetry: rss/alloc/reserved/peak/device/frag/dark/wall/
  prompt_sha256 written per prompt.
- Workers run under `systemd-run --scope` to avoid the
  niri/omp cgroup.

## Live state (~14:24)

- train 94/301 (+9)
- val 70/70; total 164/371
- Recorder is healthy (`4169571` is a `stage1_record`,
  supervisor spawned it after the 8/8 epoch).

## Freeze

SafeCapture infrastructure is distinct from the SCALE1000
recipe. Recipe stays frozen.
