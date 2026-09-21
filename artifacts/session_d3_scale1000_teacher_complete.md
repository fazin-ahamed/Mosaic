# Session Report: D3 SCALE1000 Teacher Generation Complete

**Date:** 2026-09-18
**Status:** TEACHER GENERATION COMPLETE & AUDITED (301/301)
**Teacher Process:** PID 666485 completed at 15:24 after 2769s (46 min)
**Log:** `/tmp/scale1000_teacher.log`

---

## 1. Execution Summary

- All 301 gold-only SourceTail replays for the SCALE1000 train split completed without error.
- Wall time: 2769 seconds (~46 minutes), ~9.2s per prompt average.
- Zero CUDA contention after duplicate process cleanup.
- Output: `artifacts/d3_scale1000_teacher/*.pt` + `audit.json`

---

## 2. Audit Results (audit.json)

| Field | Value |
|-------|-------|
| classification | `D3_SCALE1000_TEACHER` |
| record count | 301 / 301 |
| manifest_sha256 | `a023922b0359e4d0…` |
| dev3_sha256 | `41c5b929560e4605…` (frozen SCALE-DEV3 intact) |
| overlap: train_vs_prior_manifests | 0 |
| overlap: train_texts_vs_prior | 0 |
| overlap: train_vs_scale_dev3 | 0 |
| **all_empty_required** | **True** |

Zero prompt-ID or prompt-text leakage against every historical corpus and the frozen SCALE-DEV3 validation set.

---

## 3. Independent Re-Verification (301/301 PASS)

Beyond the script's own audit, an independent re-check of every record confirmed:

- `record_sha256` recomputed from `(category, file bytes)` matches audit
- `category` and `S`/`K` match the audit entry
- `K == len(decisions)` for every record
- `prompt_sha256` matches the manifest prompt text
- `tokens` prefix aligns exactly with the source capture
- All decision logits finite

**Result: 301/301 VERIFIED, 0 BAD.**

---

## 4. Union Corpus for HIST1_1001

The 1001-prompt training union is fully provisioned:

| Corpus | Train Prompts | Teacher Records |
|--------|--------------|-----------------|
| Original (DEV21-era) | 84 | 84 |
| SCALE210 | 126 | 126 / 126 |
| SCALE700 | 490 | 490 / 490 |
| SCALE1000 | 301 | 301 / 301 |
| **Total** | **1001** | **1001 / 1001** |

---

## 5. Current Pipeline Status

```text
SCALE1000 CAPTURE PRESENCE        PASS — 371/371
deep schema/content validation    PASS (371/371)
SafeCapture v2 stability          PASS (216 batches, RSS <= 5.5GB)
SCALE-DEV3 baselines              FROZEN (HIST1_700: S1=46/70, WS1_700: S1=41/70)
teacher generation (301 gold)     PASS — 301/301 + audit PASS
teacher independent re-verify     PASS — 301/301
union corpus provisioning         PASS — 1001/1001 teacher records

HIST1_1001 training (Stage A)     READY — NOT STARTED
WS1_1001 training (Stage B)       PENDING (after Stage A)
SCALE-DEV3 evaluation             PENDING
continuity (DEV2 + DEV21)         PENDING
paired ledgers                    PENDING
SCALE1000 verdict                 UNMEASURED
capability E[A] >= 4              OPEN
BLIND                             SEALED
H2                                CO-PRIMARY
engineering                       92-93%
```

---

## 6. Next Immediate Action

Launch Stage A training:

```bash
.venv-cu/bin/python mosaicvm/bench/d3_scale1000_train.py --stage hist
```

Contract: DDHistNet(1024, 3, 256), prev-aux, full-vocab CE + stored margin + KL,
AdamW 1e-3 / wd 1e-4, cosine 600 steps, CHUNK 512, clip 1.0, seed 104729,
output `artifacts/d3_scale1000_train/hist1001.pt`.

Baseline to beat on frozen SCALE-DEV3: **HIST1_700 S1 = 46/70 (0.657)**.
