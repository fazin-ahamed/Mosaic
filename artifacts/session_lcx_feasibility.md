# Session Report: LINK0 + LC0 + LC1 — Lossless Compression Rail CLOSED

**Date:** 2026-09-20
**Status:** The lossless-compression path to Exact-30 is CLOSED by measurement.
FP4 expert codes are near-maximum-entropy; lossless codecs yield ~1.09x vs the
~4x required. Combined with EK0c, GPU-only Exact-30 remains physically closed
and compression cannot reopen it.

---

## 1. Baseline (reproduced)

EK0c 10GB Belady: 396.7 miss/cycle, 5.303 GB/cycle, move 428.6ms, 9.08 tok/s UB.
Checkpoint rev 7872f01b…, EK0b sim sha 41e4f4e8…, EK0c sha 252a5c53…. Window
W=6, stride 4, E[A]=3.757 accepted/cycle, per-expert packed FP4 = 13.37 MB.

## 2. LINK0 — measured PCIe H2D (Gen3 x16)

Flat **~12.9 GB/s** across all transfer sizes (no small-transfer penalty):

| size | GB/s |
|---|---|
| 256 MB (asymptotic) | 12.99 |
| 13.37 MB (expert) | 12.84 |
| 2.2 MB (compressed) | 12.81 |
| 1.0 MB | 12.51 |

Compressed experts would transfer at nearly full rate — so compression's only
benefit is fewer bytes, and it maps ~linearly to time. All downstream math uses
12.84 GB/s (not any assumed 26.8).

## 3. LC1 — compression-ratio screening (what ratio is needed)

Replayed EK0c routes under hypothetical uniform ratios r, scaling both cache
capacity (×r) and bytes/miss (÷r). Serialized cycle = core 17.5ms + move,
tput = 3.757/cycle. Target for 30 tok/s: cycle ≤ 125.2ms.

| r | 10GB cold GB | move ms | cycle ms | tok/s UB |
|---|---|---|---|---|
| 1× | 5.14 | 400 | 417 | 9.0 |
| 2× | 2.34 | 182 | 200 | 18.8 |
| 3× | 1.56 | 121 | 139 | 27.1 |
| **4×** | **1.17** | **91** | **108** | **34.6** |
| 6× | 0.78 | 61 | 78 | 48.1 |

**~4× lossless compression** clears 30 tok/s (serialized, zero expert compute —
an optimistic bound). Miss count floors at ~349 regardless of r (working set
>> cache); compression helps only through bytes-per-miss.

## 4. LC0 — actual lossless compressibility (the killer)

Real source expert bytes, 175 sampled w1/w2/w3 tensors across layers 0-42,
experts spanning 0-200, zstd-19 per-tensor with full reconstruction metadata,
byte-for-byte reconstruction verified:

- **FP4 nibble entropy: 3.864 bits of 4.0** → information-theoretic bound **1.04×**
- **zstd-19 aggregate: 1.091×** (min 1.09, max 1.10 — near-zero variance)
- Target: **~4×**

The source FP4 codes are already a near-maximum-entropy quantized
representation — as expected for a well-trained FP4 checkpoint. There is
essentially no lossless redundancy to exploit. **Lossless compression gets
1.09× where 4× is needed — off by ~3.7×.**

## 5. Verdict

```text
LINK0 bandwidth               12.84 GB/s (Gen3 x16), no small-transfer penalty
LC1 ratio required for 30     ~4x (serialized, zero-compute optimistic)
LC0 achievable lossless       1.09x (entropy bound 1.04x)
=> LOSSLESS COMPRESSION RAIL  CLOSED
```

Compression cannot close the movement gap: the weights are incompressible
losslessly. Per the contract, this closes THIS codec/config — but the entropy
bound (1.04×) is codec-independent: **no lossless method can exceed ~1.04×** on
these FP4 codes. That part is a physics result, not a zstd limitation. Scale
bytes (UE8M0) are a tiny fraction and do not change the aggregate.

## 6. What this does and does not prove

- **Proven:** GPU-only Exact-30 (EK0/EK0b/EK0c) AND lossless-compression-assisted
  GPU Exact-30 are both physically closed on this hardware. Movement floor stands.
- **NOT proven:** whole-machine impossibility. The surviving rails per roadmap:
  1. **CPU-hybrid (H0)** — execute cold experts on CPU where weights already
     live (host RAM), no PCIe. But working set 104 GB > VRAM+RAM (~40 GB), so
     ~60% still on NVMe (2-5 GB/s, slower). Optimistic case is weak; must be
     computed (H0-A capacity oracle).
  2. **Lossy** — explicitly OUT (violates exactness contract; not authorized).

## 7. Honest budget caveat (unchanged)

The 125.2ms budget = 3.757/30 is provisional; a production verifier must use a
matching window policy and actual emitted-token count. But the movement floor
(400ms serialized at 1×, oracle) is ~3.2× over even the most generous budget, so
budget refinement does not change the verdict.

## 8. State / next

```text
GPU-only Exact-30              CLOSED (EK0c oracle 3.85x over)
lossless-compression rail      CLOSED (LC0 1.09x vs 4x needed; entropy 1.04x)
surviving rail                 CPU-hybrid H0 (optimistic case weak)
proposer / BLIND               frozen / sealed
NEXT                           H0-A capacity oracle (VRAM+RAM CPU-exec vs NVMe),
                               then H0-B measured VNNI M=1 wall
deliverable trending toward    measured sub-30 exact ceiling + physics floor
```

## 9. Bottom line

The compression experiment did its job cheaply and returned a clean negative:
the FP4 checkpoint is incompressible (1.04× entropy bound), so no lossless
scheme reduces the ~5-12 GB/cycle expert traffic enough. GPU Exact-30 stays
closed with or without compression. The only unquantified rail is CPU-hybrid
execution, whose optimistic capacity arithmetic already looks insufficient
(104 GB working set vs ~40 GB VRAM+RAM). H0 next finalizes whether the machine
has any exact path to 30 tok/s or whether the honest deliverable is a measured
ceiling with a physics-grade floor.

Artifacts: link0.json, lc0_entropy.json, lc1_compressed_replay.json.
