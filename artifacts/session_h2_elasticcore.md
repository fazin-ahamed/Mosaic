# Session Report: H2 ElasticCore — wo_a leasing (parity PASS, decode still blocked)

**Date:** 2026-09-23
**Status:** ElasticCore Mode A (host-BF16 `wo_a` leasing) is **implemented and
parity-validated**, and it **measurably frees 2.625 GiB of resident GPU core**.
End-to-end decode **still does not complete**, but the failure mode has
**shifted away from the prefill VRAM OOM** — after freeing 2.6 GiB, a short
forward should not exceed device-free, and the observed abort is a hard
process kill with a tilelang/nvcc JIT compile error in the path, not a caught
CUDA OOM. Evidence classes: MEASURED · SIMULATED · CONDITIONAL · PROJECTED.

## Source facts verified (MEASURED against the checkpoint)

- `wo_a` is a **BF16 execution weight** (`model.py:468`,
  `ColumnParallelLinear(..., dtype=torch.bfloat16)`), consumed by the
  attention einsum `torch.einsum("bsgd,grd->bsgr", o, wo_a)` (`model.py:546`).
- Checkpoint stores it **FP8 E4M3 [8192,4096]** (32 MiB) + scale
  `float8_e8m0fnu [64,32]`; H2's `dequant_wo_a` expands to **BF16 [8192,4096]
  = 64 MiB/layer**, **2.6875 GiB across 43 layers**. Confirmed for layers
  0/20/42 and by dequant. This matches the ElasticCore document exactly.

## ElasticCore Mode A — implemented

Each attention `wo_a` BF16 matrix is moved to **pinned host RAM**; all 43
layers share **one 64 MiB GPU lease** refreshed by a forward-pre-hook before
that layer's einsum. Layers run sequentially on the default stream, so
reusing the lease is causally safe (layer i's einsum is enqueued before
layer i+1's `copy_`). wo_a bytes, layout, and the source einsum are
unchanged, so the arithmetic is exact by construction.

## Gates (MEASURED)

| gate | result |
|---|---|
| **wo_a lease bytes == resident BF16** | **True** (`torch.equal`) |
| **leased-wo_a einsum == resident-wo_a einsum** | **True**, bit-identical |
| **resident core, full wo_a** | **8.255 GiB** |
| **resident core, elastic wo_a** | **5.63 GiB** |
| **freed** | **2.625 GiB** (= 2.6875 bank − 0.0625 retained lease) |
| micro-parity (one expert op vs source StreamLinear) | True (prior session) |

ElasticCore does exactly what it claims: it relocates 2.625 GiB of permanent
GPU residency at zero arithmetic cost (bit-exact), giving a plausible path to
a 10 GiB then 8 GiB process cap.

## Decode: still NOT completed — failure mode has shifted

Every decode attempt aborted. New evidence changes the attribution:

- **Not host-RAM:** `dmesg` shows **no OOM-killer** event; RAM had 17 GiB free.
- **Not (any longer) the prefill VRAM OOM:** ElasticCore freed 2.625 GiB. The
  earlier 9-token peak was 9.16 GiB on a 8.255 GiB core; on a 5.63 GiB core
  the same forward should peak ~6.5 GiB, well under the ~10.6 GiB device-free.
  A pure short-prompt prefill VRAM OOM is no longer the likely cause.
- **A tilelang/nvcc JIT compile error appears in the forward path:**
  `unsupported GNU version! gcc versions later than 15 are not supported`
  while tilelang tried to `nvcc --cubin -arch=sm_86 tvm_kernels.cu`. tilelang
  is **not imported by the H2 code or the kernel shim** — the source model
  imports only the six shimmed `kernel` functions — so this is triggered
  transitively and is an **environment-level** blocker (host gcc newer than
  nvcc supports), deterministic and independent of memory.

**Honest limit:** the process aborts hard without flushing the memory
diagnostic, so I have **not isolated** whether the abort is the tilelang
compile failure or a hard CUDA OOM that aborts rather than raises. What is
established: it is **not** a host-RAM OOM, and it is **not** cleanly the
prefill VRAM OOM that motivated ElasticCore — that specific wall was moved.

## Reporting fixes applied to the scaffold

- Throughput split into **TTFT** (prefill wall to first token),
  **`steady_decode_tok_s` = (N−1)/decode_wall** (first token is prefill), and
  **`request_tok_s` = N/(TTFT+decode_wall)** — replacing the old
  `N_GEN/decode_wall` which mixed prefill and decode.
- NVMe counter relabeled **`nvme_logical_bytes`** (requested size; `safe_open`
  may hit OS page cache) — not proven device reads.
- `wo_a` H2D leases counted separately (`woa_leases`, `woa_h2d_bytes/ms`).

## What is NOT established

- No completed decode, no tok/s, no token parity vs a reference decode.
- The exact abort cause (tilelang JIT vs hard CUDA OOM) — the survivable
  diagnostic (per-step checkpoints + catchable-OOM capture + logged tilelang
  invocation) is the next step and was not captured because the process died.

## Next steps

1. **Resolve the tilelang JIT path first** — it is deterministic and may be
   the actual blocker. Either force the reference/shim path for whatever op
   triggers it (log the op and shape), or set `-allow-unsupported-compiler` /
   point nvcc at a gcc ≤ 15, or confirm it falls back and is not fatal.
2. Only then re-attempt the short-prompt decode under a measured 10 GiB cap,
   with per-step memory checkpoints written to disk (survive hard abort) and
   catchable-OOM capture, then 8 GiB. Label results by (prompt, gen) lengths.
3. Compare residency policies once a baseline completes.
4. Mode B/fused (keep FP8 resident, expand one layer on demand, ~1.34 GiB
   saved without repeated PCIe) only after Mode A yields a completed run —
   with its own BF16-byte parity gate.

## Bottom line

ElasticCore's central claim is **validated**: `wo_a` leasing is bit-exact and
frees a measured **2.625 GiB** of resident core (8.255 → 5.63 GiB). The exact
runtime is now much closer to fitting an 8–10 GiB cap. But **no decode has
completed**, and the remaining blocker is no longer the prefill VRAM OOM — the
next gate is resolving the tilelang/nvcc JIT compile failure in the forward
path, then measuring a capped short-prompt decode. This addresses fit, not the
expert-storage bottleneck, and makes no throughput claim.

## Artifacts

- `mosaicvm/bench/h2_runtime.py` — ElasticCore Mode A + gates + reporting fixes.
- `artifacts/h2_elasticcore_gates.json` — measured parity + VRAM numbers.
- No `h2_runtime.json` — no run completed.
