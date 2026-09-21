"""Phase 2.7 read roofline: cold DRAM, expert reuse, L2 ceiling.

Authoritative kernel: CUDA-C via ptx_jit. Block 256. Each thread vector-loads
uint4 into a private uint32 xor accumulator and stores that one word.
No 4096-element cross-thread reduction.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.kernels.ptx_jit import compile_cuda, launch, load_kernel

ARTIFACT = Path("artifacts/read_roofline.json")

SHAPES = {
    "gate_up": (4096, 4096),  # N, K
    "down": (4096, 2048),
}

_XOR_SRC = r"""
extern "C" __global__ void xor_read(const unsigned int* src, unsigned int* dst, int n_words) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  unsigned acc = 0;
  // vector-load uint4 (16 bytes) into private xor
  int n4 = n_words / 4;
  const uint4* v = (const uint4*)src;
  for (int i = tid; i < n4; i += stride) {
    uint4 t = v[i];
    acc ^= t.x ^ t.y ^ t.z ^ t.w;
  }
  int rem = n_words - n4 * 4;
  if (tid < rem) acc ^= src[n4 * 4 + tid];
  dst[tid] = acc;
}
"""

_fn = None


def _kernel():
    global _fn
    if _fn is None:
        ptx = compile_cuda(_XOR_SRC, "xor_read")
        _fn = load_kernel(ptx, "xor_read")
    return _fn


def _packed_bytes(n: int, k: int) -> int:
    return n * (k // 2) + n * (k // 32)


def _run_stream(buf: torch.Tensor, nbytes: int, start_off: int = 0, stride: int = 0, iters: int = 20) -> float:
    """Read nbytes from buf, optionally rotating start by stride each iter."""
    fn = _kernel()
    n_words = nbytes // 4
    dst = torch.empty(256 * 256, dtype=torch.uint32, device="cuda")
    grid = 256
    block = 256
    off = [start_off]

    def step():
        o = off[0] % max(1, buf.numel() * buf.element_size() - nbytes)
        view = buf.view(torch.uint8)[o : o + nbytes].view(torch.uint32)
        launch(fn, grid, block, [view, dst, n_words])
        if stride:
            off[0] += stride

    return _bench(step, iters=iters, warmup=5)


def measure_shape(n: int, k: int) -> dict:
    expert_bytes = _packed_bytes(n, k)
    # cold DRAM: 512 MiB buffer, 64 MiB window, start += 64 MiB/iter.
    # Window is the stride so both gate_up (~8.9 MiB) and down (~4.5 MiB)
    # stay DRAM-bound; GB/s still uses expert_bytes as the kernel payload.
    cold_window = 64 * 1024 * 1024
    cold_buf = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    sec_cold = _run_stream(cold_buf, cold_window, start_off=0, stride=cold_window)
    cold_gbs = cold_window / sec_cold / 1e9
    del cold_buf
    torch.cuda.empty_cache()

    # expert reuse: one expert's bytes, reused (not L2-sized)
    reuse_buf = torch.empty(expert_bytes, dtype=torch.uint8, device="cuda")
    sec_reuse = _run_stream(reuse_buf, expert_bytes, start_off=0, stride=0)
    reuse_gbs = expert_bytes / sec_reuse / 1e9
    del reuse_buf

    # L2 ceiling: 2 MiB reused
    l2_buf = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sec_l2 = _run_stream(l2_buf, l2_buf.numel(), start_off=0, stride=0)
    l2_gbs = l2_buf.numel() / sec_l2 / 1e9
    del l2_buf
    return {
        "N": n,
        "K": k,
        "expert_bytes": expert_bytes,
        "cold_dram_gbs": cold_gbs,
        "expert_reuse_gbs": reuse_gbs,
        "l2_ceiling_gbs": l2_gbs,
        "cold_sec": sec_cold,
        "reuse_sec": sec_reuse,
        "l2_sec": sec_l2,
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    shapes = {}
    for name, (n, k) in SHAPES.items():
        shapes[name] = measure_shape(n, k)
        print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in shapes[name].items()})
    out = {"gpu": gpu, "shapes": shapes}
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
