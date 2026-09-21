from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.ptx_jit import compile_mma_zeros, launch, load_kernel, ptx_has_m16n8k16


def test_mma_zeros_launch_no_crash():
    ptx = compile_mma_zeros()
    assert ptx_has_m16n8k16(ptx), "zeros MMA PTX missing m16n8k16.f32"
    fn = load_kernel(ptx, "mma_zeros")
    launch(fn, 1, 32, [])
    torch.cuda.synchronize()
