"""NVRTC CUDA-C JIT: compile, load, launch. No nvcc, no mma.h.

Ampere MMA (fp32 accum only):
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
A: 4x .b32, B: 2x .b32, C/D: 4x .f32.
"""
from __future__ import annotations

import ctypes
import hashlib
from typing import Any

import torch

_SUCCESS = 0
_cache: dict[tuple[str, str], Any] = {}
_modules: list[Any] = []
_ctx_ready = False


def _nvrtc():
    from cuda.bindings import nvrtc
    return nvrtc


def _drv():
    from cuda.bindings import driver
    return driver


def _err_code(err) -> int:
    if isinstance(err, tuple):
        err = err[0]
    try:
        return int(err)
    except Exception:
        return 0 if "SUCCESS" in str(err) else -1


def _nvrtc_log(prog) -> str:
    nvrtc = _nvrtc()
    err, logsz = nvrtc.nvrtcGetProgramLogSize(prog)
    if _err_code(err) != _SUCCESS or logsz <= 1:
        return ""
    buf = bytearray(logsz)
    try:
        nvrtc.nvrtcGetProgramLog(prog, buf)
        return bytes(buf).decode("utf-8", "replace")
    except Exception:
        buf2 = bytes(logsz)
        nvrtc.nvrtcGetProgramLog(prog, buf2)
        return buf2.decode("utf-8", "replace")


def _ensure_ctx() -> None:
    global _ctx_ready
    if _ctx_ready:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("ptx_jit requires CUDA")
    torch.zeros(1, device="cuda")
    drv = _drv()
    err = drv.cuInit(0)
    if _err_code(err) not in (0, _SUCCESS):
        # context may already exist via torch
        pass
    _ctx_ready = True


def compile_cuda(src: str, name: str, arch: str = "sm_86") -> bytes:
    """Compile CUDA-C source to PTX. Raises with NVRTC log on failure."""
    nvrtc = _nvrtc()
    src_b = src.encode("utf-8")
    fname = (name if name.endswith(".cu") else name + ".cu").encode("utf-8")
    err, prog = nvrtc.nvrtcCreateProgram(src_b, fname, 0, [], [])
    if _err_code(err) != _SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
    opts = [f"--gpu-architecture={arch}".encode("utf-8"), b"--std=c++11"]
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if _err_code(err) != _SUCCESS:
        log = _nvrtc_log(prog)
        raise RuntimeError(f"nvrtcCompileProgram {name} failed: {err}\n{log}")
    err, ptxsz = nvrtc.nvrtcGetPTXSize(prog)
    if _err_code(err) != _SUCCESS or ptxsz <= 1:
        raise RuntimeError(f"nvrtcGetPTXSize {name} failed: {err} size={ptxsz}\n{_nvrtc_log(prog)}")
    buf = bytearray(ptxsz)
    try:
        err = nvrtc.nvrtcGetPTX(prog, buf)
        if _err_code(err) != _SUCCESS:
            raise RuntimeError(f"nvrtcGetPTX {name} failed: {err}")
        ptx = bytes(buf)
    except TypeError:
        buf2 = bytes(ptxsz)
        err = nvrtc.nvrtcGetPTX(prog, buf2)
        if _err_code(err) != _SUCCESS:
            raise RuntimeError(f"nvrtcGetPTX {name} failed: {err}")
        ptx = buf2
    try:
        nvrtc.nvrtcDestroyProgram(prog)
    except Exception:
        pass
    return ptx


def load_kernel(ptx: bytes, entry: str):
    """Load PTX module and return the named function. Cached on (hash, entry)."""
    _ensure_ctx()
    key = (hashlib.sha256(ptx).hexdigest(), entry)
    if key in _cache:
        return _cache[key]
    drv = _drv()
    err, mod = drv.cuModuleLoadData(ptx)
    if _err_code(err) != _SUCCESS:
        err2, mod = drv.cuModuleLoadDataEx(ptx, 0, [], [])
        if _err_code(err2) != _SUCCESS:
            raise RuntimeError(f"cuModuleLoadData({entry}) failed: {err} / {err2}")
    _modules.append(mod)
    err, fn = drv.cuModuleGetFunction(mod, entry.encode("utf-8"))
    if _err_code(err) != _SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction({entry}) failed: {err}")
    _cache[key] = fn
    return fn


def _as_grid(g):
    if isinstance(g, int):
        return (int(g), 1, 1)
    g = tuple(int(x) for x in g)
    return (g[0], g[1] if len(g) > 1 else 1, g[2] if len(g) > 2 else 1)


def launch(func, grid, block, args: list, shared: int = 0):
    """Launch a CUfunction. args: tensors (device ptr) or int/float scalars."""
    _ensure_ctx()
    gx, gy, gz = _as_grid(grid)
    bx, by, bz = _as_grid(block)
    keep = []
    ptrs = []
    for a in args:
        if a is None:
            c = ctypes.c_void_p(0)
        elif torch.is_tensor(a):
            if not a.is_cuda:
                raise ValueError("launch args tensors must be CUDA")
            c = ctypes.c_void_p(int(a.data_ptr()))
        elif isinstance(a, float):
            c = ctypes.c_float(a)
        elif isinstance(a, bool):
            c = ctypes.c_int(int(a))
        elif isinstance(a, int):
            c = ctypes.c_int(int(a))
        elif isinstance(a, ctypes._SimpleCData):
            c = a
        else:
            raise TypeError(f"unsupported launch arg {type(a)}")
        keep.append(c)
        ptrs.append(ctypes.cast(ctypes.pointer(c), ctypes.c_void_p))
    arr = (ctypes.c_void_p * len(ptrs))(*ptrs) if ptrs else None
    drv = _drv()
    err = drv.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, int(shared), 0, arr, 0)
    if _err_code(err) != _SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {err}")
    # keep ctypes allocations alive across the call
    return keep


MMA_F32_PTX = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"

MMA_ZEROS_SRC = r"""
extern "C" __global__ void mma_zeros() {
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
  );
  if (d0 == 123456.f) {
    ((volatile float*)0)[0] = d0 + d1 + d2 + d3;
  }
}
"""


def function_attributes(func):
    """cuFuncGetAttribute over the standard CUfunction_attribute set. Returns dict."""
    drv = _drv()
    names = [
        "NUM_REGS",
        "LOCAL_SIZE_BYTES",
        "SHARED_SIZE_BYTES",
        "CONST_SIZE_BYTES",
        "MAX_THREADS_PER_BLOCK",
        "PTX_VERSION",
        "BINARY_VERSION",
    ]
    out = {}
    for name in names:
        attr = getattr(drv.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name)
        try:
            err, val = drv.cuFuncGetAttribute(attr, func)
        except Exception:
            err, val = drv.cuFuncGetAttribute(func, attr)
        if _err_code(err) != _SUCCESS:
            raise RuntimeError(f"cuFuncGetAttribute({name}) failed: {err}")
        out[name.lower()] = int(val)
    return out


def max_active_blocks(func, tpb: int, smem: int = 0) -> int:
    """cuOccupancyMaxActiveBlocksPerMultiprocessor for a loaded function."""
    drv = _drv()
    err, n = drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, int(tpb), int(smem))
    if _err_code(err) != _SUCCESS:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {err}")
    return int(n)


def compile_mma_zeros() -> bytes:
    return compile_cuda(MMA_ZEROS_SRC, "mma_zeros")


def ptx_has_m16n8k16(ptx: bytes | str) -> bool:
    text = ptx.decode("utf-8", "replace") if isinstance(ptx, (bytes, bytearray)) else ptx
    return MMA_F32_PTX in text
