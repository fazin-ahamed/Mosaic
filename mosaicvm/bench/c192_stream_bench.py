"""Track P3: REAL C192 streaming implementation benchmark.

Implements the roadmap §52/§79 design:

  all 256 experts (one real layer's weights) in PINNED CPU RAM
        (rolling warm tier, 3.43 GB)
        -> C192 slot-mapped VRAM working set (MXINTPacked4Bank over
           mutable slot arrays, 2.57 GB)
        -> async H2D on a dedicated stream
        -> per-layer ensure/wait around the TRUE-P2 full block (imma4)

Warm start: top-192 experts by trace frequency preloaded (steady
state per p3_deadline_sim). Tail misses load reactively (JIT) with
optional 1-layer lookahead (golden routes are known ahead in this
benchmark — an oracle upper bound on any predictor).

Measures per N: block wall time (steady), explicit H2D wait stall,
bytes moved — versus the static-bank TRUE-P2 block times.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, ".")

from mosaicvm.bench.floors import HIDDEN, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.fp8_core import (
    CORE_PATH, HEAD_DIM, N_HEADS, Q_LORA, O_GROUPS, O_LORA,
    CTX as KV_CTX, load_core_resident, alloc_workspace, decode_layer,
)
from mosaicvm.kernels.mxint import (
    fused_moe_mxint, MXINTPacked4Bank)
from mosaicvm.kernels.fp4_source import act_quant_ref
from mosaicvm.bench.wide_verify import _golden_windows
from mosaicvm.bench.wideverify_bootstrap import (
    layer_token_pools, bootstrap_window)
from mosaicvm.bench.true_p2_sweep import (
    hc_split_sinkhorn, hc_pre, hc_post, router_source, shared_source,
    attn_proxy, hc_head_apply, fit_linear)

from safetensors.torch import safe_open

N_VALUES = (16, 32)
OUT = Path("artifacts/c192_stream_bench.json")
GOLDEN = Path("artifacts/golden_routes.json")
CAPACITY = 256
IMMA_K = 4
HC = 4
EPS = 1e-6


class PinnedExpertPool:
    """All 256 experts of one real layer, packed4 layout, pinned."""

    def __init__(self, snap, index):
        from safetensors import safe_open as so
        e, inter, hidden = 256, 2048, 4096
        self.p_gu = torch.empty(
            e, 2 * inter, hidden // 2, dtype=torch.uint8,
            device="cpu", pin_memory=True)
        self.s_gu = torch.empty(
            e, 2 * inter, hidden // 32, dtype=torch.uint8,
            device="cpu", pin_memory=True)
        self.p_d = torch.empty(
            e, hidden, inter // 2, dtype=torch.uint8,
            device="cpu", pin_memory=True)
        self.s_d = torch.empty(
            e, hidden, inter // 32, dtype=torch.uint8,
            device="cpu", pin_memory=True)
        import random
        random.seed(7)
        # real weights from layer 5's experts (one shard set)
        loaded = 0
        for eid in range(e):
            pre = f"layers.5.ffn.experts.{eid}"
            sh = index[pre + ".w1.weight"]
            with so(str(snap / sh), framework="pt") as f:
                g = f.get_tensor(pre + ".w1.weight")
                gs = f.get_tensor(pre + ".w1.scale")
                u = f.get_tensor(pre + ".w3.weight")
                us = f.get_tensor(pre + ".w3.scale")
                d = f.get_tensor(pre + ".w2.weight")
                ds = f.get_tensor(pre + ".w2.scale")
            self.p_gu[eid, :inter] = g.view(torch.uint8)
            self.s_gu[eid, :inter] = gs.view(torch.uint8)
            self.p_gu[eid, inter:] = u.view(torch.uint8)
            self.s_gu[eid, inter:] = us.view(torch.uint8)
            self.p_d[eid] = d.view(torch.uint8)
            self.s_d[eid] = ds.view(torch.uint8)
            loaded += 1
        print(f"pinned pool: {loaded} experts, "
              f"{(self.p_gu.numel() + self.s_gu.numel() + self.p_d.numel() + self.s_d.numel())/1e9:.2f} GB",
              flush=True)


class C192WorkingSet:
    """Slot-mapped VRAM expert cache over mutable bank tensors."""

    def __init__(self, pool: PinnedExpertPool, capacity=CAPACITY):
        self.pool = pool
        self.cap = capacity
        self.p_gu = torch.empty_like(pool.p_gu[:capacity]).cuda()
        self.s_gu = torch.empty_like(pool.s_gu[:capacity]).cuda()
        self.p_d = torch.empty_like(pool.p_d[:capacity]).cuda()
        self.s_d = torch.empty_like(pool.s_d[:capacity]).cuda()
        self.bank = MXINTPacked4Bank(
            p_gu=self.p_gu, s_gu=self.s_gu, p_d=self.p_d, s_d=self.s_d,
            n_out_gu=self.p_gu.shape[1], n_out_d=self.p_d.shape[1])
        self.slot = {}
        self.lru = OrderedDict()          # expert -> last-use seq
        self.free = list(range(capacity - 1, -1, -1))  # slot stack
        self.seq = 0
        self.pinned = set()               # current-layer demands
        self.h2d = torch.cuda.Stream()
        self.pending = {}                 # slot -> (event, expert)
        self.bytes_moved = 0
        self.stall = 0.0

    def _alloc(self):
        """Evict the LRU-oldest NON-PINNED resident expert.
        Invariant: self.lru keys == self.slot keys (residents only)."""
        if self.free:
            return self.free.pop()
        for e in self.lru:
            if e not in self.pinned:
                slot = self.slot.pop(e)
                del self.lru[e]
                ev = self.pending.get(slot)
                if ev is not None:
                    ev[0].synchronize()
                    del self.pending[slot]
                return slot
        return None   # everything resident is pinned

    def preload(self, ids):
        for e in ids:
            slot = self._alloc()
            assert slot is not None, "preload capacity"
            self.p_gu[slot].copy_(self.pool.p_gu[e], non_blocking=False)
            self.s_gu[slot].copy_(self.pool.s_gu[e], non_blocking=False)
            self.p_d[slot].copy_(self.pool.p_d[e], non_blocking=False)
            self.s_d[slot].copy_(self.pool.s_d[e], non_blocking=False)
            self.slot[e] = slot
            self.seq += 1
            self.lru[e] = self.seq
        torch.cuda.synchronize()

    def ensure(self, ids):
        """Pin the demand set; async-load misses (best-effort when the
        cache is saturated with pinned demands)."""
        self.pinned.update(ids)
        misses = [e for e in ids if e not in self.slot]
        # refresh resident LRU
        for e in ids:
            if e in self.slot:
                self.seq += 1
                self.lru[e] = self.seq
        if not misses:
            return
        with torch.cuda.stream(self.h2d):
            for e in misses:
                slot = self._alloc()
                if slot is None:
                    break
                self.p_gu[slot].copy_(self.pool.p_gu[e], non_blocking=True)
                self.s_gu[slot].copy_(self.pool.s_gu[e], non_blocking=True)
                self.p_d[slot].copy_(self.pool.p_d[e], non_blocking=True)
                self.s_d[slot].copy_(self.pool.s_d[e], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.h2d)
                self.pending[slot] = (ev, e)
                self.slot[e] = slot
                self.seq += 1
                self.lru[e] = self.seq
                self.bytes_moved += (self.pool.p_gu[e].numel()
                                     + self.pool.s_gu[e].numel()
                                     + self.pool.p_d[e].numel()
                                     + self.pool.s_d[e].numel())

    def wait(self, ids):
        """Block until all needed residents' transfers completed."""
        t0 = time.perf_counter()
        waited = False
        for e in ids:
            slot = self.slot.get(e)
            if slot is None:
                continue
            ev = self.pending.get(slot)
            if ev is not None:
                ev[0].synchronize()
                del self.pending[slot]
                waited = True
        if waited:
            torch.cuda.synchronize()
            self.stall += time.perf_counter() - t0
        for e in ids:
            if e in self.slot:
                self.seq += 1
                self.lru[e] = self.seq

    def remap(self, route_idx: torch.Tensor) -> torch.Tensor:
        """Expert ids -> slot ids. One D2H + dict map + one H2D per
        call. Defensive: any non-resident expert is force-loaded
        (pinning the whole current route set so room always exists —
        max layer demand ~108 << capacity 192)."""
        shape = route_idx.shape
        cpu = route_idx.cpu().flatten().tolist()
        mp = self.slot
        missing = [e for e in set(cpu) if e not in mp]
        if missing:
            self.pinned = set(cpu) | set(missing)
            self.ensure(missing)
            self.wait(missing)
            mp = self.slot
            still = [e for e in set(cpu) if e not in mp]
            if still:
                raise RuntimeError(
                    f"capacity {self.cap} cannot serve layer demand "
                    f"({len(still)} experts missing)")
        out = torch.tensor([mp[e] for e in cpu],
                           dtype=torch.int64,
                           device=route_idx.device)
        return out.reshape(shape)


def build_block(core, ws, ws0, K, routes, head_w, hc_head, lookahead):
    x_hc = torch.randn(K, HC, HIDDEN, dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.randn(KV_CTX, 1, HEAD_DIM, dtype=torch.bfloat16,
                           device="cuda")
    state = {"i": 0, "moved0": 0.0, "stall0": 0.0}

    def step():
        nonlocal x_hc
        i = state["i"]
        state["moved0"] = ws.bytes_moved
        state["stall0"] = ws.stall
        for L in range(N_LAYERS):
            lp = core["layers"][L]
            small = lp["small"]
            # routes for this (block, layer)
            r_idx, r_w = routes[i % len(routes)][L]
            ids = sorted(set(r_idx.flatten().tolist()))
            ws.pinned = set()
            ws.ensure(ids)
            if lookahead and L + 1 < N_LAYERS:
                nx = sorted(set(
                    routes[i % len(routes)][L + 1][0]
                    .flatten().tolist()))
                ws.ensure(nx)
            decode_layer(lp["fp8"], lp["scales"], ws0)
            # ---- attention half ----
            residual = x_hc
            x, post, comb = hc_pre(
                x_hc, small["hc_attn_fn"].float(),
                small["hc_attn_base"].float(),
                small["hc_attn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["attn_norm"])
            a = attn_proxy(x, ws0, small, K, kv_cache)
            x_hc = hc_post(a, residual, post, comb)
            # ---- ffn half ----
            residual = x_hc
            x, post, comb = hc_pre(
                x_hc, small["hc_ffn_fn"].float(),
                small["hc_ffn_base"].float(),
                small["hc_ffn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["ffn_norm"])
            router_source(x, small["gate_w"], small["gate_bias"])
            ws.wait(ids)
            slot_idx = ws.remap(r_idx)
            y = fused_moe_mxint(
                x, slot_idx, r_w, bank=ws.bank,
                weight_repr="packed4", crossover=0, imma_k=IMMA_K,
                swiglu_limit=10.0)
            y = y + shared_source(x, ws0)
            x_hc = hc_post(y, residual, post, comb)
        yh = hc_head_apply(
            x_hc, hc_head["fn"], hc_head["base"], hc_head["scale"])
        yn = torch.nn.functional.rms_norm(
            yh.float(), (HIDDEN,), core["norm_w"].float())
        torch.nn.functional.linear(yn.to(torch.bfloat16).cpu(), head_w)
        state["i"] += 1

    def stats():
        return (ws.bytes_moved - state["moved0"],
                ws.stall - state["stall0"])

    return step, stats


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    t_start = snapshot()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())

    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    index = json.loads((snap / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    pool = PinnedExpertPool(snap, index)

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_resident(f)
        hc_head = {
            "fn": f.get_tensor("hc_head_fn").float().to("cuda"),
            "base": f.get_tensor("hc_head_base").float().to("cuda"),
            "scale": f.get_tensor("hc_head_scale").float().to("cuda"),
        }
        # head on CPU (bf16): frees 1.06GB for the full-residency
        # bank; bf16 halves the CPU GEMM traffic vs fp32
        head_w = f.get_tensor("head.weight")
    # the bench computes the head on CPU; drop the unused GPU copy
    del core["head_w"]
    torch.cuda.empty_cache()
    ws0 = alloc_workspace("cuda")

    # realistic blocks: for each capture and start offset, the 43
    # layer-windows form one verification block (route reuse structure
    # matters for streaming; random layer assignment destroys it)
    def build_realistic_routes(K, n_blocks=16):
        routes = []
        for cap in golden.get("captures", []):
            layers = sorted(
                cap["layers"],
                key=lambda l: int(l["layer_name"].split(".")[2]))
            L = len(layers[0]["expert_ids"])
            for o in range(0, L - K + 1, K):
                blk = []
                for lay in layers:
                    ids = lay["expert_ids"][o:o + K]
                    ws_ = lay["weights"][o:o + K]
                    blk.append((
                        torch.tensor(ids, device="cuda",
                                     dtype=torch.int64),
                        torch.tensor(ws_, device="cuda",
                                     dtype=torch.float32)))
                routes.append(blk)
                if len(routes) >= n_blocks:
                    return routes
        return routes

    results = {"gpu": torch.cuda.get_device_name(0),
               "classification": "C256_FULL_RESIDENCY_CPU_HEAD_BENCH",
               "capacity": CAPACITY, "imma_k": IMMA_K,
               "rows": {}}
    for K in N_VALUES:
        routes = build_realistic_routes(K)
        # frequency-warm the working set (steady state)
        freq = Counter()
        for blk in routes:
            for r_idx, _ in blk:
                freq.update(r_idx.flatten().tolist())
        top = [e for e, _ in freq.most_common(CAPACITY)]
        ws = C192WorkingSet(pool, CAPACITY)
        ws.preload(top)
        torch.cuda.empty_cache()
        for policy_name, lookahead in (("reactive_jit", False),
                                       ("oracle_1layer", True)):
            step, stats = build_block(core, ws, ws0, K, routes, head_w,
                                      hc_head, lookahead)
            # warm
            step(); torch.cuda.synchronize()
            ws.stall = 0.0
            ws.bytes_moved = 0
            times = []
            moved = []
            stalls = []
            for _ in range(8):
                t0 = time.perf_counter()
                step()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                m, s = stats()
                moved.append(m)
                stalls.append(s)
            times.sort()
            t_med = times[len(times) // 2]
            results["rows"][f"N{K}_{policy_name}"] = {
                "block_ms_median": round(t_med * 1e3, 1),
                "h2d_wait_ms_median": round(
                    sorted(stalls)[len(stalls)//2] * 1e3, 1),
                "moved_gb_median": round(
                    sorted(moved)[len(moved)//2] / 1e9, 3),
                "static_bank_reference_ms": (
                    427.7 if K == 16 else 599.7),
            }
            print(f"N={K:2d} {policy_name:14s}: "
                  f"block={t_med*1e3:6.1f}ms "
                  f"h2d_wait={sorted(stalls)[len(stalls)//2]*1e3:6.1f}ms "
                  f"moved={sorted(moved)[len(moved)//2]/1e9:5.2f}GB "
                  f"(static ref {427.7 if K==16 else 599.7}ms)",
                  flush=True)
        step = stats = None
        del ws
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
    results["thermal"] = {"start": t_start, "end": snapshot()}
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
