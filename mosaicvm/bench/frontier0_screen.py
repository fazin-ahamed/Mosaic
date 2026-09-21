"""FRONTIER-0 tier-1 screen (v2, corrected).

Fixes five methodology bugs found in review of v1:
  1. SHARED acceptance ledger: one acceptance sequence is drawn once and
     replayed identically across every policy, so all policies see the same
     cycles, positions, and acceptances (paired comparison).
  2. Explicit token accounting: `a` = accepted draft tokens (truncated-geometric
     fit to mean 3.757). Emitted tokens/cycle = a + 1 (a accepted drafts + 1
     correction/bonus). Verification must cover positions up to the first
     rejection = min(a+1, W) rows. Results are reported PER EMITTED TOKEN.
  3. Real chronological cache: a layer-major access stream (for layer: for row:
     for expert) is replayed through an LRU of fixed capacity. EVERY load
     (speculative included) occupies capacity and can evict; there is no
     "commit only accepted rows" shortcut. Hits/misses are counted on the
     actual access order, not set-collapsed per window.
  4. No unproven lower-bound claim: window rows reuse accepted-path routes as a
     proxy for speculative candidates. The demand DIRECTION vs real candidates
     is UNKNOWN (a rejected candidate may hit resident OR colder experts).
  5. LRU is not optimal: an offline Belady lower bound is computed on the SAME
     request stream, so the gap between LRU and the best legal residency policy
     is explicit.

This is a demand+cache SIMULATION on real routes. It is NOT latency, NOT
achieved throughput, and NOT a proof that every residency policy fails.
"""
from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import torch

BYTES = 13369344
W = 6
STORE_GB = [30, 35, 38]
EA = 3.757
SEED = 20260920


def fit_p():
    """continue prob p s.t. sum_{i=0..W-1} p^i == EA (mean accepted drafts)."""
    lo, hi = 0.0, 1.0
    for _ in range(90):
        p = (lo + hi) / 2
        if sum(p ** i for i in range(W)) < EA:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def build_ledger(routes, p):
    """One shared sequence of (cur, a, navail); a = accepted drafts (1..W).

    Advances the committed prefix by a+1 (a accepted + 1 correction) per cycle.
    Identical for every policy.
    """
    rng = random.Random(SEED)
    ledger = []
    cur = 0
    n = len(routes)
    while cur < n - 1:
        navail = min(W, n - cur)
        a = 1
        while a < navail and rng.random() < p:
            a += 1
        ledger.append((cur, a, navail))
        cur += a + 1              # a accepted drafts + 1 correction/bonus
    return ledger


def rows_loaded(chunks, rows_needed, navail):
    """staged policy stops once cumulative rows cover rows_needed."""
    n = 0
    for c in chunks:
        n = min(n + c, navail)
        if n >= rows_needed:
            break
    return n


def access_stream(routes, nlayers, cur, nload):
    """layer-major chronological access order over nload verification rows."""
    for layer in range(nlayers):
        for pos in range(cur, cur + nload):
            for e in routes[pos].get(layer, []):
                yield layer * 256 + e


def lru_misses(accesses, cap):
    od = OrderedDict()
    miss = 0
    for k in accesses:
        if k in od:
            od.move_to_end(k)
        else:
            miss += 1
            od[k] = 1
            if len(od) > cap:
                od.popitem(last=False)
    return miss


def belady_misses(accesses, cap):
    """offline optimal (Belady) miss count on a fixed access list."""
    accesses = list(accesses)
    n = len(accesses)
    # next-use index for each position
    nxt = [0] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = accesses[i]
        nxt[i] = last.get(k, n)   # n == "never again"
        last[k] = i
    cache = {}                    # key -> next-use of its most recent access
    miss = 0
    for i, k in enumerate(accesses):
        if k in cache:
            cache[k] = nxt[i]
        else:
            miss += 1
            if len(cache) >= cap:
                # evict farthest next use
                ev = max(cache, key=lambda x: cache[x])
                del cache[ev]
            cache[k] = nxt[i]
    return miss


POLICIES = {
    "eager6": [W],
    "frontier2_4": [2, 4],
    "frontier1_2_3": [1, 2, 3],
    "sequential": [1] * W,
}


def main():
    d = torch.load("artifacts/ek0c_routes.pt", weights_only=False)
    routes = d["routes"]
    nlayers = d["n_layers"]
    p = fit_p()
    ledger = build_ledger(routes, p)
    ncyc = len(ledger)
    emitted_total = sum(a + 1 for _, a, _ in ledger)
    cold_cut = ncyc // 3          # first third = cold warmup interval

    out = {"classification": "FRONTIER0_SCREEN_V2", "n_positions": len(routes),
           "accept_p": p, "EA": EA, "cycles": ncyc,
           "emitted_tokens_total": emitted_total,
           "note": "shared ledger; chronological all-loads LRU + Belady lower "
                   "bound; accepted-path routes as speculative-candidate proxy "
                   "(demand direction vs real candidates UNKNOWN). demand+cache "
                   "sim only.",
           "by_store": {}}

    print(f"cycles={ncyc} emitted_tokens={emitted_total} "
          f"(mean {emitted_total/ncyc:.2f}/cyc)")
    print(f"{'store':>5} {'policy':>13} {'LRUmiss':>8} {'GB/tok':>7} "
          f"{'cold':>6} {'later':>6} {'Belady':>7} {'LRU/Bel':>7}")
    for gb in STORE_GB:
        cap = int(gb * 1e9 / BYTES)
        rec = {}
        for name, chunks in POLICIES.items():
            # build the full chronological access stream for this policy
            full = []
            cold_acc, later_acc = [], []
            cold_emit = later_emit = 0
            for ci, (cur, a, navail) in enumerate(ledger):
                rows_needed = min(a + 1, navail)
                nload = rows_loaded(chunks, rows_needed, navail)
                accs = list(access_stream(routes, nlayers, cur, nload))
                full.extend(accs)
                if ci < cold_cut:
                    cold_acc.extend(accs); cold_emit += a + 1
                else:
                    later_acc.extend(accs); later_emit += a + 1
            lru_m = lru_misses(full, cap)
            bel_m = belady_misses(full, cap)
            # per-interval LRU misses (fresh cache each interval for labelling)
            cold_m = lru_misses(cold_acc, cap)
            later_m = lru_misses(later_acc, cap)
            gb_tok = lru_m * BYTES / 1e9 / emitted_total
            rec[name] = {
                "lru_miss_total": lru_m,
                "belady_miss_total": bel_m,
                "lru_over_belady": round(lru_m / bel_m, 3) if bel_m else None,
                "GB_per_emitted_token": round(gb_tok, 4),
                "cold_GB_per_emit": round(cold_m * BYTES / 1e9 / max(cold_emit, 1), 4),
                "later_GB_per_emit": round(later_m * BYTES / 1e9 / max(later_emit, 1), 4),
                "accesses": len(full),
            }
            r = rec[name]
            print(f"{gb:>5} {name:>13} {lru_m:>8} {r['GB_per_emitted_token']:>7} "
                  f"{r['cold_GB_per_emit']:>6} {r['later_GB_per_emit']:>6} "
                  f"{bel_m:>7} {r['lru_over_belady']:>7}")
        base = rec["eager6"]["lru_miss_total"]
        for name in rec:
            rec[name]["reduction_vs_eager_pct"] = round(
                100 * (1 - rec[name]["lru_miss_total"] / base), 1) if base else 0
        out["by_store"][str(gb)] = rec

    Path("artifacts/frontier0_screen.json").write_text(json.dumps(out, indent=1))
    print("\nwrote artifacts/frontier0_screen.json")
    print("38GB paired (shared ledger) reduction vs eager, GB/emitted-token, "
          "LRU/Belady:")
    for n, v in out["by_store"]["38"].items():
        print(f"  {n:>13}: {v['reduction_vs_eager_pct']:>5}%  "
              f"{v['GB_per_emitted_token']} GB/tok  LRU/Bel={v['lru_over_belady']}")


if __name__ == "__main__":
    main()
