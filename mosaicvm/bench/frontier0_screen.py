"""FRONTIER-0 tier-1 screen (v3, canonical acceptance + real staged order).

Corrects v2 defects found in review, reconciled against the CANONICAL
acceptance definition in d1b_tail_robustness.prefix_metrics:

  depth = consecutive accepted draft decisions (0..W), excluding the exact
          root, stopping at first miss.  depth CAN be 0 (immediate rejection).
  E_A   = depth + 1   (the accepted drafts + 1 correction/bonus emission).
  Canonical mean E_A = 3.757  ->  mean depth = 2.757.

Fixes vs v2:
  1. Token semantics reconciled to the evaluator: draw depth (mean 2.757, may
     be 0), emit depth+1, advance cursor depth+1. Report per EMITTED token.
     The +1 bonus is a FREE emission (argmax of the last verified position's
     output); no extra forward pass. Free bonuses are counted and reconciled.
  2. Real staged access order: each verification STAGE is its own layer-major
     forward sweep over its positions; stages run in sequence; eager is one
     sweep over all W positions; sequential is W single-position sweeps.
  3. ONE continuous LRU across the whole run (no per-interval reset); interval
     stats come from binning misses by cycle within that single simulation.
  4. Two offline oracles on the same stream: mandatory-admission Belady AND
     bypass-capable Belady (may serve a miss via staging buffer without
     admitting). Neither is claimed as a universal floor; the bypass model's
     transient staging-buffer cost is noted, not free.

SIMULATION only: real routes as a speculative-candidate proxy (demand direction
vs true candidates UNKNOWN), no latency, no achieved throughput.
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
EA = 3.757            # canonical mean E_A = mean(depth)+1
MEAN_DEPTH = EA - 1.0  # 2.757
SEED = 20260920


def fit_q():
    """iid hit prob q s.t. mean depth = sum_{d=1..W} q^d == MEAN_DEPTH."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        q = (lo + hi) / 2
        if sum(q ** d for d in range(1, W + 1)) < MEAN_DEPTH:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2


def draw_depth(rng, q, navail):
    """depth in 0..navail; each draft accepted w.p. q, stop at first miss."""
    depth = 0
    while depth < navail and rng.random() < q:
        depth += 1
    return depth


def build_ledger(routes, q):
    """shared (cur, depth, navail); cursor advances depth+1 (drafts+bonus)."""
    rng = random.Random(SEED)
    ledger = []
    cur = 0
    n = len(routes)
    while cur < n - 1:
        navail = min(W, n - 1 - cur)      # positions verifiable ahead of cur
        if navail < 1:
            break
        depth = draw_depth(rng, q, navail)
        ledger.append((cur, depth, navail))
        cur += depth + 1                  # depth accepted + 1 correction/bonus
    return ledger


def stage_positions(chunks, rows_needed, navail):
    """list of per-stage position counts under a staged policy.

    Stages accumulate until cumulative >= rows_needed (the position that
    reveals the first rejection = depth+1), capped at navail.
    """
    stages = []
    cum = 0
    for c in chunks:
        take = min(c, navail - cum)
        if take <= 0:
            break
        stages.append(take)
        cum += take
        if cum >= rows_needed:
            break
    return stages          # e.g. sequential -> [1,1,...]; eager -> [navail]


def sweep(routes, nlayers, start, count):
    """layer-major access order for one forward sweep over `count` positions."""
    for layer in range(nlayers):
        for pos in range(start, start + count):
            for e in routes[pos].get(layer, []):
                yield layer * 256 + e


def build_accesses(routes, nlayers, ledger, chunks):
    """full chronological access stream + per-access cycle tag for a policy.

    eager verifies all navail positions in ONE sweep (you cannot know depth
    until you verify); staged runs a separate sweep per stage.

    BOUNDARY ACCOUNTING (v4 fix): on FULL acceptance (depth == navail) the
    cycle emits navail+1 tokens but the stage sweeps charge only navail
    positions. The bonus token is emitted from the last verified position's
    logits, but its hidden state / KV is NOT computed by that emission.
    Before the next cycle can attend over it, a BOUNDARY FORWARD on the bonus
    position is required. It is charged here as a single-position sweep
    appended after the verification sweeps, tagged to the same cycle.

    `cur` semantics: the position whose forward pass is still owed (the
    previous cycle's correction/bonus token). For depth < navail the
    correction position becomes the next cycle's `cur` and is charged there;
    for full acceptance the bonus position is charged by the boundary forward
    so the charged set stays contiguous.
    """
    accs, cyc_tag = [], []
    eager = chunks == [W]
    for ci, (cur, depth, navail) in enumerate(ledger):
        rows_needed = min(depth + 1, navail)
        if eager:
            plan = [navail]                    # one batched sweep over all W
        else:
            plan = stage_positions(chunks, rows_needed, navail)
        start = cur
        n_charged_rows = 0
        for cnt in plan:
            for k in sweep(routes, nlayers, start, cnt):
                accs.append(k)
                cyc_tag.append(ci)
            start += cnt
            n_charged_rows += cnt
        # full acceptance: charge the boundary forward on the bonus position
        if depth == navail:
            boundary_pos = cur + navail
            for k in sweep(routes, nlayers, boundary_pos, 1):
                accs.append(k)
                cyc_tag.append(ci)
            n_charged_rows += 1
    return accs, cyc_tag


def verify_boundary_accounting(routes, nlayers, ledger, chunks):
    """Deterministic check: every position needed for subsequent attention has
    a charged forward pass, and the charged set is contiguous from 0.

    Reconstructs the charged position set under the same scheduling rules as
    build_accesses (positions, not expert ids) and asserts:
      1. the charged set is exactly range(0, max+1) (no interior gaps),
      2. each cycle's next `cur` equals the previous charged end + 1.
    Runs on the first two full-acceptance cycles explicitly (deterministic,
    seed-independent) and over the whole ledger.
    """
    charged = []
    for cur, depth, navail in ledger:
        rows_needed = min(depth + 1, navail)
        plan = ([navail] if chunks == [W]
                else stage_positions(chunks, rows_needed, navail))
        start = cur
        for cnt in plan:
            charged.extend(range(start, start + cnt))
            start += cnt
        if depth == navail:
            charged.append(cur + navail)       # boundary forward
    s = set(charged)
    contiguous = s == set(range(max(charged) + 1)) if charged else True
    # per-cycle cursor consistency
    cursor_ok = True
    pos = 0
    for cur, depth, navail in ledger:
        if cur != pos:
            cursor_ok = False
        pos = cur + depth + 1
    # explicit two-cycle full-acceptance inspection
    full_cycles = [(i, c) for i, c in enumerate(ledger) if c[1] == c[2]]
    two_cycle_ok = True
    detail = []
    for i, (cur, depth, navail) in full_cycles[:2]:
        boundary_pos = cur + navail
        next_cur = cur + depth + 1
        ok = (boundary_pos in s) and (next_cur == boundary_pos + 1)
        two_cycle_ok = two_cycle_ok and ok
        detail.append({"cycle": i, "charged_boundary_pos": boundary_pos,
                       "next_cycle_cur": next_cur,
                       "boundary_charged": boundary_pos in s,
                       "cursor_is_next_after_boundary": next_cur == boundary_pos + 1})
    return {"n_charged_positions": len(s), "max_charged": max(charged) if charged else -1,
            "interior_gaps": sorted(set(range(max(charged) + 1)) - s) if charged else [],
            "contiguous": contiguous, "cursor_consistent": cursor_ok,
            "two_full_accept_cycles": detail, "two_cycle_check": two_cycle_ok,
            "all_ok": contiguous and cursor_ok and two_cycle_ok}


def lru_miss_by_cycle(accesses, cyc_tag, cap, ncyc):
    od = OrderedDict()
    per = [0] * ncyc
    total = 0
    for k, ci in zip(accesses, cyc_tag):
        if k in od:
            od.move_to_end(k)
        else:
            total += 1
            per[ci] += 1
            od[k] = 1
            if len(od) > cap:
                od.popitem(last=False)
    return total, per


def _next_use(accesses):
    n = len(accesses)
    nxt = [0] * n
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(accesses[i], n)
        last[accesses[i]] = i
    return nxt


def belady_mandatory(accesses, cap):
    nxt = _next_use(accesses)
    cache = {}
    miss = 0
    for i, k in enumerate(accesses):
        if k in cache:
            cache[k] = nxt[i]
        else:
            miss += 1
            if len(cache) >= cap:
                ev = max(cache, key=lambda x: cache[x])
                del cache[ev]
            cache[k] = nxt[i]
    return miss


def belady_bypass(accesses, cap):
    """optimal bypassing: on a full-cache miss, admit only if k is reused
    sooner than the farthest cached item; else serve via staging buffer and
    do not pollute the cache. Reduces future misses at a transient buffer cost.
    """
    nxt = _next_use(accesses)
    cache = {}
    miss = 0
    for i, k in enumerate(accesses):
        if k in cache:
            cache[k] = nxt[i]
            continue
        miss += 1
        if len(cache) < cap:
            cache[k] = nxt[i]
        else:
            far = max(cache, key=lambda x: cache[x])
            if cache[far] > nxt[i]:
                del cache[far]
                cache[k] = nxt[i]
            # else: bypass (no admission)
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

    q = fit_q()
    ledger = build_ledger(routes, q)
    ncyc = len(ledger)
    emitted_total = sum(depth + 1 for _, depth, _ in ledger)
    accepted_total = sum(depth for _, depth, _ in ledger)
    full_accept = sum(1 for _, depth, navail in ledger if depth == navail)
    cold_cut = ncyc // 3

    # BOUNDARY ACCOUNTING CHECK (deterministic, runs before any result).
    # Run for every policy's schedule (not just eager) so any policy that
    # under-charges positions fails the gate.
    chk_by_policy = {}
    for _pname, _chunks in POLICIES.items():
        chk_by_policy[_pname] = verify_boundary_accounting(
            routes, nlayers, ledger, _chunks)
    chk = chk_by_policy["eager6"]
    all_policies_ok = all(c["all_ok"] for c in chk_by_policy.values())
    print("boundary accounting check:")
    print(f"  charged positions={chk['n_charged_positions']} "
          f"max={chk['max_charged']} contiguous={chk['contiguous']} "
          f"cursor_ok={chk['cursor_consistent']} "
          f"two_cycle={chk['two_cycle_check']}")
    print(f"  all policies pass: {all_policies_ok}")
    if chk["interior_gaps"]:
        print(f"  interior gaps: {chk['interior_gaps']}")
    if not all_policies_ok:
        for _p, _c in chk_by_policy.items():
            if not _c["all_ok"]:
                print(f"  FAIL policy={_p}: contiguous={_c['contiguous']} "
                      f"cursor={_c['cursor_consistent']} "
                      f"two_cycle={_c['two_cycle_check']}")
        raise SystemExit("boundary accounting check FAILED for at least one "
                         "policy - refusing to emit results")


    out = {"classification": "FRONTIER0_SCREEN_V4",
           "n_positions": len(routes), "hit_q": q,
           "mean_E_A": emitted_total/ncyc, "mean_depth": accepted_total/ncyc,
           "cycles": ncyc, "emitted_tokens_total": emitted_total,
           "full_accept_cycles": full_accept,
           "boundary_check": chk,
           "boundary_check_by_policy": chk_by_policy,
           "note": "canonical prefix_metrics (E_A=depth+1, depth may be 0); "
                   "per-stage sweeps; one continuous LRU; mandatory+bypass "
                   "Belady; full-acceptance boundary forward CHARGED (v4); "
                   "gate run for ALL policies. real routes as speculative-"
                   "candidate proxy (direction UNKNOWN). sim only.",
           "by_store": {}}
    print(f"q(hit)={q:.4f}  cycles={ncyc}  mean_depth="
          f"{accepted_total/ncyc:.3f}  mean_E_A={emitted_total/ncyc:.3f}  "
          f"emitted={emitted_total}  full-accept_cycles={full_accept}")
    for gb in STORE_GB:
        cap = int(gb * 1e9 / BYTES)
        rec = {}
        for name, chunks in POLICIES.items():
            accs, tags = build_accesses(routes, nlayers, ledger, chunks)
            lru_m, per = lru_miss_by_cycle(accs, tags, cap, ncyc)
            bmand = belady_mandatory(accs, cap)
            bbyp = belady_bypass(accs, cap)
            cold_m = sum(per[:cold_cut])
            later_m = sum(per[cold_cut:])
            cold_emit = sum(dep + 1 for _, dep, _ in ledger[:cold_cut])
            later_emit = sum(dep + 1 for _, dep, _ in ledger[cold_cut:])
            rec[name] = {
                "lru_miss_total": lru_m,
                "belady_mandatory": bmand,
                "belady_bypass": bbyp,
                "lru_over_bypass": round(lru_m / bbyp, 3) if bbyp else None,
                "GB_per_emitted_token": round(lru_m * BYTES / 1e9 / emitted_total, 4),
                "cold_GB_per_emit": round(cold_m * BYTES / 1e9 / max(cold_emit, 1), 4),
                "later_GB_per_emit": round(later_m * BYTES / 1e9 / max(later_emit, 1), 4),
                "accesses": len(accs),
            }
            r = rec[name]
            print(f"{gb:>5} {name:>13} {lru_m:>8} {r['GB_per_emitted_token']:>7} "
                  f"{r['cold_GB_per_emit']:>7} {r['later_GB_per_emit']:>7} "
                  f"{bmand:>8} {bbyp:>7} {r['lru_over_bypass']:>7}", end="")
            print()
        base = rec["eager6"]["lru_miss_total"]
        for name in rec:
            rec[name]["reduction_vs_eager_pct"] = round(
                100 * (1 - rec[name]["lru_miss_total"] / base), 1) if base else 0
        out["by_store"][str(gb)] = rec

    Path("artifacts/frontier0_screen.json").write_text(json.dumps(out, indent=1))
    print("\nwrote artifacts/frontier0_screen.json")
    print("38GB (paired, per emitted token):")
    for n, v in out["by_store"]["38"].items():
        print(f"  {n:>13}: {v['reduction_vs_eager_pct']:>5}% vs eager  "
              f"{v['GB_per_emitted_token']} GB/tok  "
              f"LRU {v['lru_miss_total']} / BelMand {v['belady_mandatory']} / "
              f"BelByp {v['belady_bypass']}")


if __name__ == "__main__":
    main()
