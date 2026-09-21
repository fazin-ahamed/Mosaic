"""Phase 2.13P-O: BranchPack accepted-depth oracle (AGENTS.md §28-30).

Uses measured self-consistency coverage from artifacts/psc_audit.json:
  c(b) = P(true verifier-accepted token within top-b of the target
          distribution at that position | prefix accepted)

Oracle tree model: at each depth d the true-path node carries b_d candidate
children (top-b_d of the distribution); siblings are leaf-only. Node budget
N = sum(b_d). Expected accepted depth:
  E[A] = sum_{d>=1} prod_{j<=d} c(b_j)

This is the PERFECT-DRAFTER ceiling: the drafter sees the target's own
distribution (stronger than any real DSpark). Real E[A_TREE] <= this.

Classification per §30 at T_block scenarios:
  rate = E[A] / T_block   ->  >=40 TREE_STRONG / 30-40 TREE_VIABLE /
                              20-30 TREE_INSUFFICIENT_CURRENT_COMPUTE /
                              <20  TREE_ACCEPTED_DEPTH_MISS
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ARTIFACT = Path("artifacts/branchpack_oracle.json")

N_VALUES = (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 124, 128)
T_BLOCKS = {          # seconds per verification block
    "historical_K56_1.4947s": 1.4947,   # BF16 core ceiling + historical MoE
    "improved_1.2s": 1.2,
    "fp8_moe_opt_0.9s": 0.9,
    "core_only_0.1024s": 0.1024,        # degenerate diagnostic: MoE free
}
# Measured WideVerify raw candidate rates (perfect residency, cool P2-est):
MEASURED_RAW = {32: 27.2, 40: 30.65, 48: 34.35, 56: 36.02}


def load_coverage() -> dict:
    d = json.loads(Path("artifacts/psc_audit.json").read_text())
    unc = d["coverage"]["unconditional"]
    cond = d["coverage"]["prefix_conditional_by_depth"]
    return unc, cond


def oracle_dp(c_by_b, n_budget: int, max_depth: int = 64) -> dict:
    """DP: best {b_d} allocation maximizing E[A] with sum(b_d) <= n_budget.

    c_by_b: {b: coverage}. State: (depth, nodes_used) -> best future
    expected accepted depth. c assumed depth-independent (unconditional
    estimate); depth-conditional variant handled by caller via c_fn.
    """
    bs = sorted(c_by_b)                     # e.g. [1, 2, 3, 4, 6, 8]
    # best[d][n] = expected accepted tokens from depth d onward with n nodes
    # E[A] contribution at depth d given survival prob p: c(b)*p adds...
    # backward DP over survival probability is linear: E(d, n) =
    #     max_b [ c(b) * (1 + E(d+1, n-b)) ]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def ev(d: int, n: int) -> float:
        if n < bs[0] or d > max_depth:
            return 0.0
        best = 0.0
        for b in bs:
            if b > n:
                break
            v = c_by_b[b] * (1.0 + ev(d + 1, n - b))
            best = max(best, v)
        return best

    # recover allocation
    alloc = []
    n, d = n_budget, 1
    while n >= bs[0] and d <= max_depth:
        best, best_b = -1.0, None
        for b in bs:
            if b > n:
                break
            v = c_by_b[b] * (1.0 + ev(d + 1, n - b))
            if v > best:
                best, best_b = v, b
        if best_b is None:
            break
        alloc.append(best_b)
        n -= best_b
        d += 1
    return {"expected_accepted_depth": ev(1, n_budget), "allocation": alloc}


def classify(rate: float) -> str:
    if rate >= 40:
        return "TREE_STRONG"
    if rate >= 30:
        return "TREE_VIABLE"
    if rate >= 25:
        return "TREE_NEAR"
    if rate >= 20:
        return "TREE_INSUFFICIENT_CURRENT_COMPUTE"
    return "TREE_ACCEPTED_DEPTH_MISS"


def run() -> dict:
    unc, cond = load_coverage()
    c_by_b = {int(b): float(v) for b, v in unc.items()}
    print("coverage c(b):", c_by_b)

    results = {"coverage_source": "artifacts/psc_audit.json (self-consistency, "
               "perfect-drafter ceiling)", "coverage": c_by_b, "oracle": {}}

    for n in N_VALUES:
        r = oracle_dp(c_by_b, n)
        rates = {}
        classes = {}
        for label, t in T_BLOCKS.items():
            rate = r["expected_accepted_depth"] / t
            rates[label] = rate
            classes[label] = classify(rate)
        results["oracle"][str(n)] = {
            "expected_accepted_depth": r["expected_accepted_depth"],
            "allocation_first20": r["allocation"][:20],
            "rates_tok_s": rates,
            "classes": classes,
        }

        # --- wide-tree decision curve (§30-31) ---
        ea = r["expected_accepted_depth"]
        t_req = ea / 30.0                    # block time budget for 30 output/s
        raw_req = 30.0 * n / ea              # required raw candidate positions/s
        # extrapolated T(N) from measured K32-56 raw rates (log-log fit)
        ks = sorted(MEASURED_RAW)
        xs = [math.log(k) for k in ks]
        ys = [math.log(MEASURED_RAW[k]) for k in ks]
        nx = len(xs)
        slope = (nx * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (nx * sum(x * x for x in xs) - sum(xs) ** 2)
        intercept = (sum(ys) - slope * sum(xs)) / nx
        raw_est = math.exp(intercept) * (n ** slope) if n not in MEASURED_RAW \
            else MEASURED_RAW[n]
        t_est = n / raw_est
        r_oracle_est = ea / t_est
        results["oracle"][str(n)].update({
            "t_block_max_for_30": t_req,
            "raw_positions_per_s_required": raw_req,
            "raw_est_extrapolated": None if n in MEASURED_RAW else raw_est,
            "t_block_est_extrapolated_s": t_est,
            "r_oracle_est_output_tok_s": r_oracle_est,
            "class_vs_est": classify(r_oracle_est),
        })
        src = "measured" if n in MEASURED_RAW else f"extrap slope={slope:.3f}"
        print(f"N={n:3d}: E[A]={ea:5.2f} T_max(30/s)={t_req:.3f}s "
              f"raw_req={raw_req:5.1f}/s T_est={t_est:.3f}s ({src}) "
              f"R_oracle={r_oracle_est:5.1f}/s [{classify(r_oracle_est)}]")

    # headline verdict at N=56
    n56 = results["oracle"]["56"]
    results["verdict_N56"] = {
        "expected_accepted_depth": n56["expected_accepted_depth"],
        "class_at_historical_block": n56["classes"]["historical_K56_1.4947s"],
        "note": "Perfect-drafter ceiling from self-consistency coverage. "
                "Real DSpark drafter lowers coverage (eta_draft < 1), so real "
                "E[A_TREE] is strictly below this. If the ceiling itself "
                "misses, BranchPack cannot reach Exact-30 at this block time "
                "regardless of drafter quality or scheduler sophistication.",
    }
    print(f"\nN=56 E[A] ceiling = {n56['expected_accepted_depth']:.2f} "
          f"-> {n56['classes']['historical_K56_1.4947s']} at T_block=1.4947s")

    ARTIFACT.write_text(json.dumps(results, indent=2))
    print(f"wrote {ARTIFACT}")
    return results


if __name__ == "__main__":
    run()
