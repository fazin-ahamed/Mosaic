"""§39 canonical NEXT — real ≥256-position route-capture pipeline.

Runs on the Blackwell sandbox (model-property measurement). Captures
genuinely consecutive greedy-decode positions from real prompts through
the checkpoint's own model, recording per-position/per-layer routing.

Design (§1-§10 of the route-capture authorization):

  * REAL generation: 7 existing ROUTE_CAPTURE_PROMPTS, greedy decode,
    ≥256 consecutive positions each, no random token IDs, no wrapping.
  * RAW preservation: every (position, layer) pair records absolute
    position, token ID, top-6 expert IDs in execution order, route
    weights, router class (hash vs learned), prompt/capture ID.
  * FRONTIER analysis: sliding windows over the consecutive stream at
    N=64/80/96/112/124/128/144/160 — unique experts, U distribution
    (p50/p90/p95/max), expert frequency, neighboring-position overlap,
    traffic concentration, working-set estimates.
  * METADATA: model repo/revision, manifest hash, script hashes, test
    baseline, hardware, CUDA/torch versions, timestamp, thermal class,
    generation config.
  * SANITY GATES: 43 layers, 6 experts/decision, valid expert IDs,
    strictly consecutive positions, finite weights, no routing collapse.

Artifacts:
  artifacts/routes_real_256_raw.json      (full raw routing data)
  artifacts/routes_real_256_frontier.json (per-width analysis)
  artifacts/routes_real_256_frontier.md   (human summary)

Local tests (tests/test_route_capture.py) validate harness correctness
with synthetic fixtures — no checkpoint needed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

# ----------------------------------------------------------------- config --
N_POSITIONS = 256        # per prompt (≥160 required, 256 preferred)
N_LAYERS = 43
N_EXPERTS_PER_TOK = 6
N_ROUTED_EXPERTS = 256
WIDTHS = [64, 80, 96, 112, 124, 128, 144, 160]

ROUTER_CLASSES = {}      # filled by inspect_router_classes(): L0-2=hash


# ------------------------------------------------------------ utilities ----
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ------------------------------------------------------------ harness ------
def build_capture_harness(model, tokenizer, prompts, device):
    """Attach forward hooks to every Router/HashRouter module.

    Returns (hooks, capture_state) where capture_state accumulates
    per-position routing records during generation.
    """
    from mosaicvm.model.layers import HashRouter, Router

    state = {
        "records": [],           # list of per-position dicts
        "current_pos": None,     # absolute decode position
        "current_token": None,   # token being routed
        "capture_id": None,
    }
    handles = []

    # layer_idx -> router class mapping
    router_classes = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (Router, HashRouter)):
            layer_idx = None
            for part in name.split("."):
                if part.isdigit():
                    layer_idx = int(part)
            router_classes[layer_idx] = "hash" if isinstance(mod, HashRouter) else "learned"
            ROUTER_CLASSES.update(router_classes)

    def _hook(mod, args, output):
        scores, idx = output
        layer_idx = None
        # find layer index from the module's parent chain
        for name, m in model.named_modules():
            if m is mod:
                for part in name.split("."):
                    if part.isdigit():
                        layer_idx = int(part)
                break
        state["records"].append({
            "capture_id": state["capture_id"],
            "pos": state["current_pos"],
            "token": state["current_token"],
            "layer": layer_idx,
            "router_class": "hash" if isinstance(mod, HashRouter) else "learned",
            "expert_ids": idx.detach().cpu().tolist(),
            "weights": scores.detach().cpu().tolist(),
        })

    for name, mod in model.named_modules():
        if isinstance(mod, (Router, HashRouter)):
            handles.append(mod.register_forward_hook(_hook))

    return handles, state


def capture_one_prompt(model, tokenizer, prompt_text, capture_id,
                       n_positions, device, seed=42):
    """Generate n_positions consecutive tokens greedily, capturing routes.

    Returns (generated_ids, input_ids, records) where records are the
    raw per-position/per-layer routing dicts.
    """
    import torch

    state = {"records": [], "current_pos": None, "current_token": None,
             "capture_id": capture_id}

    from mosaicvm.model.layers import HashRouter, Router

    # map module -> layer_idx once
    mod_layer = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (Router, HashRouter)):
            layer_idx = None
            for part in name.split("."):
                if part.isdigit():
                    layer_idx = int(part)
            mod_layer[id(mod)] = layer_idx

    def _hook(mod, args, output):
        scores, idx = output
        state["records"].append({
            "capture_id": capture_id,
            "pos": state["current_pos"],
            "token": state["current_token"],
            "layer": mod_layer.get(id(mod)),
            "router_class": "hash" if isinstance(mod, HashRouter) else "learned",
            "expert_ids": idx.detach().cpu().tolist(),
            "weights": scores.detach().cpu().tolist(),
        })

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (Router, HashRouter)):
            handles.append(mod.register_forward_hook(_hook))

    try:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.manual_seed(seed)

        with torch.no_grad():
            # Generate one token at a time so we know exactly which position
            # and token each router call corresponds to
            generated = []
            prefix = input_ids
            for pos in range(n_positions):
                state["current_pos"] = pos
                # The routers see the LAST token of the current prefix
                state["current_token"] = prefix[0, -1].item()
                output = model.generate(prefix, max_new_tokens=1,
                                        do_sample=False, use_cache=True)
                new_token = output[0, -1].item()
                generated.append(new_token)
                # For the NEXT call, the router sees the NEW token
                prefix = output
                state["current_token"] = new_token
            # The last call generated a token but didn't route on it; we need
            # one more forward for the final position's routing
            # (records are captured during the generate call itself)
    finally:
        for h in handles:
            h.remove()

    return generated, input_ids[0].tolist(), state["records"]


# ------------------------------------------------------------ sanity ------
def validate_capture(records, n_layers=N_LAYERS, n_experts=N_EXPERTS_PER_TOK,
                     n_expert_ids=N_ROUTED_EXPERTS) -> dict:
    """Sanity gates per §9. Returns {ok, errors}."""
    errors = []

    # exactly 43 layers represented
    layers = {r["layer"] for r in records if r["layer"] is not None}
    if len(layers) != n_layers:
        errors.append(f"layer count {len(layers)} != {n_layers}")

    # positions strictly consecutive per capture
    by_capture = {}
    for r in records:
        by_capture.setdefault(r["capture_id"], set()).add(r["pos"])
    for cid, positions in by_capture.items():
        pos_list = sorted(positions)
        if pos_list and pos_list != list(range(pos_list[0], pos_list[-1] + 1)):
            errors.append(f"capture {cid}: non-consecutive positions")

    # 6 experts per decision (handle both [n,6] and flat [n*6] formats)
    for r in records:
        ids = r["expert_ids"]
        if isinstance(ids, list) and ids:
            if isinstance(ids[0], list):
                if any(len(row) != n_experts for row in ids):
                    errors.append(f"pos {r['pos']} layer {r['layer']}: "
                                  f"expert count != {n_experts}")
                flat = [e for row in ids for e in row]
            else:
                if len(ids) % n_experts != 0:
                    errors.append(f"pos {r['pos']} layer {r['layer']}: "
                                  f"flat expert list not multiple of {n_experts}")
                flat = ids
            if any(e < 0 or e >= n_expert_ids for e in flat):
                errors.append(f"pos {r['pos']} layer {r['layer']}: "
                              f"expert ID out of range")

    # weights finite
    for r in records:
        w = r["weights"]
        if isinstance(w, list) and w:
            flat_w = [x for row in w for x in row] if isinstance(w[0], list) else w
            if any(x != x or abs(x) == float("inf") for x in flat_w):
                errors.append(f"pos {r['pos']} layer {r['layer']}: "
                              f"non-finite route weight")

    # no routing collapse (all positions routing to same expert set)
    unique_sets = set()
    for r in records:
        ids = r["expert_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            unique_sets.add(tuple(sorted(tuple(row) for row in ids)))
    if len(unique_sets) <= 1:
        errors.append("routing collapse: all positions use identical expert sets")

    return {"ok": len(errors) == 0, "errors": errors[:20],
            "n_records": len(records), "n_layers": len(layers),
            "n_positions": sum(len(v) for v in by_capture.values())}


# ------------------------------------------------------------ analysis ----
def extract_expert_set(record) -> set:
    """Extract the set of expert IDs from a single routing record."""
    ids = record["expert_ids"]
    if isinstance(ids[0], list):
        return {e for row in ids for e in row}
    return set(ids)


def sliding_window_analysis(records, width):
    """Genuine sliding-window frontier analysis at the given width.

    Returns per-width metrics computed over consecutive windows of
    `width` positions from the capture stream.
    """
    # group records by (capture_id, pos) -> per-position layer routing
    by_pos = {}
    for r in records:
        key = (r["capture_id"], r["pos"])
        by_pos.setdefault(key, {})[r["layer"]] = r

    # sort positions within each capture
    captures = {}
    for (cid, pos), layers in by_pos.items():
        captures.setdefault(cid, []).append((pos, layers))

    results = {
        "width": width,
        "windows": [],
        "union_stats": {},
    }

    all_unions = []
    for cid in sorted(captures.keys()):
        positions = sorted(captures[cid])
        if len(positions) < width:
            continue  # not enough consecutive positions

        for start in range(0, len(positions) - width + 1):
            window_positions = positions[start:start + width]
            # union of all experts across all layers in this window
            window_experts = set()
            per_layer_experts = {}
            for pos, layers in window_positions:
                for layer_idx, rec in layers.items():
                    experts = extract_expert_set(rec)
                    window_experts |= experts
                    per_layer_experts.setdefault(layer_idx, set()).update(experts)

            results["windows"].append({
                "capture_id": cid,
                "start_pos": window_positions[0][0],
                "end_pos": window_positions[-1][0],
                "unique_experts": len(window_experts),
                "per_layer_unique": {str(k): len(v)
                                     for k, v in sorted(per_layer_experts.items())},
            })
            all_unions.append(len(window_experts))

    if all_unions:
        all_unions.sort()
        n = len(all_unions)
        results["union_stats"] = {
            "n_windows": n,
            "min": all_unions[0],
            "p50": all_unions[n // 2],
            "p90": all_unions[int(n * 0.9)] if n > 1 else all_unions[-1],
            "p95": all_unions[int(n * 0.95)] if n > 1 else all_unions[-1],
            "max": all_unions[-1],
            "mean": sum(all_unions) / n,
        }

    return results


def expert_frequency_analysis(records):
    """Expert frequency distribution across all captures/layers."""
    from collections import Counter
    freq = Counter()
    for r in records:
        ids = r["expert_ids"]
        if isinstance(ids[0], list):
            for row in ids:
                for e in row:
                    freq[e] += 1
        else:
            for e in ids:
                freq[e] += 1

    total_assignments = sum(freq.values())
    top10 = freq.most_common(10)
    top10_share = sum(c for _, c in top10) / total_assignments \
        if total_assignments else 0

    return {
        "total_assignments": total_assignments,
        "unique_experts_used": len(freq),
        "top10": [{"expert": e, "count": c,
                   "share": round(c / total_assignments, 4)}
                  for e, c in top10],
        "top10_share": round(top10_share, 4),
        "gini_coefficient": _gini(list(freq.values())),
    }


def _gini(values):
    """Gini coefficient of the expert frequency distribution."""
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    cumsum = 0
    for i, v in enumerate(values):
        cumsum += (i + 1) * v
    return (2 * cumsum) / (n * sum(values)) - (n + 1) / n


def neighboring_overlap(records):
    """Average expert-set overlap between consecutive positions (same layer)."""
    by_layer = {}
    for r in records:
        by_layer.setdefault((r["capture_id"], r["layer"]), []).append(r)

    overlaps = []
    for key, recs in by_layer.items():
        recs.sort(key=lambda r: r["pos"])
        for i in range(len(recs) - 1):
            a = extract_expert_set(recs[i])
            b = extract_expert_set(recs[i + 1])
            if a or b:
                overlaps.append(len(a & b) / len(a | b))
    return {
        "mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0,
        "n_comparisons": len(overlaps),
    }


def working_set_estimate(records, width, hot8_mb=8.39, p4_mb=4.19):
    """Estimated GPU bytes for the experts touched by a width-N window."""
    analysis = sliding_window_analysis(records, width)
    if not analysis["union_stats"]:
        return None
    u_mean = analysis["union_stats"]["mean"]
    return {
        "width": width,
        "mean_unique_experts": round(u_mean, 1),
        "hot8_est_gb": round(u_mean * hot8_mb / 1e3, 2),
        "p4_est_gb": round(u_mean * p4_mb / 1e3, 2),
        "mixed_a_est_gb": round(u_mean * hot8_mb / 1e3, 2),  # down=hot8
    }


def full_frontier_analysis(records, widths=None):
    """Complete frontier analysis across all widths."""
    if widths is None:
        widths = WIDTHS
    result = {
        "frequency": expert_frequency_analysis(records),
        "neighboring_overlap": neighboring_overlap(records),
        "widths": {},
    }
    for w in widths:
        result["widths"][str(w)] = sliding_window_analysis(records, w)
    return result
