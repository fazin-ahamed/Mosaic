"""Full four-stream state recorder with explicit teacher provenance.

Default: historical approximate FastStream teacher, FP16 storage, batch 12.
S1_SOURCE=1: source torch shim, BF16 storage, train/val only, batch 1.
S1_CATS selects manifest IDs; S1_N_GEN, S1_BATCH, S1_OUT configure capture.
Saves exactly N_GEN generating states/labels plus full h40/h42 input history.
State at absolute position t predicts tokens[t+1]; preserve terminal label.
"""
from __future__ import annotations

import json
import os
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from source_corpus_gen import (build_kernel_module, StreamLinear,
                               FP4_TABLE)


class FastStreamLinear(torch.nn.Module):
    """Recording-speed expert path: cache the unpacked, scale-pre-
    applied bf16 weight (exact pow2 scaling; values identical to the
    reference class), then act_quant (source E4M3 per-128) + one bf16
    tensor-core GEMM with fp32 accumulation. Accumulation class differs
    from the corpus's per-32 code-first einsum by fp32 rounding only —
    immaterial for proposer training data (drift envelope ~0.1 rel_l2)."""
    _ucache = {}
    _cap = 64

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def _weight(self):
        w = FastStreamLinear._ucache.get(self.prefix)
        if w is None:
            pw = StreamLinear._materialize(StreamLinear(self.prefix))
            wb = pw.view(torch.uint8).long()
            lut = FP4_TABLE.to(pw.device)
            w_c = torch.stack((lut[wb & 0x0F], lut[wb >> 4]),
                              dim=-1).flatten(-2)      # [N, K] codes
            s = pw.scale.float().repeat_interleave(32, dim=-1)
            w = (w_c * s).to(torch.bfloat16).contiguous()
            if len(FastStreamLinear._ucache) >= FastStreamLinear._cap:
                FastStreamLinear._ucache.pop(
                    next(iter(FastStreamLinear._ucache)))
            FastStreamLinear._ucache[self.prefix] = w
        return w

    def forward(self, x):
        import model as KM
        k = x.shape[-1]
        xq, sa = KM.act_quant(x, 128, KM.scale_fmt, KM.scale_dtype)
        xd = (xq.float().view(*xq.shape[:-1], k // 128, 128)
              * sa.unsqueeze(-1)).reshape(*xq.shape).to(torch.bfloat16)
        return (xd @ self._weight().t()).to(x.dtype)

PROMPTS = Path(os.environ.get("S1_PROMPTS", "artifacts/stage1_prompts.json"))
EXACT = os.environ.get("S1_SOURCE") == "1"
OUT = Path(os.environ.get("S1_OUT", "artifacts/d3_source_data" if EXACT else "/tmp/stage1_data"))
EXITS = (36, 38, 40, 42)
N_GEN = int(os.environ.get("S1_N_GEN", "40"))
BATCH = int(os.environ.get("S1_BATCH", "1" if EXACT else "12"))
MAX_NEW = int(os.environ.get("S1_MAX_NEW", "0"))
TELEMETRY = Path(os.environ.get(
    "S1_TELEMETRY", "artifacts/d3_scale1000_data/telemetry.jsonl"))
HC = 4


def prompt_splits(spec):
    membership = {}
    texts = {}
    for split, cats in spec["splits"].items():
        for cat in cats:
            if cat in membership:
                raise ValueError(f"Prompt appears in multiple splits: {cat}")
            membership[cat] = split
    ids = [p["category"] for p in spec["prompts"]]
    if len(set(ids)) != len(ids) or set(ids) != set(membership):
        raise ValueError("Prompt IDs must match the disjoint split manifest")
    for p in spec["prompts"]:
        text = " ".join(p["prompt"].lower().split())
        if text in texts and texts[text] != membership[p["category"]]:
            raise ValueError("Duplicate prompt text across splits")
        texts[text] = membership[p["category"]]
    return membership


def capture_span(prompt_length, n_gen):
    if prompt_length < 1 or n_gen < 1:
        raise ValueError("Positive prompt and generation lengths required")
    # State at S-1+p predicts generated token p; final label remains included.
    return prompt_length - 1, prompt_length + n_gen - 1


def _mem_row():
    """Low-overhead GPU+RSS snapshot. Never raises into capture."""
    row = {"t": time.time()}
    try:
        row["rss"] = int(Path("/proc/self/status").read_text().split(
            "VmRSS:")[1].split()[0]) * 1024
    except Exception:
        row["rss"] = None
    try:
        if torch.cuda.is_available():
            row["alloc"] = int(torch.cuda.memory_allocated())
            row["reserved"] = int(torch.cuda.memory_reserved())
            row["peak_alloc"] = int(torch.cuda.max_memory_allocated())
            row["peak_reserved"] = int(torch.cuda.max_memory_reserved())
            try:
                row["device"] = int(torch.cuda.device_memory_used())
            except Exception:
                row["device"] = None
            stats = torch.cuda.memory_stats() or {}
            row["inactive"] = int(stats.get(
                "inactive_split_bytes.all.current", 0))
        else:
            row.update(alloc=None, reserved=None, peak_alloc=None,
                       peak_reserved=None, device=None, inactive=None)
    except Exception:
        row.update(alloc=None, reserved=None, peak_alloc=None,
                   peak_reserved=None, device=None, inactive=None)
    a, r, d = row.get("alloc"), row.get("reserved"), row.get("device")
    row["frag"] = (r - a) if (a is not None and r is not None) else None
    row["dark"] = (d - r) if (d is not None and r is not None) else None
    return row


def _atomic_save(payload, dest: Path):
    """Write tmp, fsync, validate load, atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    chk = torch.load(tmp, map_location="cpu", weights_only=True)
    if chk.get("category") != payload["category"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"tmp validate failed: {dest}")
    os.replace(tmp, dest)
    dfd = os.open(str(dest.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)



def main():
    from mosaicvm.bench.d1b_tail_robustness import reset_decode_state
    if BATCH < 1 or N_GEN < 1:
        raise ValueError("Positive S1_BATCH/S1_N_GEN required")
    if EXACT and os.environ.get("SCG_SKIP_EXPERTS") == "1":
        raise ValueError("Source capture requires all experts")
    spec = json.loads(PROMPTS.read_text())
    membership = prompt_splits(spec)
    selected = os.environ.get("S1_CATS")
    cats = selected.split(",") if selected else list(membership)
    if len(set(cats)) != len(cats) or not set(cats) <= set(membership):
        raise ValueError("S1_CATS must be distinct manifest IDs")
    if EXACT and any(membership[c] == "test" for c in cats):
        raise ValueError("D3 source recorder accepts train/val only; use S1_CATS")
    prompts = [p for p in spec["prompts"] if p["category"] in cats]
    if EXACT:
        protected = set()
        for name in ("routes_overlap_corpus_3060.json", "g2_cert_prompts.json", "g2_calibration_prompts.json"):
            for entry in json.loads((Path("artifacts") / name).read_text())["prompts"]:
                protected.add(" ".join(entry["prompt"].lower().split()))
        if any(" ".join(p["prompt"].lower().split()) in protected for p in prompts):
            raise ValueError("G1/G2 prompts cannot enter D3 capture")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    storage_dtype = torch.bfloat16 if EXACT else torch.float16
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    sys.path.insert(0, str(snap / "inference"))
    sys.modules["kernel"] = build_kernel_module()
    import math as _math
    _fht = type(sys)("fast_hadamard_transform")
    _HAD = {}

    def _had(x):
        d = x.shape[-1]
        H = _HAD.get(d)
        if H is None or H.device != x.device:
            h = torch.tensor([[1.0]], device=x.device,
                             dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1),
                               torch.cat([h, -h], 1)])
            _HAD[d] = H = h / _math.sqrt(d)
        return (x.float() @ H).to(x.dtype)

    _fht.hadamard_transform = lambda x, scale=None: _had(x)
    sys.modules["fast_hadamard_transform"] = _fht
    import model as KM
    import dataclasses
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads((snap / "inference" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0
    args.max_batch_size = BATCH
    args.max_seq_len = 512
    args.dspark_block_size = 0
    args.n_mtp_layers = 0
    _orig = torch.empty

    def _patch(*a, **k):
        if k.get("dtype") == torch.float4_e2m1fn_x2:
            k["device"] = "meta"
        return _orig(*a, **k)

    torch.empty = _patch
    model = KM.Transformer(args)
    torch.empty = _orig
    for L, layer in enumerate(model.layers):
        for e_idx, ex in enumerate(layer.ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(ex, nm, (StreamLinear if EXACT else FastStreamLinear)(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}"))
    import gc
    gc.collect()
    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight", "head.weight") \
            else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()
    import types

    def _embed_fwd(self, x):
        return torch.nn.functional.embedding(
            x.cpu(), self.weight).to(x.device)

    def _head_fwd(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return torch.nn.functional.linear(
            x.cpu().float(), self.weight).to("cuda")

    model.embed.forward = types.MethodType(_embed_fwd, model.embed)
    model.head.forward = types.MethodType(_head_fwd, model.head)

    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    StreamLinear._snap = snap
    StreamLinear._index = index
    params = dict(model.named_parameters())

    def dequant_wo_a(w, s):
        rows, cols = w.shape
        return (w.float() * s.float()
                .repeat_interleave(rows // s.shape[0], 0)
                .repeat_interleave(cols // s.shape[1], 1)
                ).to(torch.bfloat16)

    for name, shard in index.items():
        if ".ffn.experts." in name and ".shared_experts." not in name:
            continue
        if name not in params:
            continue
        with safe_open(str(snap / shard), framework="pt") as f:
            t = f.get_tensor(name)
            if name.endswith("wo_a.weight"):
                s = f.get_tensor(name.replace(".weight", ".scale"))
                t = dequant_wo_a(t, s)
        with torch.no_grad():
            params[name].data.copy_(t)
    print("core loaded", flush=True)

    N_LAYERS = len(model.layers)
    REC = {"cur": []}
    keep = set(EXITS)

    def hook(L):
        def h(mod, args_, output):
            if L in keep:
                t = output[0] if isinstance(output, tuple) else output
                REC["cur"].append((L, t.detach().to(storage_dtype).cpu()))
        return h

    handles = [model.layers[L].register_forward_hook(hook(L))
               for L in EXITS]
    torch.set_default_device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)

    OUT.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in OUT.glob("*.pt")}
    if EXACT:
        for p in prompts:
            if p["category"] not in done:
                continue
            prior = torch.load(OUT / f"{p['category']}.pt", map_location="cpu", weights_only=True)
            if (prior.get("teacher_class") != "SOURCE_CLASS_TORCH_SHIM"
                    or prior.get("split") != membership[p["category"]]
                    or prior.get("prompt") != p["prompt"]
                    or prior.get("snapshot_revision") != snap.name
                    or len(prior.get("gen", [])) != N_GEN):
                raise ValueError(f"Incompatible existing capture: {p['category']}")
    todo = [p for p in prompts if p["category"] not in done]
    if MAX_NEW > 0:
        todo = todo[:MAX_NEW]
    print(f"{len(todo)} prompts to record "
          f"({len(done)} already done"
          f"{f', cap {MAX_NEW}' if MAX_NEW else ''})", flush=True)
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)

    for b0 in range(0, len(todo), BATCH):
        chunk = todo[b0:b0 + BATCH]
        B = len(chunk)
        reset_decode_state(model)
        t0 = time.time()
        encoded = [tok(p["prompt"], return_tensors="pt").input_ids[0]
                   for p in chunk]
        lens = [len(e) for e in encoded]
        min_len, max_len = min(lens), max(lens)
        total = max_len + N_GEN
        if total > args.max_seq_len:
            raise ValueError("Prompt plus generation exceeds max_seq_len")
        tokens = torch.full((B, total), -1, dtype=torch.long,
                            device="cuda")
        for i, e in enumerate(encoded):
            tokens[i, :len(e)] = e.to("cuda")
        prompt_mask = tokens != -1

        states = {i: {L: [] for L in EXITS} for i in range(B)}
        trajectories = {i: {L: [] for L in (40, 42)} for i in range(B)}
        prev_pos = 0
        for cur_pos in range(min_len, total):
            REC["cur"] = []
            with torch.no_grad():
                out_ids, _, _ = model(
                    tokens[:, prev_pos:cur_pos], prev_pos)
            # capture states of generating elements
            got = dict(REC["cur"])
            for i in range(B):
                _, end = capture_span(lens[i], N_GEN)
                valid = max(0, min(cur_pos, end) - prev_pos)
                if valid:
                    for L in (40, 42):
                        trajectories[i][L].append(got[L][i, :valid])
                if lens[i] <= cur_pos < lens[i] + N_GEN:  # element i is generating
                    for L in EXITS:
                        st = got[L][i, -1]      # state before the next generated token
                        states[i][L].append(st)
            nxt = torch.where(prompt_mask[:, cur_pos],
                              tokens[:, cur_pos], out_ids)
            tokens[:, cur_pos] = nxt
            prev_pos = cur_pos
        # save per prompt
        for i, p in enumerate(chunk):
            start, end = capture_span(lens[i], N_GEN)
            gen = tokens[i, lens[i]:lens[i] + N_GEN].tolist()
            assert len(gen) == N_GEN and all(len(states[i][L]) == N_GEN for L in EXITS)
            full = {L: torch.cat(trajectories[i][L]) for L in (40, 42)}
            assert all(t.shape[0] == end for t in full.values())
            payload = {
                "gen": gen,
                "category": p["category"], "split": membership[p["category"]],
                "S": lens[i], "state_position_start": start,
                "tokens": tokens[i, :lens[i] + N_GEN].tolist(),
                "h40": full[40], "h42": full[42],
                "snapshot_revision": snap.name, "script_sha256": script_hash,
                "prompt_sha256": hashlib.sha256(p["prompt"].encode()).hexdigest(),
                "teacher_class": "SOURCE_CLASS_TORCH_SHIM" if EXACT else "FASTSTREAM_APPROXIMATE",
                "storage_dtype": str(storage_dtype),
                "alignment": "h40[t] predicts tokens[t+1]; states[L][p] predicts gen[p]",
                "states": {L: torch.stack(states[i][L])
                           for L in EXITS},
                "prompt": p["prompt"],
            }
            dest = OUT / f"{p['category']}.pt"
            _atomic_save(payload, dest)
            mem = _mem_row()
            mem.update(category=p["category"], wall=time.time() - t0,
                       sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            with open(TELEMETRY, "a") as tf:
                tf.write(json.dumps(mem) + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        print(f"batch {b0//BATCH + 1}/"
              f"{(len(todo) + BATCH - 1)//BATCH}: "
              f"{[c['category'] for c in chunk]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    print("RECORDING DONE", flush=True)


if __name__ == "__main__":
    main()
