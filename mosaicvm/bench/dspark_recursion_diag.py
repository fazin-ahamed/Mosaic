"""Diagnostic: why does dspark_recursion's first draft fail at depth 0?

Loads the same components, prefills ONE prompt, drafts ONE block two
ways, and prints decoded tokens:
  A) via dspark_recursion.forward_spec_greedy  (the failing path)
  B) via dspark_wire.forward_spec              (the known-good path)
plus the target's greedy token and the draft's top-5.
"""
import json
import os
import sys

sys.path.insert(0, "/marimo/MosaicVM")
import torch

from mosaicvm.bench import dspark_wire as dw
from mosaicvm.bench.dspark_wire import (
    MAIN_GPU_GIB, TARGET_LAYERS, load_mtp_tensors, precompute_freqs_cis,
    DSparkStage, FP4_TABLE_VALUES)

t0 = __import__("time").time()


def log(m):
    print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}", flush=True)


model_dir = __import__("huggingface_hub").snapshot_download(
    "deepseek-ai/DeepSeek-V4-Flash-0731")
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer)

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, config=cfg, torch_dtype=torch.bfloat16,
    device_map="auto", max_memory={0: f"{MAIN_GPU_GIB}GiB", "cpu": "100GiB"},
    offload_folder="/tmp/offload", trust_remote_code=True)
model.eval()

captured = []
h = [model.model.layers[li].register_forward_hook(
    lambda m, a, o: captured.append(o.detach().mean(dim=2)))
    for li in TARGET_LAYERS]

from safetensors import safe_open
idx = json.load(open(os.path.join(model_dir,
                                  "model.safetensors.index.json")))
with safe_open(os.path.join(model_dir, idx["weight_map"]["head.weight"]),
               framework="pt", device="cpu") as f:
    head_w = f.get_tensor("head.weight").to("cuda")
# NOT tied: checkpoint ships separate embed.weight + head.weight
embed_weight = model.model.embed_tokens.weight.data
if embed_weight.device.type != "cuda":
    with safe_open(os.path.join(model_dir,
                                idx["weight_map"]["embed.weight"]),
                   framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("embed.weight").to("cuda")
dev = head_w.device

fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
raw = load_mtp_tensors(model_dir, dev)
fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(3)]
log(f"loaded ({t0 and __import__('time').time()-t0:.0f}s)")


def fs(input_ids, main_hidden, start_pos):
    """EXACT copy of both scripts' shared forward_spec."""
    h, main_x = stages[0].forward_embed(main_hidden, input_ids,
                                        embed_weight)
    for st in stages:
        h = st(h, start_pos, main_x)
    if start_pos == 0:
        return None
    return stages[-1].forward_head(h, input_ids, head_w)


prompt = "def fibonacci(n):"
input_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
n = input_ids.shape[1]
with torch.no_grad():
    out = model(input_ids, use_cache=True)
captured.clear()
# recompute mh via a second forward WITH hooks capturing
with torch.no_grad():
    _ = model(input_ids, use_cache=False)
mh_full = torch.cat(captured, dim=-1)
captured.clear()
g_next = out.logits[0, -1].argmax().item()
log(f"prompt len {n}, g_next={g_next} "
    f"({tok.decode([g_next])!r})")

# drafter prefill
with torch.no_grad():
    fs(input_ids[:, -1:], mh_full, 0)

# draft ONE block at start_pos = n-1
last_tok = input_ids[0, -1].item()
last_mh = mh_full[0, -1:]
with torch.no_grad():
    res = fs(torch.tensor([[last_tok]], device=dev),
             last_mh.unsqueeze(0), n - 1)

g_ids = res["greedy_ids"][0, 1:].tolist()
s_ids = res["sampled_ids"][0, 1:].tolist()
log(f"greedy draft : {g_ids} -> {tok.decode(g_ids)!r}")
log(f"sampled draft: {s_ids} -> {tok.decode(s_ids)!r}")
top_ids, top_scores = res["greedy_topk"][0]
log(f"draft top-5  : {top_ids[:5]} scores {top_scores[:5]}")
log(f"target tok at pos n: {g_next}")

# also decode top-5 to see if they're plausible
log(f"top-5 decoded: {[tok.decode([t]) for t in top_ids[:5]]!r}")

# accept prefix vs g_next
acc = 1 if g_ids[0] == g_next else 0
log(f"depth-0 hit: {g_ids[0] == g_next}")
