"""DD-1 val evaluation: boundary-position S1 + Top-k on DEV21."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
sys = None
import sys as _sys
_sys.path.insert(0, "."); _sys.path.insert(0, "mosaicvm/bench")
from mosaicvm.bench.d3_bridge import BridgeNet, load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON, pos_encode

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = BridgeNet(ck["hidden"], ck["depth"])
    model.load_state_dict(ck["model"]); model.eval()
    x_mean, x_std = ck["x_mean"], ck["x_std"]
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g, w_norm_g = w_head.cuda(), w_norm.cuda()
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path("artifacts/stage1_prompts.json").read_text()),
                         None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    correct = 0; topk = {1:0, 2:0, 4:0, 8:0, 16:0}
    rows = []
    for c in val:
        S = c["S"]; toks = c["tokens"]
        state = c[ck["state_field"]][S-1].float().flatten()
        feat = torch.cat([state, embed[toks[S]].float(),
                          embed[toks[S-1]].float(),
                          pos_encode(torch.tensor([S]))[0]])
        fn = (feat - x_mean) / x_std
        with torch.no_grad():
            z = model(fn.unsqueeze(0))[0]
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6) * w_norm_g
        lg = w_head_g @ zn
        pred = int(lg.argmax()); true = toks[S+1]
        hit = pred == true
        correct += hit
        for k in topk:
            topk[k] += int(true in lg.topk(k).indices.tolist())
        rows.append({"cat": c["category"], "hit": hit,
                     "pred": pred, "true": true})
    n = len(val)
    result = {
        "classification": "D3_DD1_VAL_BOUNDARY_EVAL",
        "state_field": ck["state_field"],
        "n_val": n,
        "S1": round(correct/n, 4),
        "Top_k": {str(k): round(v/n, 4) for k,v in sorted(topk.items())},
        "per_prompt": rows,
        "checkpoint": str(a.ckpt.resolve()),
        "checkpoint_sha256": hashlib.sha256(a.ckpt.read_bytes()).hexdigest(),
        "note": "single-position boundary ranking; NOT a chained drafter Top-k",
    }
    a.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k:v for k,v in result.items() if k!="per_prompt"}, indent=1))

if __name__ == "__main__":
    main()
