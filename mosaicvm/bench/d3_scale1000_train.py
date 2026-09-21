"""SCALE1000 trainer (roadmap sections 23-26) - FP32 out-of-core.

Primary recipe is numerically identical to the earlier in-RAM FP32
scale trainers; only the *execution* changed: the 1001-prompt union
(32,032 rows) now lives in disk-backed FP32 mmap tensors instead of a
Python list + torch.stack, so anonymous RSS stays well under host RAM.

Stage A (--stage hist): HIST-1 architecture (DDHistNet H1024/D3/aux256,
prev-aux, full-vocab CE + stored margin + KL, AdamW 1e-3/wd 1e-4,
cosine 600, CHUNK 512, clip 1.0, seed 104729) on the union:
84 original + 126 scale210 + 490 scale700 + 301 scale1000 = 1001 prompts.

Stage B (--stage ws): frozen HIST-1001 trunk; train only Q: z_rms -> q256
(init V_r^T), CE + 0.5 KL (WS-1 recipe).

Out-of-core details:
  - Raw features written once into cache/xm.f32, cache/xa.f32 (FP32 mmap).
  - Normalization mean/std computed in FP32 over the mmap; normalized data
    written to cache/xmn.f32, cache/xan.f32 (hist stage only).
  - Build is cached: cache/manifest.json records row count + provenance
    SHAs; a matching cache is reused (cheap restart after reboot/OOM).
  - hist training checkpoints every CKPT_EVERY steps to ckpt/hist_ckpt.pt
    (atomic tmp+rename) with full model/opt/sched/step/RNG; resumes on
    relaunch. A per-step MemAvailable watchdog checkpoints and exits
    cleanly before the host is endangered (cgroup MemoryMax is the backstop).

The bf16 variant is preserved separately as d3_scale1000_train_bf16.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      TEACHER, AUDIT)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON, POS_DIM

S210_CAP = Path("artifacts/d3_scale_data")
S210_TEACHER = Path("artifacts/d3_scale_teacher")
S210_MANIFEST = Path("artifacts/d3_scale/prompts_scale.json")
S700_CAP = Path("artifacts/d3_scale700_data")
S1000_CAP = Path("artifacts/d3_scale1000_data")
S700_TEACHER = Path("artifacts/d3_scale700_teacher")
S1000_TEACHER = Path("artifacts/d3_scale1000_teacher")
S700_MANIFEST = Path("artifacts/d3_scale700/prompts_scale700.json")
S1000_MANIFEST = Path("artifacts/d3_scale1000/prompts_scale1000.json")
ORIG_CAP = Path("artifacts/d3_data_105")
ORIG_MANIFEST = Path("artifacts/stage1_prompts.json")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
OUTDIR = Path("artifacts/d3_scale1000_train")
CACHE = OUTDIR / "cache"
CKPT = OUTDIR / "ckpt"

D_STATE = 4 * 4096
D_TOK = 4096
DM = D_STATE + 2 * D_TOK + POS_DIM   # 24640 main feature dim
DA = D_STATE                          # 16384 aux (prev-state) dim
N_ROWS = 1001 * HORIZON               # 32032
MEM_FLOOR_GB = 3.0                     # watchdog: checkpoint+exit below this
CKPT_EVERY = 50


# ---------- row emission (numerically identical to prior scale trainers) ----

def emit_core(rec, cap, embed):
    """Return (xm_rows[K,DM] fp32, xa_rows[K,DA] fp32, labels[list])."""
    S, K = rec["S"], rec["K"]
    toks = rec["tokens"]
    states = cap["h40"][S - 2:S + K - 1].float()
    xm_rows, xa_rows, labels = [], [], []
    for j in range(K):
        t = S + j
        row = rec["decisions"][j]
        y = row["true_token"]
        cur = states[j + 1].flatten()
        prev = states[j].flatten()
        mf = make_dd_feat(embed, cur, toks[t], toks[t - 1], t)
        xm_rows.append(mf)
        xa_rows.append(prev)
        labels.append(
            {"y": y, "cat": rec["category"], "j": j,
             "top_idx": row["gold"]["top_idx"],
             "top_val": row["gold"]["top_val"],
             "comps": [i for i in row["gold"]["top_idx"]
                       if i != y][:8]})
    # boundary leak check (prev state must not equal the first gold future)
    h40_S = cap["h40"][S].float().flatten()
    prev0 = cap["h40"][S - 2].float().flatten()
    if torch.allclose(prev0, h40_S, atol=1e-6):
        raise RuntimeError(f"LEAK {rec['category']}")
    return (torch.stack(xm_rows), torch.stack(xa_rows), labels)


def sources():
    """Ordered union sources. Order fixes row layout; DO NOT reorder."""
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS", "orig84 teacher audit not PASS"
    out = [("orig84", TEACHER, ORIG_CAP, ORIG_MANIFEST,
            sorted(ver["record_hashes_full"]))]
    for tag, td, cd, mf in (
            ("scale126", S210_TEACHER, S210_CAP, S210_MANIFEST),
            ("scale490", S700_TEACHER, S700_CAP, S700_MANIFEST),
            ("scale301", S1000_TEACHER, S1000_CAP, S1000_MANIFEST)):
        audit = json.loads((td / "audit.json").read_text())
        assert audit["overlap_checks"]["all_empty_required"], f"{tag} overlap"
        out.append((tag, td, cd, mf, sorted(audit["records"])))
    return out


def provenance_sha():
    """Hash of all teacher audits + manifests; changes invalidate cache."""
    h = hashlib.sha256()
    for p in (AUDIT, S210_TEACHER / "audit.json", S700_TEACHER / "audit.json",
              S1000_TEACHER / "audit.json", ORIG_MANIFEST, S210_MANIFEST,
              S700_MANIFEST, S1000_MANIFEST):
        h.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return h.hexdigest()


# ---------- mmap helpers ----------

def open_mmap(path, rows, cols):
    """FP32 tensor backed by a shared (MAP_SHARED) file; created if absent."""
    t = torch.from_file(str(path), shared=True, size=rows * cols,
                         dtype=torch.float32)
    return t.view(rows, cols)


def build_union_mmap():
    """Write raw features into cache/xm.f32, xa.f32; return (xm, xa, labels).

    Reuses an existing cache whose manifest matches current provenance.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    prov = provenance_sha()
    man = CACHE / "manifest.json"
    counts = {}
    if man.exists():
        m = json.loads(man.read_text())
        if (m.get("provenance_sha") == prov and m.get("n_rows") == N_ROWS
                and (CACHE / "xm.f32").exists()
                and (CACHE / "xa.f32").exists()
                and (CACHE / "labels.pt").exists()):
            xm = open_mmap(CACHE / "xm.f32", N_ROWS, DM)
            xa = open_mmap(CACHE / "xa.f32", N_ROWS, DA)
            labels = torch.load(CACHE / "labels.pt", weights_only=False)
            print(f"reuse cached raw union: {N_ROWS} rows "
                  f"({m.get('counts')})", flush=True)
            return xm, xa, labels

    xm = open_mmap(CACHE / "xm.f32", N_ROWS, DM)
    xa = open_mmap(CACHE / "xa.f32", N_ROWS, DA)
    embed = load_embed()
    labels = []
    off = 0
    for tag, td, cd, mf, cats in sources():
        spec = json.loads(Path(mf).read_text())
        caps = load_captures(str(cd), spec, None, HORIZON)
        by = {c["category"]: c for c in caps}
        base = off
        for cat in cats:
            rec = torch.load(td / f"{cat}.pt", map_location="cpu",
                             weights_only=True)
            rm, ra, labs = emit_core(rec, by[cat], embed)
            k = rm.shape[0]
            xm[off:off + k] = rm
            xa[off:off + k] = ra
            labels.extend(labs)
            off += k
        counts[tag] = off - base
        del caps, by
        print(f"  built {tag}: {counts[tag]} rows (total {off})", flush=True)
    assert off == N_ROWS, (off, N_ROWS)
    torch.save(labels, CACHE / "labels.pt")
    man.write_text(json.dumps(
        {"provenance_sha": prov, "n_rows": N_ROWS, "counts": counts,
         "DM": DM, "DA": DA}, indent=1))
    print(f"union built: {N_ROWS} rows ({counts})", flush=True)
    return xm, xa, labels


def normalize_mmap(xm, xa):
    """FP32 mean/std over raw mmap; write normalized xmn/xan mmap.

    Reuses cached normalized files when provenance matches.
    """
    prov = provenance_sha()
    nman = CACHE / "norm_manifest.json"
    if (nman.exists() and (CACHE / "xmn.f32").exists()
            and (CACHE / "xan.f32").exists()):
        m = json.loads(nman.read_text())
        if m.get("provenance_sha") == prov and m.get("n_rows") == N_ROWS:
            st = torch.load(CACHE / "norm_stats.pt", weights_only=True)
            xmn = open_mmap(CACHE / "xmn.f32", N_ROWS, DM)
            xan = open_mmap(CACHE / "xan.f32", N_ROWS, DA)
            print("reuse cached normalized union", flush=True)
            return xmn, xan, st

    xm_mean = xm.mean(0)
    xm_std = xm.std(0).clamp_min(1e-4)
    xa_mean = xa.mean(0)
    xa_std = xa.std(0).clamp_min(1e-4)
    xmn = open_mmap(CACHE / "xmn.f32", N_ROWS, DM)
    xan = open_mmap(CACHE / "xan.f32", N_ROWS, DA)
    CH = 1024
    for i in range(0, N_ROWS, CH):
        sl = slice(i, min(i + CH, N_ROWS))
        xmn[sl] = (xm[sl] - xm_mean) / xm_std
        xan[sl] = (xa[sl] - xa_mean) / xa_std
    st = {"xm_mean": xm_mean, "xm_std": xm_std,
          "xa_mean": xa_mean, "xa_std": xa_std}
    torch.save(st, CACHE / "norm_stats.pt")
    nman.write_text(json.dumps({"provenance_sha": prov,
                                "n_rows": N_ROWS}, indent=1))
    print("normalized union written", flush=True)
    return xmn, xan, st


def label_tensors(labels):
    y_idx = torch.tensor([m["y"] for m in labels])
    max_c = max(len(m["comps"]) for m in labels)
    comp_idx = torch.tensor(
        [m["comps"] + [m["y"]] * (max_c - len(m["comps"])) for m in labels])
    comp_mask = torch.tensor(
        [[1.0] * len(m["comps"]) + [0.0] * (max_c - len(m["comps"]))
         for m in labels])
    top_idx_t = torch.tensor([m["top_idx"] for m in labels])
    p_t = torch.stack([F.softmax(torch.tensor(m["top_val"]), dim=0)
                       for m in labels])
    return y_idx, comp_idx, comp_mask, top_idx_t, p_t


# ---------- checkpoint + watchdog ----------

def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def save_ckpt(path, step, model, opt, sched):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "py_rng": torch.random.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
    }, tmp)
    os.replace(tmp, path)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("hist", "ws"), required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    xm, xa, labels = build_union_mmap()
    n = xm.shape[0]
    assert n == N_ROWS, (n, N_ROWS)
    y_idx, comp_idx, comp_mask, top_idx_t, p_t = label_tensors(labels)
    y_g = y_idx.cuda(); comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()
    del labels

    w_head, w_norm = load_head()

    if a.stage == "hist":
        xmn, xan, st = normalize_mmap(xm, xa)
        xm_mean, xm_std = st["xm_mean"], st["xm_std"]
        xa_mean, xa_std = st["xa_mean"], st["xa_std"]
        print(f"training HIST-1001 on {n} rows "
              f"({time.time()-t0:.0f}s)", flush=True)

        model = DDHistNet(1024, 3, 256)
        n_params = sum(p_.numel() for p_ in model.parameters())
        w_head_g = w_head.cuda().requires_grad_(False)
        w_norm_g = w_norm.cuda().requires_grad_(False)
        MU, LAM_M, LAM_KL = 1.0, 1.0, 0.5
        CHUNK = 512

        def chunk_losses(zc, sl):
            zc_g = zc.cuda()
            zn = zc_g * torch.rsqrt(
                zc_g.square().mean(-1, keepdim=True) + 1e-6) * w_norm_g
            lg = zn @ w_head_g.t()
            ce = F.cross_entropy(lg.float(), y_g[sl], reduction="none")
            l_y = lg.gather(1, y_g[sl].unsqueeze(1)).squeeze(1)
            m = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
            m_rows = (F.softplus(MU - m) * cmask_g[sl]).sum(-1) \
                / cmask_g[sl].sum(-1).clamp_min(1.0)
            l16 = lg.gather(1, top_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log() - logp)).sum(-1)
            return ce + LAM_M * m_rows + LAM_KL * kl

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=600, eta_min=1e-3 * 0.02)

        start = 1
        ckpt_path = CKPT / "hist_ckpt.pt"
        if ckpt_path.exists():
            c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            torch.random.set_rng_state(c["py_rng"])
            if c.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(c["cuda_rng"])
            start = c["step"] + 1
            print(f"resumed from step {c['step']} ({time.time()-t0:.0f}s)",
                  flush=True)

        for step in range(start, 601):
            avail = mem_available_gb()
            if avail < MEM_FLOOR_GB:
                save_ckpt(ckpt_path, step - 1, model, opt, sched)
                print(f"WATCHDOG: MemAvailable {avail:.1f}GB < "
                      f"{MEM_FLOOR_GB}GB; checkpointed step {step-1}, exit",
                      flush=True)
                return
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                zc = model(xmn[sl], xan[sl])
                part = chunk_losses(zc, sl).mean() * (zc.shape[0] / n)
                part.backward()
                total += part.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % CKPT_EVERY == 0:
                save_ckpt(ckpt_path, step, model, opt, sched)
            if step % 50 == 0 or step == 600:
                with torch.no_grad():
                    accs = []
                    for i in range(0, n, 4096):
                        sl = slice(i, min(i + 4096, n))
                        zg = model(xmn[sl], xan[sl]).cuda()
                        zn = zg * torch.rsqrt(
                            zg.square().mean(-1, keepdim=True) + 1e-6) \
                            * w_norm_g
                        lg = zn @ w_head_g.t()
                        accs.append((lg.argmax(-1) == y_g[sl])
                                    .float().mean().item())
                print(f"step {step}: L {total:.4f} argmax "
                      f"{sum(accs)/len(accs):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        torch.save({
            "model": model.state_dict(), "hidden": 1024,
            "depth": 3, "aux_dim": 256, "hist_mode": "prev",
            "xm_mean": xm_mean, "xm_std": xm_std,
            "xa_mean": xa_mean, "xa_std": xa_std,
            "state_field": "h40",
            "args": {"stage": "hist", "corpus": "1001"},
            "seed": 104729, "n_params": n_params, "n_rows": n,
            "row_counts": json.loads((CACHE / "manifest.json")
                                     .read_text())["counts"],
            "teacher_audit_sha256": hashlib.sha256(
                AUDIT.read_bytes()).hexdigest(),
            "scale210_teacher_audit_sha256": hashlib.sha256(
                (S210_TEACHER / "audit.json").read_bytes()).hexdigest(),
            "scale700_teacher_audit_sha256": hashlib.sha256(
                (S700_TEACHER / "audit.json").read_bytes()).hexdigest(),
            "scale1000_teacher_audit_sha256": hashlib.sha256(
                (S1000_TEACHER / "audit.json").read_bytes()).hexdigest(),
        }, OUTDIR / "hist1001.pt")
        print(f"wrote {OUTDIR}/hist1001.pt ({time.time()-t0:.0f}s)",
              flush=True)
        return

    # ---- stage ws ----
    ck = torch.load(OUTDIR / "hist1001.pt", map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
    print(f"caching z_rms ({time.time()-t0:.0f}s)", flush=True)

    bk = torch.load(BASIS, map_location="cpu", weights_only=True)
    E_g = bk["E_r"].cuda()
    V_r = bk["V_r"]
    w_norm_g = w_norm.cuda()

    zr_cache = torch.empty(n, 4096, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            xmn_c = (xm[sl] - xm_mean) / xm_std
            xan_c = (xa[sl] - xa_mean) / xa_std
            z = hist(xmn_c, xan_c).cuda()
            zr_cache[sl] = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
    print(f"z_rms cached ({time.time()-t0:.0f}s)", flush=True)

    Q = nn.Parameter(V_r.t().clone().cuda())
    opt = torch.optim.AdamW([Q], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    LAM_KL = 0.5
    CHUNK = 512
    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            q = zr_cache[sl] @ Q.t()
            lg = q @ E_g.t()
            ce = F.cross_entropy(lg.float(), y_g[sl], reduction="none")
            l16 = lg.gather(1, top_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log() - logp)).sum(-1)
            part = (ce + LAM_KL * kl).mean() * (q.shape[0] / n)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_([Q], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == 600:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 4096):
                    sl = slice(i, min(i + 4096, n))
                    lg = (zr_cache[sl] @ Q.t()) @ E_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} argmax "
                  f"{sum(accs)/len(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"Q": Q.detach().cpu(), "rank": 256,
                "hist1001": str(OUTDIR / "hist1001.pt"),
                "hist1001_sha256": hashlib.sha256(
                    (OUTDIR / "hist1001.pt").read_bytes()).hexdigest(),
                "basis_sha256": hashlib.sha256(
                    BASIS.read_bytes()).hexdigest(),
                "n_rows": n, "seed": 104729},
               OUTDIR / "ws1001.pt")
    print(f"wrote {OUTDIR}/ws1001.pt ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
