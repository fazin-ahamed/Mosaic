#!/bin/bash
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python; T=artifacts/d3_scale1000_train
while [ ! -f "$T/ps1001_ofdm.pt" ]; do sleep 30; done
sleep 5
$PY mosaicvm/bench/d3_depth_branch_probe.py --data artifacts/d3_capdev70_data \
  --manifest artifacts/d3_capdev70/prompts_capdev70.json --split val \
  --bridge "$T/ps1001_ofdm.pt" --output "$T/ofdm_capdev70.json"
$PY mosaicvm/bench/d3_depth_branch_probe.py --data artifacts/d3_scale1000_data \
  --manifest artifacts/d3_scale1000/prompts_scale1000.json --split val \
  --bridge "$T/ps1001_ofdm.pt" --output "$T/ofdm_scaledev3.json"
$PY -c "
import json,collections
def b4(f): return json.load(open(f))['by_B']['4']['canonical']
co=b4('$T/capdev70_candidate.json'); cn=b4('$T/ofdm_capdev70.json')
so=b4('$T/depth_branch_probe_ps1001.json'); sn=b4('$T/ofdm_scaledev3.json')
print(f'CAP-DEV70 B4: ps1001 {co[\"mean\"]:.3f} -> ofdm {cn[\"mean\"]:.3f}  dEA {cn[\"mean\"]-co[\"mean\"]:+.3f}')
print(f'SCALE-DEV3 B4: ps1001 {so[\"mean\"]:.3f} -> ofdm {sn[\"mean\"]:.3f}  dEA {sn[\"mean\"]-so[\"mean\"]:+.3f}')
pp=json.load(open('$T/ofdm_capdev70.json'))['by_B']['4']['per_prompt']
dom=collections.defaultdict(list)
for c,r in pp.items(): dom[c.rsplit('_',1)[0]].append(r['E_A'])
old={'agent':2.1,'coding':2.7,'convo':5.6,'json':4.9,'longform':1.5,'math':3.2,'retrieval':6.3}
print('CAP-DEV70 ofdm per-domain B4:')
for k in sorted(dom):
    m=sum(dom[k])/len(dom[k]); print(f'  {k:>9}: {m:.2f} (was {old[k]:.2f}, {m-old[k]:+.2f})')
d=cn['mean']-co['mean']
print('OFDM GATE dEA>=0.10:', 'PASS' if d>=0.10 else 'FAIL')
"
echo "[ofdm-finish] COMPLETE"
