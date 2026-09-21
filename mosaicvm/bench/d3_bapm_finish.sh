#!/bin/bash
# BAPM pilot eval. Waits for ps1001_bapm.pt, then runs the frozen
# candidate probe (HIST1001 + ps1001_bapm + d3 B1/2/4/8) on BOTH burned
# development authorities (CAP-DEV70 and SCALE-DEV3). Reports dEA vs the
# ps1001 baseline. Early-kill gate: CAP-DEV70 B4 dEA >= 0.10.
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
TRAIN=artifacts/d3_scale1000_train
BAPM=$TRAIN/ps1001_bapm.pt

echo "[bapm-finish] waiting for $BAPM ..."
while [ ! -f "$BAPM" ]; do sleep 30; done
sleep 5

echo "[bapm-finish] CAP-DEV70 (fresh-dev, burned)"
$PY mosaicvm/bench/d3_depth_branch_probe.py \
  --data artifacts/d3_capdev70_data \
  --manifest artifacts/d3_capdev70/prompts_capdev70.json --split val \
  --bridge "$BAPM" \
  --output "$TRAIN/bapm_capdev70.json"

echo "[bapm-finish] SCALE-DEV3 (burned)"
$PY mosaicvm/bench/d3_depth_branch_probe.py \
  --data artifacts/d3_scale1000_data \
  --manifest artifacts/d3_scale1000/prompts_scale1000.json --split val \
  --bridge "$BAPM" \
  --output "$TRAIN/bapm_scaledev3.json"

echo "[bapm-finish] COMPARISON"
$PY -c "
import json
def b4(f): return json.load(open(f))['by_B']['4']['canonical']
cap_old=b4('$TRAIN/capdev70_candidate.json')
cap_new=b4('$TRAIN/bapm_capdev70.json')
s3_old=b4('$TRAIN/depth_branch_probe_ps1001.json')
s3_new=b4('$TRAIN/bapm_scaledev3.json')
print(f\"CAP-DEV70 B4: ps1001 {cap_old['mean']:.3f} -> bapm {cap_new['mean']:.3f}  dEA {cap_new['mean']-cap_old['mean']:+.3f}\")
print(f\"SCALE-DEV3 B4: ps1001 {s3_old['mean']:.3f} -> bapm {s3_new['mean']:.3f}  dEA {s3_new['mean']-s3_old['mean']:+.3f}\")
import collections
pp=json.load(open('$TRAIN/bapm_capdev70.json'))['by_B']['4']['per_prompt']
dom=collections.defaultdict(list)
for c,r in pp.items(): dom[c.rsplit('_',1)[0]].append(r['E_A'])
print('CAP-DEV70 bapm per-domain B4:')
for k in sorted(dom): print(f'  {k:>9}: {sum(dom[k])/len(dom[k]):.2f}')
d=cap_new['mean']-cap_old['mean']
print('EARLY-KILL GATE dEA>=0.10:', 'PASS -> extend' if d>=0.10 else 'FAIL -> stop/rethink')
"
echo "[bapm-finish] COMPLETE"
