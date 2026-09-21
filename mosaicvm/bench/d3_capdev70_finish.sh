#!/bin/bash
# CAP-DEV70 frozen-candidate evaluation. Waits for 70/70 captures,
# deep-validates, then runs the frozen candidate probe
# (HIST1001 + ps1001 + d3 B1/2/4/8) on the FRESH holdout.
# Primary verdict = B4 E[A] >= 4. FAIL-CLOSED.
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
OUT=artifacts/d3_capdev70
DATA=artifacts/d3_capdev70_data
MAN=$OUT/prompts_capdev70.json

echo "[cap-finish] waiting for 70/70 captures..."
while true; do
  n=$($PY -c "
import json
from pathlib import Path
spec=json.loads(Path('$MAN').read_text())
print(sum((Path('$DATA')/f'{c}.pt').exists() for c in spec['splits']['val']))
")
  [ "$n" -ge 70 ] && ! pgrep -f stage1_record.py >/dev/null && break
  sleep 60
done
sleep 5

echo "[cap-finish] deep validation..."
$PY -c "
import json, hashlib, torch
from pathlib import Path
spec=json.loads(Path('$MAN').read_text())
prompts={p['category']:p['prompt'] for p in spec['prompts']}
cats=spec['splits']['val']
assert len(cats)==70
bad=[]
for c in cats:
    x=torch.load(Path('$DATA')/f'{c}.pt', map_location='cpu', weights_only=True)
    try:
        assert x['category']==c and x['split']=='val'
        assert 1<x['S']<60
        assert x['prompt_sha256']==hashlib.sha256(prompts[c].encode()).hexdigest()
        assert torch.isfinite(x['h40'].float()).all()
        assert x['h40'].shape[0]>=x['S']+32
        assert x['teacher_class']=='SOURCE_CLASS_TORCH_SHIM'
    except Exception as e:
        bad.append((c,str(e)[:60]))
if bad:
    print('BAD:',bad); raise SystemExit(1)
print('all 70 captures validated')
"

echo "[cap-finish] frozen-candidate probe (HIST1001+ps1001+d3 B1/2/4/8)"
$PY mosaicvm/bench/d3_depth_branch_probe.py \
  --data "$DATA" --manifest "$MAN" --split val \
  --bridge artifacts/d3_scale1000_train/ps1001.pt \
  --output "$OUT/capdev70_candidate.json"

echo "[cap-finish] COMPLETE — verdict in $OUT/capdev70_candidate.json"
$PY -c "
import json
d=json.load(open('$OUT/capdev70_candidate.json'))
for B in ('1','2','4','8'):
    c=d['by_B'][B]['canonical']
    print(f\"B={B}: E[A]={c['mean']} trim20={c['trim20']} S1={c['S1']} S3={c['S3']} S4={c['S4']}\")
b4=d['by_B']['4']['canonical']['mean']
print('PRIMARY B4 E[A]=',b4,'=> ',('CAPABILITY PASS' if b4>=4 else 'below 4'))
"
