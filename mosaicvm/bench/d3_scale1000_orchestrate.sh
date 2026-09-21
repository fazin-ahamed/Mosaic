#!/bin/bash
# SCALE1000 orchestrator. FAIL-CLOSED.
# 1. wait for 70 SCALE-DEV3 captures + recorder idle
# 2. baseline HIST1_700 + WS1_700 on SCALE-DEV3 (+ census)
# 3. set baseline flag (engine starts phase B)
# 4. wait for all 371 captures; validate deeply
# 5. teacher (301 gold replays) -> HIST1_1001 -> WS1_1001
# 6. evals on SCALE-DEV3 (primary); continuity SCALE-DEV2 + DEV21
# 7. paired ledgers hist700->hist1001 (primary), ws700->ws1001
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
OUT=artifacts/d3_scale1000_train
FLAG=/tmp/scale1000_baseline_done.flag
mkdir -p "$OUT"

SM=artifacts/d3_scale1000/prompts_scale1000.json
SD=artifacts/d3_scale1000_data
DEV3_SHA=41c5b929560e4605cb66ce139fef761b136d4dad7be3c0fbddac4ac18718eacf

echo "[orch] verifying frozen SCALE-DEV3..."
got=$($PY -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('artifacts/d3_scale1000/prompts_scale_dev3.json').read_bytes()).hexdigest())")
if [ "$got" != "$DEV3_SHA" ]; then
  echo "[orch] SCALE-DEV3 mutated: $got"
  exit 1
fi

echo "[orch] waiting for 70 SCALE-DEV3 captures..."
while true; do
  vn=$($PY -c "
import json
from pathlib import Path
spec = json.loads(Path('$SM').read_text())
print(sum((Path('$SD')/f'{c}.pt').exists() for c in spec['splits']['val']))
")
  if [ "$vn" -ge 70 ]; then
    if pgrep -f stage1_record.py > /dev/null; then
      sleep 120
    else
      break
    fi
  else
    sleep 120
  fi
done
sleep 60
if [ -f "$OUT/baseline_hist700_scaledev3.json" ] \
   && [ -f "$OUT/baseline_ws700_scaledev3.json" ]; then
  echo "[orch] SCALE-DEV3 baselines already frozen — skip"
  touch "$FLAG"
else
  echo "[orch] SCALE-DEV3 captures complete — baselining"
  $PY mosaicvm/bench/d3_scale_eval.py --model hist700 \
    --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
    --output $OUT/baseline_hist700_scaledev3.json
  $PY mosaicvm/bench/d3_scale_eval.py --model ws700 \
    --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
    --output $OUT/baseline_ws700_scaledev3.json
  $PY mosaicvm/bench/d3_scale700_census.py --model hist700 \
    --data $SD --manifest $SM --split val \
    --output $OUT/baseline_census_hist700_scaledev3.json
  $PY mosaicvm/bench/d3_scale700_census.py --model ws700 \
    --data $SD --manifest $SM --split val \
    --output $OUT/baseline_census_ws700_scaledev3.json
  echo "[orch] baselines frozen — releasing phase B captures"
  touch "$FLAG"
fi

echo "[orch] 371 captures check..."
n=$(ls $SD/*.pt 2>/dev/null | wc -l)
echo "[orch] found $n capture files (target 371)"
if [ "$n" -lt 371 ]; then
  echo "[orch] FATAL: captures incomplete ($n < 371)"
  exit 1
fi
echo "[orch] all captures complete: validating..."
$PY -c "
import json, hashlib, torch
from pathlib import Path
spec = json.loads(Path('artifacts/d3_scale1000/prompts_scale1000.json').read_text())
prompts = {p['category']: p['prompt'] for p in spec['prompts']}
splits = {}
for s, cats in spec['splits'].items():
    for c in cats:
        splits[c] = s
cats = spec['splits']['train'] + spec['splits']['val']
assert len(cats) == 371
bad = []
for c in cats:
    p = Path('artifacts/d3_scale1000_data') / f'{c}.pt'
    try:
        x = torch.load(p, map_location='cpu', weights_only=True)
        assert x['category'] == c, 'category'
        assert x['split'] == splits[c], 'split'
        assert 1 < x['S'] < 60, 'S bounds'
        assert x['prompt_sha256'] == hashlib.sha256(
            prompts[c].encode()).hexdigest(), 'prompt hash'
        assert torch.isfinite(x['h40'].float()).all(), 'h40 finite'
        assert torch.isfinite(x['h42'].float()).all(), 'h42 finite'
        assert x['h40'].shape == x['h42'].shape, 'shape'
        assert x['h40'].shape[0] >= x['S'] + 32, 'horizon'
        assert x['teacher_class'] == 'SOURCE_CLASS_TORCH_SHIM', 'class'
        assert x['snapshot_revision'] == '7872f01b1d1fe23eabc4c98b48bffcef5a386062', 'snapshot'
    except Exception as e:
        bad.append((c, str(e)[:80]))
if bad:
    print('BAD CAPTURES:', bad)
    raise SystemExit(1)
print('all 371 captures fully validated')
"

echo "[orch] teacher records (301 gold replays)..."
$PY mosaicvm/bench/d3_scale1000_teacher.py

echo "[orch] training HIST1_1001 (Stage A)..."
$PY mosaicvm/bench/d3_scale1000_train.py --stage hist

echo "[orch] training WS1_1001 (Stage B)..."
$PY mosaicvm/bench/d3_scale1000_train.py --stage ws

echo "[orch] primary eval: SCALE-DEV3..."
$PY mosaicvm/bench/d3_scale_eval.py --model hist1001 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
  --output $OUT/hist1001_scaledev3.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws1001 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
  --output $OUT/ws1001_scaledev3.json
$PY mosaicvm/bench/d3_scale700_census.py --model hist1001 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_hist1001_scaledev3.json
$PY mosaicvm/bench/d3_scale700_census.py --model ws1001 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_ws1001_scaledev3.json

echo "[orch] continuity eval: SCALE-DEV2 + legacy DEV21..."
$PY mosaicvm/bench/d3_scale_eval.py --model hist1001 \
  --data artifacts/d3_scale700_data \
  --manifest artifacts/d3_scale700/prompts_scale700.json \
  --split val --tag SCALE-DEV2 \
  --output $OUT/hist1001_scaledev2.json
$PY mosaicvm/bench/d3_scale_eval.py --model hist1001 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/hist1001_legacy_dev21.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws1001 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/ws1001_legacy_dev21.json

echo "[orch] paired ledgers..."
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_hist700_scaledev3.json \
  $OUT/hist1001_scaledev3.json \
  $OUT/ledger_hist700_vs_hist1001.json
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_ws700_scaledev3.json \
  $OUT/ws1001_scaledev3.json \
  $OUT/ledger_ws700_vs_ws1001.json

echo "[orch] COMPLETE — all outputs in $OUT"
