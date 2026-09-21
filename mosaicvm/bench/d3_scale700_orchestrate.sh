#!/bin/bash
# SCALE700 orchestrator (roadmap S48 steps 6-14). FAIL-CLOSED.
# 1. wait for 70 SCALE-DEV2 captures + recorder idle
# 2. baseline HIST1_210 + WS1_210 on SCALE-DEV2 (+ census)
# 3. set baseline flag (engine then starts phase B captures)
# 4. wait for all 560 captures; validate deeply
# 5. teacher (490 gold replays) -> HIST1_700 -> WS1_700
# 6. evals on SCALE-DEV2 (primary), SCALE-DEV70 + DEV21
#    (continuity); censuses
# 7. paired ledgers: hist210->hist700 (primary),
#    ws210->ws700 (secondary)
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
OUT=artifacts/d3_scale700_train
FLAG=/tmp/scale700_baseline_done.flag
mkdir -p "$OUT"

SM=artifacts/d3_scale700/prompts_scale700.json
SD=artifacts/d3_scale700_data

echo "[orch] waiting for 70 SCALE-DEV2 captures..."
while true; do
  vn=$($PY -c "
import json
from pathlib import Path
spec = json.loads(Path('$SM').read_text())
cats = spec['splits']['val']
print(sum((Path('$SD')/f'{c}.pt').exists() for c in cats))
")
  if [ "$vn" -ge 70 ]; then
    if pgrep -f stage1_record.py > /dev/null; then
      sleep 120   # let the val batch exit cleanly
    else
      break
    fi
  else
    sleep 300
  fi
done
sleep 60   # GPU release
echo "[orch] SCALE-DEV2 captures complete — baselining"

$PY mosaicvm/bench/d3_scale_eval.py --model hist210 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV2 \
  --output $OUT/baseline_hist210_scaledev2.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws210 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV2 \
  --output $OUT/baseline_ws210_scaledev2.json
$PY mosaicvm/bench/d3_scale700_census.py --model hist210 \
  --data $SD --manifest $SM --split val \
  --output $OUT/baseline_census_hist210_scaledev2.json
$PY mosaicvm/bench/d3_scale700_census.py --model ws210 \
  --data $SD --manifest $SM --split val \
  --output $OUT/baseline_census_ws210_scaledev2.json
echo "[orch] baselines frozen — releasing phase B captures"
touch "$FLAG"

echo "[orch] waiting for all 560 captures..."
while true; do
  tn=$($PY -c "
import json
from pathlib import Path
spec = json.loads(Path('$SM').read_text())
cats = spec['splits']['train'] + spec['splits']['val']
print(sum((Path('$SD')/f'{c}.pt').exists() for c in cats))
")
  if [ "$tn" -ge 560 ]; then break; fi
  sleep 300
done
sleep 120   # recorder exit + GPU release
echo "[orch] all captures complete: validating..."
$PY -c "
import json, hashlib, torch
from pathlib import Path
spec = json.loads(Path('artifacts/d3_scale700/prompts_scale700.json').read_text())
prompts = {p['category']: p['prompt'] for p in spec['prompts']}
splits = {}
for s, cats in spec['splits'].items():
    for c in cats:
        splits[c] = s
cats = spec['splits']['train'] + spec['splits']['val']
assert len(cats) == 560
bad = []
for c in cats:
    p = Path('artifacts/d3_scale700_data') / f'{c}.pt'
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
print('all 560 captures fully validated')
"

echo "[orch] teacher records (490 gold replays)..."
$PY mosaicvm/bench/d3_scale700_teacher.py

echo "[orch] training HIST1_700 (Stage A)..."
$PY mosaicvm/bench/d3_scale700_train.py --stage hist

echo "[orch] training WS1_700 (Stage B)..."
$PY mosaicvm/bench/d3_scale700_train.py --stage ws

echo "[orch] primary eval: SCALE-DEV2..."
$PY mosaicvm/bench/d3_scale_eval.py --model hist700 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV2 \
  --output $OUT/hist700_scaledev2.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws700 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV2 \
  --output $OUT/ws700_scaledev2.json
$PY mosaicvm/bench/d3_scale700_census.py --model hist700 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_hist700_scaledev2.json
$PY mosaicvm/bench/d3_scale700_census.py --model ws700 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_ws700_scaledev2.json

echo "[orch] continuity eval: SCALE-DEV70 + legacy DEV21..."
$PY mosaicvm/bench/d3_scale_eval.py --model hist700 \
  --data artifacts/d3_scale_data \
  --manifest artifacts/d3_scale/prompts_scale.json \
  --split val --tag SCALE-DEV70 \
  --output $OUT/hist700_scaledev70.json
$PY mosaicvm/bench/d3_scale_eval.py --model hist700 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/hist700_legacy_dev21.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws700 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/ws700_legacy_dev21.json

echo "[orch] paired ledgers..."
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_hist210_scaledev2.json \
  $OUT/hist700_scaledev2.json \
  $OUT/ledger_hist210_vs_hist700.json
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_ws210_scaledev2.json \
  $OUT/ws700_scaledev2.json \
  $OUT/ledger_ws210_vs_ws700.json

echo "[orch] COMPLETE — all outputs in $OUT"
