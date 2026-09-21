#!/bin/bash
# SCALE210 autonomous orchestrator (roadmap S64 steps 5-13).
# Waits for ALL 196 captures (capture process holds ~9GB
# VRAM; sequential GPU use only), then: validate -> baselines
# on SCALE-DEV70 -> teacher -> HIST210 -> WS210 -> all evals
# -> paired ledger. Logs to /tmp/scale_orchestrator.log.
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
CAP=artifacts/d3_scale_data
OUT=artifacts/d3_scale210
mkdir -p "$OUT"

N_TOTAL=196

echo "[orch] waiting for all $N_TOTAL captures..."
while true; do
  done_all=$($PY -c "
import json
from pathlib import Path
spec = json.loads(Path('artifacts/d3_scale/prompts_scale.json').read_text())
cats = spec['splits']['train'] + spec['splits']['val']
print(sum((Path('artifacts/d3_scale_data')/f'{c}.pt').exists() for c in cats))
")
  if [ "$done_all" -ge "$N_TOTAL" ]; then break; fi
  sleep 300
done
echo "[orch] all captures complete: $done_all/$N_TOTAL"
# let the capture process release the GPU
sleep 120

echo "[orch] validating all captures load cleanly..."
$PY -c "
import json, hashlib, torch
from pathlib import Path
spec = json.loads(Path('artifacts/d3_scale/prompts_scale.json').read_text())
prompts = {p['category']: p['prompt'] for p in spec['prompts']}
splits = {}
for s, cats in spec['splits'].items():
    for c in cats:
        splits[c] = s
cats = spec['splits']['train'] + spec['splits']['val']
bad = []
for c in cats:
    p = Path('artifacts/d3_scale_data') / f'{c}.pt'
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
print('all 196 captures fully validated')
"
if [ $? -ne 0 ]; then
  echo "[orch] CAPTURE VALIDATION FAILED — aborting"
  exit 1
fi

echo "[orch] baselining WS1_84 + HIST1_84 on SCALE-DEV70..."
$PY mosaicvm/bench/d3_scale_eval.py --model ws84 \
  --data artifacts/d3_scale_data \
  --manifest artifacts/d3_scale/prompts_scale.json \
  --split val --tag SCALE-DEV70 \
  --output $OUT/baseline_ws84_scaledev70.json
$PY mosaicvm/bench/d3_scale_eval.py --model hist84 \
  --data artifacts/d3_scale_data \
  --manifest artifacts/d3_scale/prompts_scale.json \
  --split val --tag SCALE-DEV70 \
  --output $OUT/baseline_hist84_scaledev70.json
echo "[orch] baselines done"

echo "[orch] generating scale teacher records (gold replay)..."
$PY mosaicvm/bench/d3_scale_teacher.py

echo "[orch] training HIST1_210 (Stage A)..."
$PY mosaicvm/bench/d3_scale_train.py --stage hist

echo "[orch] training WS1_210 (Stage B)..."
$PY mosaicvm/bench/d3_scale_train.py --stage ws

echo "[orch] primary eval: SCALE-DEV70..."
$PY mosaicvm/bench/d3_scale_eval.py --model ws210 \
  --data artifacts/d3_scale_data \
  --manifest artifacts/d3_scale/prompts_scale.json \
  --split val --tag SCALE-DEV70 \
  --output $OUT/ws210_scaledev70.json
$PY mosaicvm/bench/d3_scale_eval.py --model hist210 \
  --data artifacts/d3_scale_data \
  --manifest artifacts/d3_scale/prompts_scale.json \
  --split val --tag SCALE-DEV70 \
  --output $OUT/hist210_scaledev70.json

echo "[orch] legacy eval: DEV21..."
$PY mosaicvm/bench/d3_scale_eval.py --model ws210 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/ws210_legacy_dev21.json
$PY mosaicvm/bench/d3_scale_eval.py --model hist210 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/hist210_legacy_dev21.json

echo "[orch] paired ledger + gates..."
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_ws84_scaledev70.json \
  $OUT/ws210_scaledev70.json \
  $OUT/ledger_ws84_vs_ws210.json

echo "[orch] COMPLETE — all outputs in $OUT"
