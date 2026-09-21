#!/bin/bash
# SCALE1000 downstream chain. Waits for hist1001.pt, then:
#   eval hist1001 (SCALE-DEV3) -> census -> continuity
#   train WS1_1001 (memory-bounded) -> eval ws1001 -> census
#   paired ledgers hist700->hist1001 (primary), ws700->ws1001
# FAIL-CLOSED. Persistent log. No /tmp.
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
OUT=artifacts/d3_scale1000_train
SD=artifacts/d3_scale1000_data
SM=artifacts/d3_scale1000/prompts_scale1000.json

echo "[finish] waiting for hist1001.pt..."
while [ ! -f "$OUT/hist1001.pt" ]; do sleep 30; done
sleep 5
echo "[finish] hist1001.pt present; primary eval SCALE-DEV3"
$PY mosaicvm/bench/d3_scale_eval.py --model hist1001 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
  --output $OUT/hist1001_scaledev3.json
$PY mosaicvm/bench/d3_scale700_census.py --model hist1001 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_hist1001_scaledev3.json

echo "[finish] continuity eval hist1001 (DEV2 + DEV21)"
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

echo "[finish] primary ledger hist700 -> hist1001"
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_hist700_scaledev3.json \
  $OUT/hist1001_scaledev3.json \
  $OUT/ledger_hist700_vs_hist1001.json

echo "[finish] training WS1_1001 (Stage B)"
$PY mosaicvm/bench/d3_scale1000_train.py --stage ws

echo "[finish] eval ws1001 SCALE-DEV3"
$PY mosaicvm/bench/d3_scale_eval.py --model ws1001 \
  --data $SD --manifest $SM --split val --tag SCALE-DEV3 \
  --output $OUT/ws1001_scaledev3.json
$PY mosaicvm/bench/d3_scale700_census.py --model ws1001 \
  --data $SD --manifest $SM --split val \
  --output $OUT/census_ws1001_scaledev3.json
$PY mosaicvm/bench/d3_scale_eval.py --model ws1001 \
  --data artifacts/d3_data_105 \
  --manifest artifacts/stage1_prompts.json \
  --split val --tag LEGACY-DEV21 \
  --output $OUT/ws1001_legacy_dev21.json

echo "[finish] secondary ledger ws700 -> ws1001"
$PY mosaicvm/bench/d3_scale_ledger.py \
  $OUT/baseline_ws700_scaledev3.json \
  $OUT/ws1001_scaledev3.json \
  $OUT/ledger_ws700_vs_ws1001.json

echo "[finish] COMPLETE — all outputs in $OUT"
