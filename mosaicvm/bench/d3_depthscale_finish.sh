#!/bin/bash
# DepthScale-1001 downstream. Waits for ps1001.pt, then measures
# HIST1001 + ps1001 (greedy + d3 branching) vs the frozen ps_old
# baseline (depth_branch_probe.json already on disk). FAIL-CLOSED.
set -euo pipefail
cd /home/fazinahamed/Documents/MosaicVM
export SCG_CACHE_CAP=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
PY=.venv-cu/bin/python
OUT=artifacts/d3_scale1000_train

echo "[ds-finish] waiting for ps1001.pt..."
while [ ! -f "$OUT/ps1001.pt" ]; do sleep 60; done
sleep 5
echo "[ds-finish] ps1001.pt present; probe with ps1001"
$PY mosaicvm/bench/d3_depth_branch_probe.py \
  --bridge "$OUT/ps1001.pt" \
  --output "$OUT/depth_branch_probe_ps1001.json"

echo "[ds-finish] COMPLETE — compare:"
echo "  ps_old : $OUT/depth_branch_probe.json"
echo "  ps1001 : $OUT/depth_branch_probe_ps1001.json"
