#!/bin/bash
# SCALE1000 phase-aware capture engine.
# Phase A: capture the 70 SCALE-DEV3 (val) prompts.
# Phase gate: wait for /tmp/scale1000_baseline_done.flag
#            (orchestrator runs frozen HIST700/WS700 baselines).
# Phase B: capture the 301 train prompts.
# Relaunches on recorder death or 45-min stall. Exits at 371.
set -u
cd /home/fazinahamed/Documents/MosaicVM

N_VAL=70
N_TRAIN=301
N_TOTAL=371
STALL_LIMIT=2700
FLAG=/tmp/scale1000_baseline_done.flag
LOG=/tmp/scale1000_engine.log
MANIFEST=artifacts/d3_scale1000/prompts_scale1000.json
DATA=artifacts/d3_scale1000_data

count() { # $1 = split
  python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
cats = spec['splits']['$1']
print(sum((Path('$DATA')/f'{c}.pt').exists() for c in cats))
"
}

newest_age() {
  local newest
  newest=$(ls -t $DATA/*.pt 2>/dev/null | head -1)
  if [ -z "$newest" ]; then echo 999999; return; fi
  echo $(( $(date +%s) - $(stat -c %Y "$newest") ))
}

launch_phase() { # $1 = split
  echo "[$(date '+%F %T')] launch phase=$1" >> "$LOG"
  pkill -9 -f stage1_record.py 2>/dev/null
  sleep 5
  for f in $DATA/*.pt; do
    [ -e "$f" ] || continue
    if ! .venv-cu/bin/python -c "
import torch
torch.load('$f', map_location='cpu', weights_only=True)
" 2>/dev/null; then
      echo "[$(date '+%F %T')] removing partial capture: $f" >> "$LOG"
      rm -f "$f"
    fi
  done
  local cats
  cats=$(python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
print(','.join(spec['splits']['$1']))
")
  mkdir -p "$DATA"
  S1_SOURCE=1 \
  S1_PROMPTS=$MANIFEST \
  S1_OUT=$DATA \
  S1_CATS="$cats" \
  SCG_CACHE_CAP=96 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 \
  setsid nohup .venv-cu/bin/python mosaicvm/bench/stage1_record.py \
    >> /tmp/scale1000_capture.log 2>&1 &
  echo "[$(date '+%F %T')] launched pid $! (phase $1)" >> "$LOG"
}

recorder_running() {
  pgrep -f stage1_record.py > /dev/null
}

recorder_age() {
  local pid
  pid=$(pgrep -f stage1_record.py | head -1)
  if [ -z "$pid" ]; then echo 0; return; fi
  ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' '
}

echo "[$(date '+%F %T')] engine started" >> "$LOG"
while true; do
  total=$(ls $DATA/*.pt 2>/dev/null | wc -l)
  if [ "$total" -ge "$N_TOTAL" ]; then
    echo "[$(date '+%F %T')] all $total captures present — engine exiting" >> "$LOG"
    exit 0
  fi
  vn=$(count val)
  if [ "$vn" -lt "$N_VAL" ]; then
    if recorder_running; then
      nf=$(ls $DATA/*.pt 2>/dev/null | wc -l)
      if [ "$nf" -gt 0 ]; then
        age=$(newest_age)
        if [ "$age" -gt "$STALL_LIMIT" ]; then
          echo "[$(date '+%F %T')] STALL in phase A (newest ${age}s) — relaunch" >> "$LOG"
          launch_phase val
        fi
      else
        ra=$(recorder_age)
        if [ "$ra" -gt 3600 ]; then
          echo "[$(date '+%F %T')] STALL in phase A startup (recorder ${ra}s, 0 files) — relaunch" >> "$LOG"
          launch_phase val
        fi
      fi
    else
      echo "[$(date '+%F %T')] phase A recorder dead at $vn/$N_VAL — relaunch" >> "$LOG"
      launch_phase val
    fi
  elif [ ! -f "$FLAG" ]; then
    if recorder_running; then
      sleep 120
    else
      echo "[$(date '+%F %T')] phase gate: val complete, waiting for baseline flag" >> "$LOG"
      sleep 120
    fi
  else
    if recorder_running; then
      nf=$(ls $DATA/*.pt 2>/dev/null | wc -l)
      if [ "$nf" -gt 0 ]; then
        age=$(newest_age)
        if [ "$age" -gt "$STALL_LIMIT" ]; then
          echo "[$(date '+%F %T')] STALL in phase B (newest ${age}s) — relaunch" >> "$LOG"
          launch_phase train
        fi
      else
        ra=$(recorder_age)
        if [ "$ra" -gt 3600 ]; then
          echo "[$(date '+%F %T')] STALL in phase B startup (recorder ${ra}s, 0 files) — relaunch" >> "$LOG"
          launch_phase train
        fi
      fi
    else
      echo "[$(date '+%F %T')] phase B recorder dead at $(count train)/$N_TRAIN — relaunch" >> "$LOG"
      launch_phase train
    fi
  fi
  sleep 300
done
