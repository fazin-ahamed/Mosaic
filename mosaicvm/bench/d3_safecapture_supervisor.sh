#!/bin/bash
# SafeCapture v2 — CPU supervisor, bounded GPU workers.
# Does NOT change SCALE1000 scientific recipe.
# Recycles the CUDA context every N prompts so host-RAM / allocator
# state cannot accumulate across a 20h marathon.
#
# Workers are launched via systemd-run --user --scope so they are
# NOT in the niri/omp cgroup that the 10:46 OOM killer targeted.
set -u
cd /home/fazinahamed/Documents/MosaicVM

N_VAL=70
N_TRAIN=301
N_TOTAL=371
EPOCH="${SAFECAP_EPOCH:-8}"
STALL_LIMIT=2700
FLAG=/tmp/scale1000_baseline_done.flag
LOG=/tmp/safecapture_supervisor.log
CAPLOG=/tmp/scale1000_capture.log
MANIFEST=artifacts/d3_scale1000/prompts_scale1000.json
DATA=artifacts/d3_scale1000_data
SNAPDIR=/tmp/safecapture_snaps
mkdir -p "$DATA" "$SNAPDIR"

count() {
  python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
cats = spec['splits']['$1']
print(sum((Path('$DATA')/f'{c}.pt').exists() for c in cats))
"
}

next_cats() { # $1=split $2=n
  python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
data = Path('$DATA')
need = [c for c in spec['splits']['$1'] if not (data/f'{c}.pt').exists()]
print(','.join(need[:int('$2')]))
"
}

scrub_tmp() {
  shopt -s nullglob
  for f in "$DATA"/*.pt.tmp "$DATA"/*.tmp; do
    echo "[$(date '+%F %T')] removing tmp $f" >> "$LOG"
    rm -f "$f"
  done
  # Tiny/truncated .pt only. Full schema audit is not per-loop.
  for f in "$DATA"/*.pt; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -lt 100000 ]; then
      echo "[$(date '+%F %T')] removing tiny $f ($sz)" >> "$LOG"
      rm -f "$f"
    fi
  done
}

dump_on_death() {
  local tag=$1
  local out="$SNAPDIR/${tag}_$(date +%s).txt"
  {
    echo "tag=$tag date=$(date -Is)"
    echo "--- nvidia-smi ---"
    nvidia-smi
    echo "--- free ---"
    free -h
    echo "--- last capture log ---"
    tail -30 "$CAPLOG"
  } > "$out" 2>&1
  echo "[$(date '+%F %T')] death dump $out" >> "$LOG"
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

newest_age() {
  local newest
  newest=$(ls -t "$DATA"/*.pt 2>/dev/null | head -1)
  if [ -z "$newest" ]; then echo 999999; return; fi
  echo $(( $(date +%s) - $(stat -c %Y "$newest") ))
}

launch_epoch() { # $1=split
  local cats
  cats=$(next_cats "$1" "$EPOCH")
  if [ -z "$cats" ]; then
    echo "[$(date '+%F %T')] no remaining $1 cats" >> "$LOG"
    return 0
  fi
  echo "[$(date '+%F %T')] epoch split=$1 n=$(echo "$cats" | tr ',' '\n' | grep -c .) cats=${cats:0:80}..." >> "$LOG"
  pkill -9 -f stage1_record.py 2>/dev/null || true
  sleep 3
  scrub_tmp
  # Isolate from niri/omp cgroup. Fallback: setsid if systemd-run missing.
  local cmd=(env
    S1_SOURCE=1
    S1_PROMPTS="$MANIFEST"
    S1_OUT="$DATA"
    S1_CATS="$cats"
    S1_MAX_NEW="$EPOCH"
    S1_TELEMETRY="$DATA/telemetry.jsonl"
    SCG_CACHE_CAP=96
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MALLOC_ARENA_MAX=2
    OMP_NUM_THREADS=8
    .venv-cu/bin/python mosaicvm/bench/stage1_record.py)
  if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope --collect \
      -- "${cmd[@]}" >> "$CAPLOG" 2>&1 &
  else
    setsid nohup "${cmd[@]}" >> "$CAPLOG" 2>&1 &
  fi
  echo "[$(date '+%F %T')] launched pid $! split=$1" >> "$LOG"
}

echo "[$(date '+%F %T')] SafeCapture supervisor started epoch=$EPOCH" >> "$LOG"
scrub_tmp

while true; do
  total=$(ls "$DATA"/*.pt 2>/dev/null | wc -l)
  if [ "$total" -ge "$N_TOTAL" ]; then
    echo "[$(date '+%F %T')] all $total captures present — exit" >> "$LOG"
    exit 0
  fi
  vn=$(count val)
  if [ "$vn" -lt "$N_VAL" ]; then
    split=val
  elif [ ! -f "$FLAG" ]; then
    if recorder_running; then
      sleep 60
    else
      echo "[$(date '+%F %T')] phase gate: val=$vn waiting for baseline flag" >> "$LOG"
      sleep 120
    fi
    continue
  else
    split=train
  fi

  if recorder_running; then
    ra=$(recorder_age)
    fa=$(newest_age)
    # Stall only if the process is old AND no new file is landing.
    # etime-only kills a healthy 8-prompt epoch (~45 min).
    # file-age-only kills a worker still loading against a 2h-old .pt.
    if [ "$ra" -gt "$STALL_LIMIT" ] && [ "$fa" -gt "$STALL_LIMIT" ]; then
      echo "[$(date '+%F %T')] STALL worker=${ra}s file=${fa}s — dump+relaunch" >> "$LOG"
      dump_on_death stall
      pkill -9 -f stage1_record.py 2>/dev/null || true
      sleep 3
      launch_epoch "$split"
    fi
  else
    echo "[$(date '+%F %T')] worker idle $split $(count "$split") remaining — launch" >> "$LOG"
    launch_epoch "$split"
  fi
  sleep 30
done
