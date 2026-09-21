#!/bin/bash
# CAP-DEV70 capture supervisor (val-only, 70 prompts).
# SafeCapture v2 pattern: CPU supervisor + bounded GPU workers,
# CUDA context recycled every EPOCH prompts. systemd-run --scope
# isolates workers from the niri/omp OOM cgroup. Evaluation capture
# only — no training records, source recipe unchanged.
set -u
cd /home/fazinahamed/Documents/MosaicVM

N_TOTAL=70
EPOCH="${CAPDEV_EPOCH:-8}"
STALL_LIMIT=2700
LOG=/tmp/capdev70_supervisor.log
CAPLOG=/tmp/capdev70_capture.log
MANIFEST=artifacts/d3_capdev70/prompts_capdev70.json
DATA=artifacts/d3_capdev70_data
SNAPDIR=/tmp/capdev70_snaps
mkdir -p "$DATA" "$SNAPDIR"

count_val() {
  python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
print(sum((Path('$DATA')/f'{c}.pt').exists() for c in spec['splits']['val']))
"
}

next_cats() { # $1=n
  python3 -c "
import json
from pathlib import Path
spec = json.loads(Path('$MANIFEST').read_text())
data = Path('$DATA')
need = [c for c in spec['splits']['val'] if not (data/f'{c}.pt').exists()]
print(','.join(need[:int('$1')]))
"
}

scrub_tmp() {
  shopt -s nullglob
  for f in "$DATA"/*.pt.tmp "$DATA"/*.tmp; do rm -f "$f"; done
  for f in "$DATA"/*.pt; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -lt 100000 ]; then
      echo "[$(date '+%F %T')] removing tiny $f ($sz)" >> "$LOG"; rm -f "$f"
    fi
  done
}

dump_on_death() {
  local out="$SNAPDIR/$1_$(date +%s).txt"
  { echo "tag=$1 $(date -Is)"; nvidia-smi; free -h; tail -30 "$CAPLOG"; } \
    > "$out" 2>&1
  echo "[$(date '+%F %T')] death dump $out" >> "$LOG"
}

recorder_running() { pgrep -f stage1_record.py > /dev/null; }
recorder_age() {
  local pid; pid=$(pgrep -f stage1_record.py | head -1)
  [ -z "$pid" ] && { echo 0; return; }
  ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' '
}
newest_age() {
  local n; n=$(ls -t "$DATA"/*.pt 2>/dev/null | head -1)
  [ -z "$n" ] && { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$n") ))
}

launch_epoch() {
  local cats; cats=$(next_cats "$EPOCH")
  [ -z "$cats" ] && { echo "[$(date '+%F %T')] no remaining" >> "$LOG"; return 0; }
  echo "[$(date '+%F %T')] epoch n=$(echo "$cats" | tr ',' '\n' | grep -c .) cats=${cats:0:80}" >> "$LOG"
  pkill -9 -f stage1_record.py 2>/dev/null || true
  sleep 3; scrub_tmp
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
    systemd-run --user --scope --collect -- "${cmd[@]}" >> "$CAPLOG" 2>&1 &
  else
    setsid nohup "${cmd[@]}" >> "$CAPLOG" 2>&1 &
  fi
  echo "[$(date '+%F %T')] launched pid $!" >> "$LOG"
}

echo "[$(date '+%F %T')] CAP-DEV70 supervisor started epoch=$EPOCH" >> "$LOG"
scrub_tmp
while true; do
  total=$(count_val)
  if [ "$total" -ge "$N_TOTAL" ]; then
    echo "[$(date '+%F %T')] all $total captures present — exit" >> "$LOG"
    exit 0
  fi
  if recorder_running; then
    ra=$(recorder_age); fa=$(newest_age)
    if [ "$ra" -gt "$STALL_LIMIT" ] && [ "$fa" -gt "$STALL_LIMIT" ]; then
      echo "[$(date '+%F %T')] STALL worker=${ra}s file=${fa}s — relaunch" >> "$LOG"
      dump_on_death stall
      pkill -9 -f stage1_record.py 2>/dev/null || true
      sleep 3; launch_epoch
    fi
  else
    echo "[$(date '+%F %T')] idle $total/$N_TOTAL — launch" >> "$LOG"
    launch_epoch
  fi
  sleep 30
done
