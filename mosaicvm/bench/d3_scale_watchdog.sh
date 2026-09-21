#!/bin/bash
# Capture-marathon watchdog: keeps the 196-prompt capture
# job alive unattended.
#  - every 600s: count captures, check the recorder process
#  - recorder dead + <196 captures  -> relaunch (resumable:
#    the recorder skips already-captured prompts)
#  - stalled (no new capture file in 45 min) -> kill + relaunch
#  - 196 captures -> exit 0 (orchestrator takes over)
# Logs to /tmp/scale_watchdog.log
set -uo pipefail
cd /home/fazinahamed/Documents/MosaicVM

N_TOTAL=196
STALL_LIMIT=2700   # seconds without a new capture file
LOG=/tmp/scale_watchdog.log

count_captures() {
  ls artifacts/d3_scale_data/*.pt 2>/dev/null | wc -l
}

newest_age() {
  local f newest
  newest=$(ls -t artifacts/d3_scale_data/*.pt 2>/dev/null | head -1)
  if [ -z "$newest" ]; then echo 999999; return; fi
  echo $(( $(date +%s) - $(stat -c %Y "$newest") ))
}

recorder_running() {
  pgrep -f "stage1_record.py" > /dev/null
}

relaunch() {
  echo "[$(date '+%H:%M:%S')] RELAUNCH: $(count_captures)/196 captured" >> "$LOG"
  # kill any stragglers (process group via pattern)
  pkill -9 -f stage1_record.py 2>/dev/null
  sleep 5
  export S1_SOURCE=1 \
         S1_PROMPTS=artifacts/d3_scale/prompts_scale.json \
         S1_OUT=artifacts/d3_scale_data \
         SCG_CACHE_CAP=96 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8
  # no S1_CATS: capture all remaining (recorder skips done)
  setsid nohup .venv-cu/bin/python mosaicvm/bench/stage1_record.py \
    >> /tmp/scale_capture.log 2>&1 &
  echo "[$(date '+%H:%M:%S')] relaunched: pid $!" >> "$LOG"
}

echo "[$(date '+%H:%M:%S')] watchdog started" >> "$LOG"
while true; do
  n=$(count_captures)
  if [ "$n" -ge "$N_TOTAL" ]; then
    echo "[$(date '+%H:%M:%S')] all $n captures present — watchdog exiting" >> "$LOG"
    exit 0
  fi
  if recorder_running; then
    age=$(newest_age)
    if [ "$age" -gt "$STALL_LIMIT" ]; then
      echo "[$(date '+%H:%M:%S')] STALL: newest capture ${age}s old ($n/196)" >> "$LOG"
      relaunch
    fi
  else
    echo "[$(date '+%H:%M:%S')] recorder DEAD at $n/196" >> "$LOG"
    relaunch
  fi
  sleep 600
done
