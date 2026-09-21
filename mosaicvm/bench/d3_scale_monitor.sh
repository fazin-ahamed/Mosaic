#!/bin/bash
# Persistent SCALE210 pipeline monitor (detached, no timeout).
# Every 30 min appends a status line to /tmp/scale_monitor.log.
# Flags failures loudly. Exits when the ledger appears.
cd /home/fazinahamed/Documents/MosaicVM
LOG=/tmp/scale_monitor.log
LEDGER=artifacts/d3_scale210/ledger_ws84_vs_ws210.json

echo "[$(date '+%F %T')] monitor started" >> "$LOG"
while true; do
  n=$(ls artifacts/d3_scale_data/*.pt 2>/dev/null | wc -l)
  m=$(pgrep -f stage1_record.py > /dev/null && echo RUN || echo DEAD)
  w=$(pgrep -f d3_scale_watchdog.sh > /dev/null && echo RUN || echo DEAD)
  o=$(pgrep -f d3_scale_orchestrate.sh > /dev/null && echo RUN || echo DEAD)
  orch_stage=$(tail -1 /tmp/scale_orchestrator.log 2>/dev/null | cut -c1-60)
  echo "[$(date '+%F %T')] caps=$n/196 marathon=$m watchdog=$w orch=$o | $orch_stage" >> "$LOG"
  # failure escalation: pipeline dead before completion
  if [ "$w" = "DEAD" ] && [ "$n" -lt 196 ]; then
    echo "[$(date '+%F %T')] *** ALERT: watchdog DEAD with $n/196 — manual intervention needed ***" >> "$LOG"
  fi
  if [ "$o" = "DEAD" ] && [ ! -f "$LEDGER" ] && [ "$n" -ge 196 ]; then
    echo "[$(date '+%F %T')] *** ALERT: orchestrator DEAD after captures complete, no ledger ***" >> "$LOG"
  fi
  if [ -f "$LEDGER" ]; then
    echo "[$(date '+%F %T')] LEDGER PRESENT — pipeline complete. Monitor exiting." >> "$LOG"
    exit 0
  fi
  sleep 1800
done
