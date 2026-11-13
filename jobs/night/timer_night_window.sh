#!/bin/bash
# Open and close one night window for SoyMultiRes (pattern of the pan project's timer_06_night_window.sh).
# Waits until <night> 00:00, runs submit_night_window.sh, and at 06:57 cancels any smr_* job outside q04/q05 that has still
# not started (Slurm's --deadline normally drops it first). Running jobs end by their time limits before 06:55.
# Usage (login node): setsid nohup bash jobs/night/timer_night_window.sh YYYY-MM-DD >> logs/night/night_window_timer.log 2>&1 < /dev/null &
set -euo pipefail
NIGHT=${1:?night date YYYY-MM-DD}
ROOT=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
open=$(date -d "$NIGHT 00:00:05" +%s)
close=$(date -d "$NIGHT 06:57" +%s)
log() { echo "$(date '+%F %T') [$$] $*"; }
[ "$(date +%s)" -lt "$close" ] || { log "the $NIGHT window is over"; exit 2; }
log "waiting for the $NIGHT window"
while [ "$(date +%s)" -lt "$open" ]; do sleep 30; done
bash "$ROOT/jobs/night/submit_night_window.sh" "$NIGHT" || { rc=$?; log "submit_night_window.sh exit $rc"; exit "$rc"; }
while [ "$(date +%s)" -lt "$close" ]; do
  q=$(squeue -h -u "$USER" -o '%j') || { sleep 120; continue; }
  grep -q '^smr_' <<< "$q" || { log "no smr_ job left in the queue"; exit 0; }
  sleep 120
done
pending=$(squeue -h -u "$USER" -t PENDING -o '%i|%j|%P') || { log "squeue failed at 06:57: check the queue by hand"; exit 1; }
awk -F'|' '$2 ~ /^smr_/ && $3 !~ /^q0[45](,q0[45])?$/ {print $1}' <<< "$pending" | while read -r id; do
  log "cancel $id: not started by 06:57"
  scancel "$id" || log "scancel $id failed"
done
log "the $NIGHT window is closed"
