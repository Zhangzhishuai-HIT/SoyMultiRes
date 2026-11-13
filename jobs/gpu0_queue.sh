#!/bin/bash
# Run the tasks of a list on gpu10 GPU 0, at most P at a time. Started through jobs/launch_gpu0.sh, which selects the
# card by UUID (CUDA_VISIBLE_DEVICES) and sets nice; every task inherits that. Few CPU threads per task (the user's
# rule: no CPU compute on gpu10; the heavy work runs on the card).
# List lines: "<task_name> <command ...>"; a task whose name already has logs/gpu/<name>.done is skipped.
# usage (via launch_gpu0.sh): jobs/gpu0_queue.sh <list> <parallel>
set -uo pipefail
LIST=$1; P=${2:-4}
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
echo "queue $LIST: $(grep -c . "$LIST") tasks, $P in parallel, card $CUDA_VISIBLE_DEVICES, start $(date)"
# The done-check and an atomic lock (mkdir) come right before a task starts, after waiting for a free slot, so several
# runners can share task names without running a task twice (the first version checked when the previous task was
# launched and ran two cv1_inner splits twice on 2026-11-12).
while read -r name cmd; do
  [ -z "$name" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 3; done
  [ -f "logs/gpu/$name.done" ] && continue
  mkdir "logs/gpu/$name.lock" 2>/dev/null || { echo "$(date '+%T') $name taken by another runner"; continue; }
  ( bash -c "$cmd" > "logs/gpu/$name.log" 2>&1 && touch "logs/gpu/$name.done"; echo "$(date '+%T') $name exit $?" ) &
done < "$LIST"
wait
echo "queue done $(date)"
