#!/bin/bash
# On AutoDL: GBLUP starts of TraitBoost (CPU, train_mtboost.py start) for the 32 splits of the race-10 rerun, four at a
# time with 8 threads each; then the GPU task queue (run_queue.py) is started.
set -uo pipefail
cd /root/SoyMultiRes
PY=/root/miniforge3/envs/soymultires/bin/python
mkdir -p logs/start
run_one() {
  name=$(echo "$*" | tr ' ' '_' | sed 's/--//g')
  [ -f "logs/start/$name.done" ] && return 0
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
    nice -n 10 $PY scripts/03_model/train_mtboost.py start $* --threads 8 > "logs/start/$name.log" 2>&1 && touch "logs/start/$name.done"
}
export -f run_one; export PY
echo "starts begin $(date)"
xargs -P 4 -I{} bash -c 'run_one {}' < jobs/autodl/start_list.txt
n_done=$(ls logs/start/*.done 2>/dev/null | wc -l); n_all=$(grep -c . jobs/autodl/start_list.txt)
echo "starts end $(date): $n_done done of $n_all"
if [ "$n_done" -eq "$n_all" ]; then
  mkdir -p logs/gpu
  nohup setsid $PY jobs/autodl/run_queue.py jobs/autodl/tasks_race10.txt --per_card 1 --headroom_gb 1.5 >> logs/gpu/runner.log 2>&1 < /dev/null &
  echo "GPU runner launched $(date)"
fi
