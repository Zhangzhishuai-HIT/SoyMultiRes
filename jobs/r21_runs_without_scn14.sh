#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-21 check (W1): the two further tuning runs against the best single-trait model over the sparse traits without
# SCN race 14 (same resamples as the main comparisons).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
for r in v2r1 v2r2; do
  $PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models mtboost_${r}_final best_single_avg4 --drop_from_sparse NEMATCYST_14 --out cv1_mean_diff_${r}_without_scn14
done
echo "end $(date)"
