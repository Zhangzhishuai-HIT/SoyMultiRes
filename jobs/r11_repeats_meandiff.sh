#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Joint bootstrap of the mean differences of TraitBoost (main run and two repeated tuning runs) against the best
# single-trait model, LightGBM, the mean of LightGBM and TraitBoost without shared trees, and TraitBoost without shared
# trees of the same run (two refits).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
for tag in v2 v2r1 v2r2; do
  $PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models mtboost_${tag}_final best_single_avg4 lightgbm_avg4 ens_lightgbm_noshare mtboost_${tag}_noshare2 --out cv1_mean_diff_$tag
done
echo "end $(date)"
