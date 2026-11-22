#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-11 referee (M2c): TraitBoost against the mean of LightGBM and TraitBoost without shared trees (two single-trait
# tree models), and that ensemble against the best single-trait model and LightGBM; joint bootstrap of mean differences.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
$PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models mtboost_v2_final ens_lightgbm_noshare --out cv1_mean_diff_ens_lightgbm_noshare
$PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models ens_lightgbm_noshare best_single_avg4 lightgbm_avg4 --out cv1_mean_diff_ens_lightgbm_noshare_vs_single
echo "end $(date)"
