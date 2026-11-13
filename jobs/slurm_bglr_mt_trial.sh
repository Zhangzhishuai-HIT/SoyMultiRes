#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7500M
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Week 1: BGLR Multitrait timing trial on the five SCN races (N = 1000, 2000, 4000).
set -uo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires_r/bin/Rscript
for N in 1000 2000 4000; do
  echo "== N=$N $(date)"
  /usr/bin/time -v $R scripts/02_baselines/bglr_mt_trial.R work/megalmm/trial_$N 200 2>&1 | grep -E '^\{|^ *NEMAT|correlations|Maximum resident|Elapsed|Error|error'
done
