#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# Step 5: BGLR Multitrait, one stress category x fold per array task (line N of <list>). Usually through the night
# window (jobs/night/submit_night_window.sh adds -p cu,fat --begin/--deadline).
# usage: sbatch --array=1-N jobs/slurm_bglrmt_array.sh <list> [nIter] [burnIn]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task $SLURM_ARRAY_TASK_ID: $DIR on $(hostname) partition ${SLURM_JOB_PARTITION:-?} $(date)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires_r/bin/Rscript scripts/02_baselines/bglr_mt_fit.R "$DIR" "${2:-12000}" "${3:-2000}"
echo "end $(date)"
