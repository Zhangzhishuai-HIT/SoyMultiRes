#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# BayesB (BGLR) for one fold directory per array task (line N of the list). Usually submitted through the night window
# (jobs/night/submit_night_window.sh), which overrides partition/memory/time and adds --begin/--deadline.
# usage: sbatch --array=1-N jobs/slurm_bglr_array.sh <list_file> [model] [nIter] [burnIn]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task $SLURM_ARRAY_TASK_ID: $DIR on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires_r/bin/Rscript scripts/02_baselines/bglr_fit.R "$DIR" "${2:-BayesB}" "${3:-12000}" "${4:-2000}"
