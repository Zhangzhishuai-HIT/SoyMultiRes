#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# Paired bootstrap over groups of near-identical accessions (100,000 resamples) of all models for one trait per array task: line N of <list> = "<scenario> <trait>".
# usage: sbatch --array=1-N jobs/slurm_boot_array.sh <list> [--allow_missing]; then python -I scripts/04_analysis/bootstrap_compare.py collect <scenario>
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
read -r SC TR <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "bootstrap $SC $TR on $(hostname) $(date)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/bootstrap_compare.py run "$SC" --trait "$TR" --reps 100000 "${@:2}"
