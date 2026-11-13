#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# Paired bootstrap comparisons in the third scenario, one per array task: line N of <list> = "<scenario> <a> <b> <set>".
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
read -r SC A B SET <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "compare $SC $A vs $B ($SET) on $(hostname) $(date)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/compare_models.py "$SC" "$A" "$B" --set "$SET" --reps 1000 --exclude NSCANKER
