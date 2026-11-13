#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7500M
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# Step 4: single-trait GBLUP, one trait per array task (line N of the trait list).
# usage: sbatch -J smr_gblup --array=1-32 jobs/slurm_gblup_array.sh <scenario> <trait_list>
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=${PY:-/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python}
TRAIT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$2")
echo "task $SLURM_ARRAY_TASK_ID: $TRAIT on $(hostname) with $PY"
$PY scripts/02_baselines/run_gblup.py --scenario "$1" --traits "$TRAIT" --threads 8
