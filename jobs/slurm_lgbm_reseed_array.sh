#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7500M
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# Step 4: LightGBM refits with 3 extra seeds (stored tuned settings, no tuning), one trait x fold per array task. Line N of <list> = "<trait> <fold> <scenario>".
# usage: sbatch -J smr_lgbm --array=1-N%K jobs/slurm_lgbm_array.sh <list> [trials] [timeout_s]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
read -r TRAIT FOLD SCEN <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
SCEN=${SCEN:-cv1}
echo "task $SLURM_ARRAY_TASK_ID: $TRAIT fold $FOLD ($SCEN) on $(hostname)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python scripts/02_baselines/run_lgbm.py \
  --scenario "$SCEN" --trait "$TRAIT" --fold "$FOLD" --threads 8 --reseed "${2:-3}"
