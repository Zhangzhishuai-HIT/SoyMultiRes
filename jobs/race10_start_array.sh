#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# GBLUP start of TraitBoost for one split per array task (line N of the list = arguments of train_mtboost.py start).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "start $(date) on $(hostname): start $ARGS"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/03_model/train_mtboost.py start $ARGS --threads 8
echo "end $(date)"
