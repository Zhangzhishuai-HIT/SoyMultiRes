#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7500M
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Step 6 on CPU (gpu10 GPU 0 is down since 2026-11-11 14:48): one train_mt.py call; arguments passed through.
# usage: sbatch -J smr_mt jobs/slurm_mt.sh inner --fold 0 --inner 0 --config '{...}' --tag dev
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "start $(date) on $(hostname): $*"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python scripts/03_model/train_mt.py "$@" --threads 8 --device cpu
echo "end $(date)"
