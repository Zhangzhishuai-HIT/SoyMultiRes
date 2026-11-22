#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "start $(date) on $(hostname): bootstrap_mean_diff.py $*"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/bootstrap_mean_diff.py "$@"
echo "end $(date)"
