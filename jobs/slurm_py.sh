#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Run one project Python script with the soymultires env on CPU; arguments passed through.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "start $(date) on $(hostname): $*"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python "$@"
echo "end $(date)"
