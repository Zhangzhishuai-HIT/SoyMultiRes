#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# GBLUP of P. sojae race 10 (added 2026-11-17): new accessions and the inner cross-validation.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python
echo "start $(date) on $(hostname)"
$PY -I scripts/02_baselines/run_gblup.py --scenario cv1 --traits PHYTOROT_10 --threads 4
for f in 0 1 2 3 4; do $PY -I scripts/02_baselines/run_gblup_inner.py --fold $f --traits PHYTOROT_10 --threads 4; done
echo "end $(date)"
