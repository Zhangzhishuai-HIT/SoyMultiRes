#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# After the paired bootstrap of the held-out experiments: collect it, the joint bootstrap of the mean differences and
# the checks of the choices made within training sets (33-trait rerun).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
echo "start $(date) on $(hostname)"
$PY $S/bootstrap_compare.py collect cv3
$PY $S/bootstrap_mean_diff.py cv3
$PY $S/inner_choice_check.py
echo "end $(date)"
