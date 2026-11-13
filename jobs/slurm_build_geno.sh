#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7500M
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Step 2: genotype QC, fixed-size blocks, VanRaden kernel, PCs, known-gene blocks (phenotype-free).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=${PY:-/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python}
echo "start $(date) on $(hostname) with $PY"
$PY scripts/01_data/build_geno.py 8
echo "end $(date)"
