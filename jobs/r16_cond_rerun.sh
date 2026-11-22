#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-16 methods review: the two conditional association analyses rerun after gwas_emmax.py stopped testing the
# SNPs that are fitted as fixed effects (their statistics are now 0; all other statistics are unchanged).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
$PY scripts/04_analysis/gwas_emmax.py --traitset stress --device cpu --condition major_genes --tag _cond
$PY scripts/04_analysis/gwas_emmax.py --traitset stress --device cpu --tag _condscn \
    --condition_snps "NEMATCYST_1,NEMATCYST_3,NEMATCYST_5,NEMATCYST_14:ss715597431,ss715628625,ss715610396,ss715629273,ss715602753"
echo "end $(date)"
