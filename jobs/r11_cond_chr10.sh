#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-11 referee (M4-iii): association of SCN races 1, 3, 5 and 14 with the SNPs most correlated with the chromosome
# 10 lead SNP (in NSF on chromosome 7, the chromosome 18 lead SNP, near GmSNAP11 on chromosome 11) and the lead SNPs
# near Rhg1 and Rhg4 fitted as fixed effects; the statistics at the chromosome 10 lead SNP are read from the output.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
$PY scripts/04_analysis/gwas_emmax.py --traitset stress --device cpu --tag _condscn \
    --condition_snps "NEMATCYST_1,NEMATCYST_3,NEMATCYST_5,NEMATCYST_14:ss715597431,ss715628625,ss715610396,ss715629273,ss715602753"
echo "end $(date)"
