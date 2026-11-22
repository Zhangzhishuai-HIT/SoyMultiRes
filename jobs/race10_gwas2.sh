#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Association analysis of P. sojae race 10, continued (the main analysis and the minor-homozygote counts are done):
# simulated threshold, then taxon+PC, G. max only (association, then its minor-homozygote counts) and conditional.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
echo "start $(date) on $(hostname)"
$PY $S/gwas_null_threshold.py --traitset stress --reps 1000 --threads 8 --add_traits PHYTOROT_10 --watch "" --watch_blocks ""
$PY $S/gwas_emmax.py --traitset stress --device cpu --covar taxon_pcs --tag _taxonpc --add_traits PHYTOROT_10
$PY $S/gwas_emmax.py --traitset stress --device cpu --taxon gmax --tag _gmax --add_traits PHYTOROT_10
$PY $S/gwas_mac.py --traitset stress --taxon gmax --tag _gmax
$PY $S/gwas_emmax.py --traitset stress --device cpu --condition major_genes --tag _cond --add_traits PHYTOROT_10
echo "end $(date)"
