#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-19 referee: (1) the chromosome 10 lead SNP with the Peking (and PI 88788) SNP haplotype at Rhg1 as fixed effects;
# (2) mean differences over the sparse traits without SCN race 14 (same bootstrap resamples as the main comparisons).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
echo "start $(date) on $(hostname)"
$PY scripts/04_analysis/chr10_haplotype_conditional.py --threads 8
$PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models mtboost_v2_final best_single_avg4 ens_lightgbm_noshare lightgbm_avg4 --drop_from_sparse NEMATCYST_14 --out cv1_mean_diff_without_scn14
$PY scripts/04_analysis/bootstrap_mean_diff.py cv1 --models ens_lightgbm_noshare best_single_avg4 --drop_from_sparse NEMATCYST_14 --out cv1_mean_diff_ens_without_scn14
echo "end $(date)"
