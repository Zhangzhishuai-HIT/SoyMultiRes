#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Plan-stage pilot (2026-11-11): near-identical accession groups, then single-trait GBLUP h2 and 5-fold accuracy
# for the 32 traits. Reuses HapGAM's env and REML code read-only.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
TH=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
DOS=../HapGAM/data/interim/soysnp50k/dosage.npy
SAM=../HapGAM/data/interim/soysnp50k/samples.tsv
echo "start $(date) on $(hostname)"
$PY scripts/01_data/near_identical_groups.py $DOS $SAM results/inventory/accessions.tsv results/near_identical "$TH"
$PY scripts/02_baselines/pilot_gblup.py $DOS $SAM results/pheno/scores_by_batch.tsv.gz results/pilot_gblup "$TH"
echo "end $(date)"
