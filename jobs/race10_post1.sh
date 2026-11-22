#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Analyses that need the TraitBoost results of new accessions, the inner cross-validation and the model fitted to all
# accessions (33 traits): seed variation, the inner choices, the choice of the locus score, the multi-resistance index.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
echo "start $(date) on $(hostname)"
# seed_variation.py done in job 8974839
$PY $S/inner_choice_check.py
$PY $S/calibrate_readouts.py
$PY $S/germplasm.py
$PY $S/candidate_relatives.py
echo "end $(date)"
