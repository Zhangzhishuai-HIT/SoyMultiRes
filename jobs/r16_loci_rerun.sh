#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-16 methods review: main locus map rerun (candidate genes also get the identifier of the best Arabidopsis hit,
# for Table S12) and the conditional locus map rerun after the conditional association analysis was rerun
# (jobs/r16_cond_rerun.sh). Every other output is expected to be unchanged (fixed random seeds).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
echo "start $(date) on $(hostname)"
$PY $S/locus_map.py --alpha 0.05 --min_mac 30 --perm 10000
$PY $S/locus_map.py --gwas _cond --tag _cond
echo "end $(date)"
