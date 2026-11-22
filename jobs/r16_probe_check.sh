#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-16 referee (M3): is the sequence around the chromosome 10 lead SNP (and the chromosome 18 lead SNP) unique in
# the Wm82.a2, Wm82.a4 and Peking assemblies, compared with 500 random SNPs of the array?
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export PYTHONUNBUFFERED=1
echo "start $(date) on $(hostname)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/probe_check.py --controls 500 --threads 4
echo "end $(date)"
