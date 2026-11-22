#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-16 referee (M3): copies of the sequence around the chromosome 10 lead SNP in the Rhg1 region of all Glycine max
# assemblies of the Legume Information System, against the SoySNP50K calls of the same cultivars.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export PYTHONUNBUFFERED=1
echo "start $(date) on $(hostname)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/probe_rhg1_scan.py --workers 2
echo "end $(date)"
