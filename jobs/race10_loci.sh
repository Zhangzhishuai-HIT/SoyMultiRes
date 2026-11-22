#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Locus maps of the 33 traits (main analysis and the robustness analyses) and the checks that use only the association
# analysis; after jobs/race10_gwas2.sh. The choice of the locus score (calibrate_readouts.py) follows when the model
# fitted to all accessions is done.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
LOW=NEMATCYST_1,NEMATCYST_4,NEMATCYST_5,NEMATCYST_14,PHYTOROT_12,PHYTOROT_20,SDS,PYTHIUMROT,PMV,BPMV,APHID
echo "start $(date) on $(hostname)"
$PY $S/locus_map.py --alpha 0.05 --min_mac 30 --perm 10000
$PY $S/locus_map.py --gwas _taxonpc --tag _taxonpc
$PY $S/locus_map.py --gwas _gmax --mac _gmax --drop BPMV --tag _gmax
$PY $S/locus_map.py --block 8 --tag _block8
$PY $S/locus_map.py --block 32 --tag _block32
$PY $S/locus_map.py --zlim 2.5 --tag _z25
$PY $S/locus_map.py --zlim 3.5 --tag _z35
$PY $S/locus_map.py --drop NEMATCYST_14 --tag _noSCN14
$PY $S/locus_map.py --min_snps 2 --tag _min2snps
$PY $S/locus_map.py --drop $LOW --tag _res100
$PY $S/locus_map.py --gwas _cond --tag _cond
$PY $S/locus_map.py --null none --tag _bonf
$PY $S/locus_map.py --read_races 0 --tag _locusreading
$PY $S/trait_regions.py
$PY $S/nbarc_shift.py
$PY $S/control_loci.py
$PY $S/locus_robustness.py
$PY $S/race_differential_check.py --block 2059
$PY $S/race_differential_check.py --block 1173
$PY $S/race_differential_check.py --genomewide
$PY $S/race_differential_check.py --crossval
$PY $S/pubescence_check.py
echo "end $(date)"
