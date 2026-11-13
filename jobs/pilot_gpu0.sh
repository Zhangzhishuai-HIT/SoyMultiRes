#!/bin/bash
# Plan-stage pilot (2026-11-11) on gpu10: near-identical groups (CPU, 8 threads) then GBLUP h2 + 5-fold (GPU 0).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
DOS=../HapGAM/data/interim/soysnp50k/dosage.npy
SAM=../HapGAM/data/interim/soysnp50k/samples.tsv
echo "start $(date) on $(hostname)"
CUDA_VISIBLE_DEVICES="" $PY scripts/01_data/near_identical_groups.py $DOS $SAM results/inventory/accessions.tsv results/near_identical 8
$PY scripts/02_baselines/pilot_gblup.py $DOS $SAM results/pheno/scores_by_batch.tsv.gz results/pilot_gblup 8 cuda
echo "end $(date)"
