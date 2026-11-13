#!/bin/bash
# Two largest SCN traits of the pilot on gpu10's CPU (8 threads, nice) after the GPU run hit its 5 GiB cap.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
echo "start $(date) on $(hostname)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 nice -n 10 \
  $PY scripts/02_baselines/pilot_gblup.py ../HapGAM/data/interim/soysnp50k/dosage.npy ../HapGAM/data/interim/soysnp50k/samples.tsv \
  results/pheno/scores_by_batch.tsv.gz results/pilot_gblup 8 cpu NEMATCYST_5,NEMATCYST_3
echo "end $(date)"
