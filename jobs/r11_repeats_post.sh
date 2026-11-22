#!/bin/bash
# After the two repeated tuning runs of TraitBoost for new accessions on AutoDL (tags v2r1 and v2r2, five folds each;
# tasks mtbrep_r<k>_f<f>): copy the fold predictions back, collect, summarize (tuning_repeats.py) and submit the joint
# bootstrap of the mean differences of every run.  usage: bash jobs/r11_repeats_post.sh [partitions, default q04,q05]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
PY="nice -n 10 /home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
for k in 1 2; do
  mkdir -p runs/cv1/mtboost_dev/v2r$k
  rsync -a autodl-vgpu48:/root/SoyMultiRes/runs/cv1/mtboost_dev/v2r$k/ runs/cv1/mtboost_dev/v2r$k/
  n=$(ls runs/cv1/mtboost_dev/v2r$k/fold*.tsv.gz | wc -l); [ "$n" -eq 5 ] || { echo "v2r$k: only $n folds"; exit 1; }
  $PY scripts/03_model/train_mtboost.py collect --scenario cv1 --tag v2r$k
done
$PY scripts/04_analysis/tuning_repeats.py
sbatch -J smr_mdrep -p "${1:-q04,q05}" jobs/r11_repeats_meandiff.sh
