#!/bin/bash
# Held-out experiments after the 33-trait TraitBoost rerun on AutoDL (16 units, fit + reseed = four refits per model).
# Run on the login node once all units are done: copies the unit predictions back, collects TraitBoost, rebuilds the
# best single-trait model and the GBLUP-LightGBM ensemble, writes the summary and submits the paired bootstrap of the
# 21 traits; the dependent job jobs/race10_cv3_post2.sh collects it and runs the mean differences and the inner checks.
# usage: bash jobs/race10_cv3_post.sh [partitions, default q04,q05]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
PART=${1:-q04,q05}
PY="nice -n 10 /home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
rsync -a autodl-vgpu48:/root/SoyMultiRes/runs/cv3/mtboost_dev/v2/ runs/cv3/mtboost_dev/v2/
n=$(ls runs/cv3/mtboost_dev/v2/unit*.tsv.gz | wc -l); [ "$n" -eq 16 ] || { echo "only $n units"; exit 1; }
$PY scripts/03_model/train_mtboost.py collect --scenario cv3 --tag v2
$PY scripts/04_analysis/make_best_single.py results/gate/cv1_inner_best_single_avg4.tsv best_single_avg4 cv3
$PY scripts/04_analysis/make_ensemble.py
$PY scripts/04_analysis/cv3_summary.py
A=$(sbatch --parsable -J smr_bootcv3 -p "$PART" --array=1-21 jobs/slurm_boot_array.sh jobs/lists/boot_cv3.txt)
B=$(sbatch --parsable -J smr_cv3post -p "$PART" --dependency=afterok:$A jobs/race10_cv3_post2.sh)
echo "bootstrap array $A, post job $B"
