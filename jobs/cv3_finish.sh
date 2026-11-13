#!/bin/bash
# Third scenario (held-out experiments): collect the predictions of all models and submit the paired comparisons.
# Run on the login node once the 16 MT-Boost CV3 units, the LightGBM CV3 parts (+ 3 extra seeds) and BayesB CV3 are done.
# usage: bash jobs/cv3_finish.sh
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python
$PY scripts/03_model/train_mtboost.py collect --scenario cv3 --tag v2
$PY scripts/02_baselines/run_lgbm.py --scenario cv3 --collect
$PY scripts/02_baselines/bglr_prepare.py collect --scenario cv3
ls runs/cv3/
# NSCANKER: its only held-out experiment leaves one class in the training set (not evaluable)
L=jobs/lists/compare_cv3.txt; : > $L
for B in mtboost_v2_gblup bayesb lightgbm_avg4 mtboost_v2_noshare2; do echo "cv3 mtboost_v2_final $B all" >> $L; done
sbatch --job-name=smr_cmp_cv3 --array=1-4 jobs/slurm_compare_cv3_array.sh $L
