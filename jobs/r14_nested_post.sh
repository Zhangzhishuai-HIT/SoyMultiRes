#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%j.log
# Round-14 referee (M1): best single-trait model chosen within each training set (new accessions) and without the
# records of the held-out experiment (held-out experiments), make_best_single.py --nested. After the paired bootstraps
# of cv1, cv2 and cv3: collect them, the joint bootstraps of the mean differences (main run, repeated tuning runs,
# ensemble of LightGBM and TraitBoost without shared trees) and the summary of the tuning runs.
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I"
S=scripts/04_analysis
echo "start $(date) on $(hostname)"
for sc in cv1 cv2 cv3; do $PY $S/bootstrap_compare.py collect $sc; done
$PY $S/bootstrap_mean_diff.py cv1
$PY $S/bootstrap_mean_diff.py cv3
for tag in v2 v2r1 v2r2; do
  $PY $S/bootstrap_mean_diff.py cv1 --models mtboost_${tag}_final best_single_avg4 lightgbm_avg4 ens_lightgbm_noshare mtboost_${tag}_noshare2 --out cv1_mean_diff_$tag
done
$PY $S/bootstrap_mean_diff.py cv1 --models mtboost_v2_final ens_lightgbm_noshare --out cv1_mean_diff_ens_lightgbm_noshare
$PY $S/bootstrap_mean_diff.py cv1 --models ens_lightgbm_noshare best_single_avg4 lightgbm_avg4 --out cv1_mean_diff_ens_lightgbm_noshare_vs_single
$PY $S/tuning_repeats.py
echo "end $(date)"
