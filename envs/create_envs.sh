#!/bin/bash
# Build the two project environments (2026-11-11). Direct connection (no proxy); conda-forge/bioconda via the TUNA mirror (~/.condarc).
# Versions pinned to HapGAM's so the shared package cache is reused.
set -euo pipefail
MAMBA=/home/user/zhangzhishuai/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/user/zhangzhishuai/.local/share/mamba
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
which=${1:-both}
if [[ $which == py || $which == both ]]; then
  CONDA_OVERRIDE_CUDA=12.4 $MAMBA create -y -n soymultires -c conda-forge \
    python=3.11 pytorch=2.5.1 cuda-version=12.4 numpy=2.4.6 pandas=3.0.6 scipy=1.17.1 scikit-learn=1.9.1 \
    lightgbm=4.7.0 statsmodels=0.15.0 matplotlib=3.10.9 pyarrow threadpoolctl optuna=5.0.0
fi
if [[ $which == r || $which == both ]]; then
  $MAMBA create -y -n soymultires_r -c conda-forge -c bioconda \
    r-base=4.3.3 r-bglr=1.1.4 r-rrblup r-rcpp r-rcppeigen r-rcppparallel r-rcppziggurat r-matrix r-lme4 r-caret \
    r-doparallel r-foreach r-abind r-reshape2 r-coda r-mass r-tidyr r-remotes r-data.table r-coloc gemma
fi
