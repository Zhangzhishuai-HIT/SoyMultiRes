#!/bin/bash
# Launch a command on gpu10, GPU 0 ONLY (user rule since 2026-11-09), in the background, low priority.
# Launched only if GPU 0 has at least NEED_MB + HEADROOM MiB free (HEADROOM env var, default 4096; the user allowed a
# second MT-Boost process with HEADROOM=2048 on 2026-11-12); never touches other processes.
# The card is selected by UUID, never by ordinal: when a card drops off the bus (GPUs 0 and 1 did on 2026-11-11 14:48
# after NVSwitch link faults) CUDA renumbers the rest, and CUDA_VISIBLE_DEVICES=0 would silently land on another card.
# With the UUID a missing card means no device (safe failure).
# usage: launch_gpu0.sh NEED_MB LOGNAME -- command args...
set -euo pipefail
NEED=$1; NAME=$2; shift 2
HEADROOM=${HEADROOM:-4096}
[ "$1" == "--" ] && shift
ROOT=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
mkdir -p "$ROOT/logs/gpu"
CMD=$(printf '%q ' "$@")
UUID=GPU-ab1c1594-cdd6-628f-5856-f61653d27bd0   # physical card 0 (PCI 0000:10:00.0)
ssh -o BatchMode=yes gpu10 "
  cd $ROOT
  FREE=\$(nvidia-smi -i $UUID --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{print \$1-\$2}')
  [ -n \"\$FREE\" ] || { echo \"GPU 0 ($UUID) not visible to nvidia-smi; not launching $NAME\"; exit 4; }
  if [ \"\$FREE\" -lt \$(( $NEED + $HEADROOM )) ]; then echo \"GPU 0 has only \${FREE} MiB free; not launching $NAME\"; exit 3; fi
  echo \"launch $NAME on GPU 0 (free \${FREE} MiB)\"
  CUDA_VISIBLE_DEVICES=$UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
    nohup nice -n 10 $CMD > logs/gpu/$NAME.log 2>&1 < /dev/null &
  echo \$! > logs/gpu/$NAME.pid
"
