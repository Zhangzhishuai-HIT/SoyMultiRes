#!/bin/bash
# Two lock-based runners for outer CV1 folds 1-4 on gpu10 card 0 (user allowed two MT-Boost processes, 2026-11-12):
# runner A when the inner reseed queue is done, runner B when outer fold 0 (started 18:41 by the old chain) is done.
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
L=jobs/lists/gpu_mtb_v2_outer_b.txt
launch() { for i in $(seq 1 60); do HEADROOM=1900 jobs/launch_gpu0.sh 3584 "$1" -- jobs/gpu0_queue.sh $L 1 && return 0; sleep 300; done; return 1; }
until grep -q "queue done" logs/gpu/gpu_mtb_reseed_a.log 2>/dev/null; do sleep 60; done
launch gpu_mtb_v2_outer_A
until [ -f logs/gpu/mtb_v2_cv1_f0.done ]; do sleep 60; done
sleep 30
launch gpu_mtb_v2_outer_B
