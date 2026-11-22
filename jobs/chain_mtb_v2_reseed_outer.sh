#!/bin/bash
# After all five outer-CV1 TraitBoost fits are done: two lock-based runners add refit seeds (fold 0: s2-s4, folds 1-4:
# s3-s4) for the final definition and the 4-refit ablations (FREEZE_DEVIATIONS 2026-11-12 19:45).
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
L=jobs/lists/gpu_mtb_v2_reseed_outer.txt
launch() { for i in $(seq 1 60); do HEADROOM=1900 jobs/launch_gpu0.sh 3584 "$1" -- jobs/gpu0_queue.sh $L 1 && return 0; sleep 300; done; return 1; }
until [ -f logs/gpu/mtb_v2_cv1_f0.done ]; do sleep 60; done
launch gpu_mtb_v2_reseed_outer_A            # fold 0 can start as soon as its fit is done
until [ "$(ls logs/gpu/mtb_v2_cv1_f{1,2,3,4}.done 2>/dev/null | wc -l)" -ge 3 ]; do sleep 60; done
launch gpu_mtb_v2_reseed_outer_B
