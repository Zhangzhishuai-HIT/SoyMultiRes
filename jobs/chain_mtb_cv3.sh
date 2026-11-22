#!/bin/bash
# CV3 TraitBoost (16 units) on gpu10 card 0 after both all-data runners are done; two lock-based runners (the user
# allowed two TraitBoost processes). Largest units first.
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
L=jobs/lists/gpu_mtb_cv3.txt
launch() { for i in $(seq 1 60); do jobs/launch_gpu0.sh 7168 "$1" -- jobs/gpu0_queue_v2.sh $L 1 && return 0; sleep 300; done; return 1; }
until grep -q "queue done" logs/gpu/gpu_mtb_all_A.log 2>/dev/null && grep -q "queue done" logs/gpu/gpu_mtb_all_B.log 2>/dev/null; do sleep 60; done
launch gpu_mtb_cv3_A
sleep 20
launch gpu_mtb_cv3_B
