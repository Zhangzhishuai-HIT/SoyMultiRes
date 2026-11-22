#!/bin/bash
# Start the outer-CV1 TraitBoost v2 queue on gpu10 card 0 once the cv1_inner queue has finished (one GPU process at a
# time while another project shares the card). Outer results are not to be read before the inner gate has been judged.
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
until grep -q "queue done" logs/gpu/gpu_mtb_v2_inner.log 2>/dev/null; do sleep 60; done
for i in $(seq 1 120); do
  if jobs/launch_gpu0.sh 5120 gpu_mtb_v2_outer -- jobs/gpu0_queue.sh jobs/lists/gpu_mtb_v2_outer.txt 1; then exit 0; fi
  sleep 300                                   # card busy: retry every 5 minutes for up to 10 hours
done
