#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g ge43
#PJM -j
module load gcc/8.3.1
module load cuda/12.2
module load cudnn
cd /work/ge43/e43001/s3prl
source venv/bin/activate
cd s3prl

for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    # The default config is "downstream/emotion/config.yaml"
    python3 run_downstream.py -n ER_rinna_hubert_$test_fold -m train -u hf_hubert_rinna -d emotion -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/ExpName_$test_fold/dev-best.ckpt
done