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
PATH=$PATH:~/bin/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bin/lib
python3 run_downstream.py -n KS_necobert_eng -m train -u necobert_eng -d speech_commands
