#!/bin/bash

python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 4 \
    --abits 4 \
    --model /localssd/hyzhang/llama-2-7b-hf \
    --alpha 0.6 \
    --smooth \
    --lac 1.2 \
    --swc 1.2 \
    --eval_ppl \
    --task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa\
    --calib_dataset pile \
    # --lwc \
    # --symmetric \
    



    # --lac 0.9 \
    # --swc 0.8 \