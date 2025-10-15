python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 4 \
    --abits 4 \
    --model /localssd/hyzhang/llama-2-70b-hf \
    --alpha 0.6 \
    --smooth \
    --lac 1.15 \
    --swc 1.15 \
    --task hellaswag
    # --task arc_easy,arc_challenge,winogrande,piqa \



