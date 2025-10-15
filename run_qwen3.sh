# python main.py \
#     --block_size 32 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits 4 \
#     --abits 4 \
#     --model /localssd/hyzhang/qwen3-32b \
#     --alpha 0.6 \
#     --smooth \
#     --lac 1.15 \
#     --swc 1.15 \
#     --task piqa
#     # --task arc_easy,arc_challenge,winogrande,piqa \


python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 16 \
    --abits 16 \
    --model /localssd/hyzhang/qwen3-32b \
    --alpha 0.6 \
    --smooth \
    --lac 1.15 \
    --swc 1.15 \
    --task triviaqa \

python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 4 \
    --abits 4 \
    --model /localssd/hyzhang/qwen3-32b \
    --alpha 0.6 \
    --smooth \
    --lac 1.15 \
    --swc 1.15 \
    --task triviaqa \
    # --task piqa
    # --task arc_easy,arc_challenge,winogrande,piqa \
    # --task hellaswag \
    # --task arc_easy,arc_challenge,winogrande,piqa \
    # --task piqa
    # --task arc_easy,arc_challenge,winogrande,piqa \

