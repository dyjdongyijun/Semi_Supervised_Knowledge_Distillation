#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

total_steps=$[2**17]
num_labeled=40
rkd_lambda=1e-3
rkd_edge=cos

python train.py --seed 5 --gpu_id 0 \
    --dataset cifar10 --num_labeled $num_labeled --arch wideresnet \
    --total_steps $total_steps --expand_labels \
    --rkd_lambda $rkd_lambda --rkd_edge $rkd_edge --teacher_arch densenet161 --teacher_mode offline \
    --rkd_downweight mask --rkd_mask_clip 0.2