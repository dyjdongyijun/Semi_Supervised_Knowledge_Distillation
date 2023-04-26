#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

dataset=cifar10
num_labeled=40
teacher_arch=densenet161
rkd_lambda=1e-3
rkd_edge=cos
total_steps=$[2**17]

python train.py --seed 5 --gpu_id 0 \
    --dataset $dataset --num_labeled $num_labeled --arch wideresnet \
    --total_steps $total_steps --expand_labels \
    --rkd_lambda $rkd_lambda --rkd_edge $rkd_edge --teacher_arch $teacher_arch --teacher_mode offline