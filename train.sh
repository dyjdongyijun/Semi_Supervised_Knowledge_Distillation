#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

total_steps=$[2**17]
num_labeled=400

python train.py \
    --dataset cifar10 --num_labeled $num_labeled --arch wideresnet \
    --total_steps $total_steps --expand_labels \
    --seed 5
