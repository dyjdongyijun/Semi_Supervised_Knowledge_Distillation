#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

total_steps=$[2**17]
num_labeled=40

python train.py --seed 5 \
    --dataset cifar10 --num_labeled $num_labeled --arch wideresnet \
    --total_steps $total_steps --expand_labels \
    
