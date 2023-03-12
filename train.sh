#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

python train.py \
    --dataset cifar10 --num_labeled 4000 --arch wideresnet \
    --batch_size 64 --lr 0.03 --expand_labels \
    --seed 5

# --amp
# --arch
# --batch_size
# --dataset
# --ema_decay
# --eval_step
# --expand_labels
# --gpu_id
# --labeler
# --lambda_u
# --local_rank
# --lr
# --learning_rate
# --mu
# --nesterov
# --no_progress
# --num_labeled
# --num_workers
# --opt_level
# --out
# --pretrain_path
# --resume
# --rkd_edge
# --rkd_edge_min
# --rkd_lambda
# --rkd_norm
# --root
# --seed
# --start_epoch
# --T
# --teacher_arch
# --teacher_data
# --teacher_dim
# --teacher_mode
# --threshold
# --total_steps
# --use_ema
# --warmup
# --wdecay