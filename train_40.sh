#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

total_steps=$[2**17]
rkd_lambda=1e-3
rkd_edge=cos
seed=5
labeler=active-fl


for percentunl in 50 80 100
    do 
    for m in 2 6 10
        do
        # cifar10 -- both FM and RKD
        python train.py --seed $seed --percentunl $percentunl --augstrength $m --labeler $labeler\
            --dataset cifar10  --num_labeled $num_labeled --arch wideresnet \
            --total_steps $total_steps --expand_labels \
            --rkd_lambda $rkd_lambda --rkd_edge $rkd_edge --teacher_arch densenet161 --teacher_mode offline --lambda_u 1  

         # cifar10 -- only FM
         python train.py --seed $seed --percentunl $percentunl --augstrength $m --labeler $labeler\
            --dataset cifar10  --num_labeled $num_labeled --arch wideresnet \
            --total_steps $total_steps --expand_labels \
            --rkd_lambda 0 --rkd_edge $rkd_edge --teacher_arch densenet161 --teacher_mode offline --lambda_u 1
    done
done
