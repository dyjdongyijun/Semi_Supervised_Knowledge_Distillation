#!/bin/bash
# wandb artifact cache cleanup 16MB

dataset=cifar10
batch_size=10000

teacher_arch=densenet161
teacher_pretrain=cifar10
teacher_dim=10

CUDA_VISIBLE_DEVICES=0 python teach.py --dataset $dataset --batch_size $batch_size --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain --teacher_dim $teacher_dim