#!/bin/bash

batch_size=1000

# dataset=cifar10
# teacher_arch=densenet161
# teacher_pretrain=cifar10

dataset=timagenet200
teacher_arch=resnet50w2
teacher_pretrain=swav

CUDA_VISIBLE_DEVICES=0 python teach.py --dataset $dataset --batch_size $batch_size --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain