#!/bin/bash

# Get the output of nvidia-smi with GPU memory information
output=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader)
# Initialize variables for tracking the maximum free memory and its corresponding GPU ID
max_free_memory=-1
max_free_memory_gpu_id=-1
# Iterate through each line of the output to find the GPU with the maximum free memory
while IFS= read -r line; do
    gpu_id=$(echo $line | cut -d',' -f1)
    free_memory=$(echo $line | cut -d',' -f2 | tr -dc '0-9')

    if (( free_memory > max_free_memory )); then
        max_free_memory=$free_memory
        max_free_memory_gpu_id=$gpu_id
    fi
done <<< "$output"
echo "The GPU with the maximum free memory is ID: $max_free_memory_gpu_id"


batch_size=1000

# dataset=cifar10
# teacher_arch=densenet161
# teacher_pretrain=cifar10

dataset=cifar100
teacher_arch=resnet50w2
teacher_pretrain=swav

# dataset=timagenet200
# teacher_arch=resnet50w2
# teacher_pretrain=swav

CUDA_VISIBLE_DEVICES=$max_free_memory_gpu_id python teach.py --dataset $dataset --batch_size $batch_size --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain