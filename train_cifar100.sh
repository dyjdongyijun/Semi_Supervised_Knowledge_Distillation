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

batch_size=64
dataset=cifar100
num_labeled=$[100*4]
total_steps=$[2**17]
threshold=0.8
rkd_lambda=1e-3
rkd_edge=cos
teacher_arch=resnet50w5
teacher_pretrain=swav

function train_unit {
    if [ "$#" -ne 4 ]; then
        echo "Required inputs: seed, labeler, augstrength, percentunl"
        return 1
    fi

    seed=$1
    labeler=$2
    augstrength=$3
    percentunl=$4

    python train.py --seed $seed --gpu_id $max_free_memory_gpu_id --batch_size $batch_size \
        --dataset $dataset --num_labeled $num_labeled --arch wideresnet \
        --total_steps $total_steps --expand_labels --threshold $threshold \
        --labeler $labeler --augstrength $augstrength --percentunl $percentunl \
        --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain --teacher_mode offline \
        --rkd_lambda $rkd_lambda --rkd_edge $rkd_edge # --amp --opt_level O2 --wdecay 0.001 

    python train.py --seed $seed --gpu_id $max_free_memory_gpu_id --batch_size $batch_size \
        --dataset $dataset --num_labeled $num_labeled --arch wideresnet \
        --total_steps $total_steps --expand_labels --threshold $threshold \
        --labeler $labeler --augstrength $augstrength --percentunl $percentunl \
        --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain --teacher_mode offline # --amp --opt_level O2 --wdecay 0.001 
}

seed=42
labeler=class
augstrength=10
percentunl=100  
train_unit $seed $labeler $augstrength $percentunl

seed=42
labeler=active-fl
augstrength=10
percentunl=100  
train_unit $seed $labeler $augstrength $percentunl  