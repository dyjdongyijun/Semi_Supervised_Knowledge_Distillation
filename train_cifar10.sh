#!/bin/bash
# wandb artifact cache cleanup 16MB
# wandb offline/online
# wandb sync
# wandb sync --clean

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
# max_free_memory_gpu_id=7
echo "The GPU with the maximum free memory is ID: $max_free_memory_gpu_id"


batch_size=64
dataset=cifar10
num_labeled=$[10*4]
total_steps=$[2**17]
T_amp=2
rkd_lambda=1e-3
rkd_edge=cos
teacher_arch=densenet161
teacher_pretrain=cifar10

labeler=class
augstrength=10
percentunl=100

# FixMatch + RKD
python train.py --seed 5 --gpu_id $max_free_memory_gpu_id --batch_size $batch_size \
    --dataset $dataset --num_labeled $num_labeled --arch wideresnet \
    --total_steps $total_steps --expand_labels --T_amp $T_amp\
    --labeler $labeler --augstrength $augstrength --percentunl $percentunl \
    --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain --teacher_mode offline \
    --rkd_lambda $rkd_lambda --rkd_edge $rkd_edge

# FixMatch
# python train.py --seed 5 --gpu_id $max_free_memory_gpu_id --batch_size $batch_size \
#     --dataset $dataset --num_labeled $num_labeled --arch wideresnet \
#     --total_steps $total_steps --expand_labels \
#     --labeler $labeler --augstrength $augstrength --percentunl $percentunl \
#     --teacher_arch $teacher_arch --teacher_pretrain $teacher_pretrain --teacher_mode offline