#!/bin/bash

# Define variables
decentralized=true # 1 for decentralized, 0 for federated learning
exact_diffusion=false
edge_prob=1 
local_ep=5
epochs=30
num_users=5

# Generate a unique identifier based on the current date and time
unique_id=$(date +'%m-%d_%H:%M')

# Include variables in the job name
job_name="${unique_id}_iid_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"

#PBS -N $job_name

# Activate the virtual environment
source /root/fed_sim/venv/bin/activate

output_dir="results/${job_name}"
mkdir -p $output_dir

## Running Python script with redirection of standard output and standard error to files including variable values
if [ "$decentralized" = true ] && [ "$exact_diffusion" = true ]; then
  python3 src/main.py --decentralized --exact_diffusion --edge_prob $edge_prob --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "/skew_ssl_comm" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --frac=1 --y_partition_skew --y_partition_ratio=0 --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"
elif [ "$decentralized" = true ] && [ "$exact_diffusion" = false ]; then
  python3 src/main.py --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "${output_dir}/skew_ssl_comm" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --frac=1 --y_partition_skew --y_partition_ratio=0 --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"
else
  python3 src/main.py --edge_prob $edge_prob --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "${output_dir}/skew_ssl_comm" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --frac=1 --y_partition_skew --y_partition_ratio=0 --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"
fi

echo "Job outputs and errors are saved in ${output_dir}"
