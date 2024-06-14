#!/bin/bash#!/bin/bash

# Activate the virtual environment
source /root/fed_sim/venv/bin/activate


dirichlet_beta=false
decentralized=false 
exact_diffusion=false

num_users=10

epochs=250
local_ep=50
unique_id=$(date +'%m-%d_%H:%M')
job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
output_dir="fed_250/${job_name}"
mkdir -p $output_dir

time python3 src/main.py --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"


epochs=246
local_ep=41
unique_id=$(date +'%m-%d_%H:%M')
job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
output_dir="fed_250/${job_name}"
mkdir -p $output_dir

time python3 src/main.py  --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

echo "Job outputs and errors are saved in ${output_dir}"


epochs=245
local_ep=35
unique_id=$(date +'%m-%d_%H:%M')
job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
output_dir="fed_250/${job_name}"
mkdir -p $output_dir

time python3 src/main.py --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"



epochs=248
local_ep=31
unique_id=$(date +'%m-%d_%H:%M')
job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
output_dir="fed_250/${job_name}"
mkdir -p $output_dir

time python3 src/main.py  --model=resnet --gpu=1 --iid=1 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

echo "Job outputs and errors are saved in ${output_dir}"

