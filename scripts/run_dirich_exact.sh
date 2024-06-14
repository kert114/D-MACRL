#!/bin/bash#!/bin/bash

# Activate the virtual environment
source /root/fed_sim/venv/bin/activate


dirichlet_beta=true
epochs=96
local_ep=16
num_users=10


edge_prob=0
decentralized=true 
exact_diffusion=true
unique_id=$(date +'%m-%d_%H:%M')
job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
output_dir="dirichlet_exact/${job_name}"
mkdir -p $output_dir

time python3 src/main.py  --dirichlet --dir_beta 0.01 --exact_diffusion --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
                      --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"


# edge_prob=0.5
# decentralized=true 
# exact_diffusion=true
# unique_id=$(date +'%m-%d_%H:%M')
# job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# output_dir="dirichlet_short/${job_name}"
# mkdir -p $output_dir

# time python3 src/main.py  --dirichlet --dir_beta 0.01 --exact_diffusion --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
#                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# echo "Job outputs and errors are saved in ${output_dir}"


# unique_id=$(date +'%m-%d_%H:%M')
# decentralized=true 
# exact_diffusion=true
# edge_prob=0.3
# job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# output_dir="dirichlet_short/${job_name}"
# mkdir -p $output_dir

# time python3 src/main.py  --dirichlet --dir_beta 0.01 --exact_diffusion --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
#                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"


# echo "Job outputs and errors are saved in ${output_dir}"

# edge_prob=0.1
# exact_diffusion=true
# decentralized=true 
# unique_id=$(date +'%m-%d_%H:%M')
# job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# output_dir="dirichlet_short/${job_name}"
# mkdir -p $output_dir

# time python3 src/main.py  --dirichlet --dir_beta 0.01 --exact_diffusion --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
#                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# echo "Job outputs and errors are saved in ${output_dir}"
# # edge_prob=1
# # exact_diffusion=false
# # unique_id=$(date +'%m-%d_%H:%M')
# # job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# # output_dir="dirichlet_short/${job_name}"
# # mkdir -p $output_dir

# # time python3 src/main.py  --dirichlet --dir_beta 0.01 --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
# #                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# # edge_prob=0.8
# # exact_diffusion=false
# # unique_id=$(date +'%m-%d_%H:%M')
# # job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# # output_dir="dirichlet_short/${job_name}"
# # mkdir -p $output_dir

# # time python3 src/main.py  --dirichlet --dir_beta 0.01 --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
# #                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# # edge_prob=0.5
# # exact_diffusion=false
# # unique_id=$(date +'%m-%d_%H:%M')
# # job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# # output_dir="dirichlet_short/${job_name}"
# # mkdir -p $output_dir

# # time python3 src/main.py --dirichlet --dir_beta 0.01 --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
# #                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# # echo "Job outputs and errors are saved in ${output_dir}"


# # edge_prob=0.3
# # exact_diffusion=false
# # unique_id=$(date +'%m-%d_%H:%M')
# # job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# # output_dir="dirichlet_short/${job_name}"
# # mkdir -p $output_dir

# # time python3 src/main.py --dirichlet --dir_beta 0.01 --decentralized --edge_prob $edge_prob --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
# #                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# # echo "Job outputs and errors are saved in ${output_dir}"


# # edge_prob=0
# # exact_diffusion=false
# # decentralized=false 
# # unique_id=$(date +'%m-%d_%H:%M')
# # job_name="${unique_id}_drich-${dirichlet_beta}_dec-${decentralized}_ED-${exact_diffusion}_ep${edge_prob}_e${epochs}_le${local_ep}_a${num_users}"
# # output_dir="dirichlet_short/${job_name}"
# # mkdir -p $output_dir

# # time python3 src/main.py --dirichlet --dir_beta 0.01 --model=resnet --gpu=1 --iid=0 --batch_size=256 --local_bs=256 --local_ep=$local_ep --epochs=$epochs --log_file_name "comm_scripts/best_linear_statistics_skew_ssl_comm.csv" \
# #                       --lr=0.001 --optimizer=adam --backbone=resnet18 --num_users=$num_users  --log_directory "$output_dir" > "${output_dir}/output_log.txt" 2> "${output_dir}/error.txt"

# # echo "Job outputs and errors are saved in ${output_dir}"


# # exit