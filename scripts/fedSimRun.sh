#!/bin/bash
#PBS -lwalltime=24:0:0
#PBS -lselect=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -N dec_basic_le_1_e1_a2

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate pytorch_env

module load tools/prod

python3 src/main.py --exact_diffusion --decentralized --edge_prob 0.2 --model=resnet --gpu=0 --iid=1 --batch_size=128 --local_bs 256 --local_ep 1 --epochs=2 --finetuning_epoch 1 --log_file_name "skew_ssl_comm" \
                                                                          --lr 0.01 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 2 --log_directory "comm_scripts" 
