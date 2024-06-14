#!/bin/bash
#PBS -lwalltime=24:0:0
#PBS -lselect=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -N fed_el_20_e_300_a_5

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate pytorch_env

module load tools/prod

python3 src/main.py --model=resnet --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 20 --epochs=300 --log_file_name "skew_ssl_comm" \
                                                                          --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1  --y_partition_skew --y_partition_ratio 0 --log_directory "IID_scripts" --log_file_name "dirichlet_alpha_sl_featt"