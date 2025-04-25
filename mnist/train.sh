#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate ai
module load cuda/12.1
python -u mlp_train.py 
