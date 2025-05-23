#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:t4:3
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate ai
module load cuda/12.1
python -u train_mlp.py --ld=$1 
