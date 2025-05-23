#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:3060:1
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate ai
module load cuda/12.1
python -u train_rd_cgan.py --ld=$1
python -u images/generate_gif.py --ld=$1 
rm images/images_epoch*.png
mv *.gif images/
