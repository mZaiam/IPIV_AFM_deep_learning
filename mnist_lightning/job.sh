#!/bin/bash
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -p batch-AMD
##SBATCH --time=12:00:00

source ~/.bashrc
conda activate pyscf
module load openmpi/4.1.1
module load orca/5.0.4


ipaddress=$(ip addr | grep 172 | awk 'NR==1{print $2}' | sed 's!/23!!g' | sed 's!/0!!g')
echo $ipaddress

jupyter-notebook --ip=$ipaddress


