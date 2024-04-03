#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:0:0
#SBATCH --mail-user=jyang297@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:V100l:1
module purge
# Change python version if necessary
module load StdEnv/2023
module load python/3.11
module load opencv/4.9

# Source the virtual env
cd ~/$projects/projects/def-jyzhao/jyang297
source ~/projects/def-jyzhao/jyang297/lstm/bin/activate

# Run python script here
cd ~/projects/def-jyzhao/jyang297/RIFE_LSTM_Context
torchrun train.py