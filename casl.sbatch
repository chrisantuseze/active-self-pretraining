#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1
module load cuda/11.0
module load anaconda3/2019.03
python3 main.py
