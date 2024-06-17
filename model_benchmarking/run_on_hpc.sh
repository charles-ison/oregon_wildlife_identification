#!/bin/bash
#SBATCH -J run_on_HPC
#SBATCH -A eecs 
#SBATCH -p dgx2
#SBATCH -t 5-00:00:00 
#SBATCH --gres=gpu:2
#SBATCH --mem=120G  

#SBATCH -o ../run_logs/logs.out
#SBATCH -e ../run_logs/logs.err

# load env
source ../env/bin/activate

module load python/3.10 cuda/11.7

python3 -u batch_animal_count_training.py
