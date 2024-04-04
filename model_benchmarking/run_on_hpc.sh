#!/bin/bash
#SBATCH -J run_on_HPC
#SBATCH -A eecs 
#SBATCH -p eecs
#SBATCH -t 2-00:00:00   
#SBATCH --gres=gpu:4
#SBATCH --mem=40G  

#SBATCH -o ../run_logs/logs.out
#SBATCH -e ../run_logs/logs.err

# load env
source ../env/bin/activate

module load python/3.10 cuda/11.7

python3 -u batch_animal_count_testing.py
