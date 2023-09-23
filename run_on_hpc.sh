#!/bin/bash
#SBATCH -J run_on_HPC
#SBATCH -A eecs 
#SBATCH -p dgx
#SBATCH -t 7-00:00:00   
#SBATCH --gres=gpu:1      
#SBATCH --mem=120G  

#SBATCH -o model_benchmarking/run_logs/logs.out
#SBATCH -e model_benchmarking/run_logs/logs.err

# load env
source env/bin/activate

module load python/3.10 cuda/11.7

python3 -u model_benchmarking/batch_animal_count_testing.py
