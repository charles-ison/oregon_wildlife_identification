#!/bin/bash
#SBATCH -J run_idaho_data_on_HPC
#SBATCH -A eecs 
#SBATCH -p gpu
#SBATCH -t 7-00:00:00   
#SBATCH --gres=gpu:4       
#SBATCH --mem=128G  

#SBATCH -o model_benchmarking/Idaho/run_logs/logs.out
#SBATCH -e model_benchmarking/Idaho/run_logs/logs.err

# load env
source env/bin/activate

module load python/3.10 cuda/11.7

python3 model_benchmarking/Idaho/classification.py
