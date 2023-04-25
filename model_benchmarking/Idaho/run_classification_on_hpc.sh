#!/bin/bash
#SBATCH -J run_idaho_data_on_HPC
#SBATCH -A eecs 
#SBATCH -p gpu
#SBATCH -t 7-00:00:00   
#SBATCH --gres=gpu:4         
#SBATCH --mem=128G  
#SBATCH -c 16
#SBATCH --constraint=rtx8000
module load cuda/11.8
python -u classification.py
