#!/bin/bash
#SBATCH -J run_idaho_data_on_HPC
#SBATCH -A eecs 
#SBATCH -p gpu
#SBATCH -t 7-00:00:00   
#SBATCH --gres=gpu:2         
#SBATCH --mem=128G  
#SBATCH -c 4
#SBATCH --constraint=rtx8000
#SBATCH -o run_idaho_data_on_HPC.txt
module load cuda/11.8
python -u classification.py
