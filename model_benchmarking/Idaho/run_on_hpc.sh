#!/bin/bash
#SBATCH -J runIdahoDataOnHPC
#SBATCH -A eecs 
#SBATCH -p share
#SBATCH -t 4-00:00:00   
#SBATCH --gres=gpu:1         
#SBATCH --mem=128G  
#SBATCH -c 16
module load cuda/11.8
python -u classification.py
