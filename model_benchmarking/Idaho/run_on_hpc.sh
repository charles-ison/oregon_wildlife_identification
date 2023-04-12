#!/bin/bash
#SBATCH -J runIdahoDataOnHPC
#SBATCH -A eecs 
#SBATCH -p eecs
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G   
#SBATCH -c 16
module load cuda/11.8
python -u classification.py
