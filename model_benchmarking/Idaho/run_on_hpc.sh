#!/bin/bash
#SBATCH -J runIdahoDataOnHPC
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G   
#SBATCH -c 16
module load cuda/10.2
python classification.py
