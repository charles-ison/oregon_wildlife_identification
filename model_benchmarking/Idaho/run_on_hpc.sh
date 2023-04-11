#!/bin/bash
#SBATCH -J runIdahoDataOnHPC
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G   
#SBATCH -c 16
#SBATCH -o runIdahoDataOnHPC.out 
#SBATCH -e runIdahoDataOnHPC.err
module load cuda/11.8
python -u classification.py
