#!/bin/bash
#SBATCH -J runIdahoDataOnHPC
#SBATCH --gres=gpu:1         
#SBATCH --mem=32G   
#SBATCH -c 16
python classification.py
