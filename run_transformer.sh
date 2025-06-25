#!/bin/bash
#SBATCH --job-name=transformer_small
#SBATCH --account=def-majewski
#SBATCH --time=00:30:00            # 30 minutes is more than enough
#SBATCH --cpus-per-task=2          # Only 2 CPUs needed for data loading
#SBATCH --mem=8G                   # Should be plenty for 100 samples
#SBATCH --gres=gpu:1               # Keep GPU to test model runs
#SBATCH --output=logs/transformer_%j.out
#SBATCH --error=logs/transformer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

source ~/deephistone/myproject/bin/activate
cd ~/deephistone
python -u train_5fold_transformer.py
