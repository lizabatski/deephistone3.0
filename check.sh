#!/bin/bash
#SBATCH --job-name=diagnose_data
#SBATCH --time=15:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=diagnose_%j.out

echo "Starting diagnosis at $(date)"
python check.py
echo "Completed at $(date)"