#!/bin/bash
#SBATCH --job-name=dnase_vis
#SBATCH --account=def-majewski
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=dnase_vis_%j.out
#SBATCH --error=dnase_vis_%j.err

source ~/deephistone/myproject/bin/activate
cd ~/deephistone

python visualize_dnase_dna_alignment.py data/final/E003_all_markers_merged.npz --num 5
