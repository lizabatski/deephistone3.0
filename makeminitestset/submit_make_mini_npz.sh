#!/bin/bash
#SBATCH --job-name=mini_npz
#SBATCH --account=def-majewski
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=mini_npz.out
#SBATCH --error=mini_npz.err

source ~/deephistone/myproject/bin/activate

# move to testing directory
cd ~/deephistone/testing

# run mini generation script
python make_mini_npz.py
