#!/bin/bash
#SBATCH --job-name=5fold_full
#SBATCH --account=def-majewski
#SBATCH --time=24:00:00               # 24 hours for full dataset 
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --output=5fold_full_%j.out
#SBATCH --error=5fold_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


# module load python/3.11.5
# module load cuda/11.8
# module load scipy-stack

#  Activate your virtual environment
source ~/deephistone/myproject/bin/activate


# pip install sympy

# move to your working directory
cd ~/deephistone

# set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#  print useful job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Checking GPU status..."
nvidia-smi

echo "----------------------------------------"
echo "Starting training script..."
python train_5fold.py
echo "----------------------------------------"
echo "Finished at: $(date)"
echo "Job completed successfully!"
