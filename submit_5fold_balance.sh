#!/bin/bash
#SBATCH --job-name=5fold_chr1
#SBATCH --account=def-majewski
#SBATCH --time=4:00:00               # Should be plenty for chr1
#SBATCH --cpus-per-task=4            # 2–4 is enough unless you parallelized
#SBATCH --mem=32G                    # Way more than you need, but safe
#SBATCH --gres=gpu:1                 # Keep 1 GPU for acceleration
#SBATCH --output=5fold_chr1_%j.out
#SBATCH --error=5fold_chr1_%j.err
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
python train_5fold_balance.py --output_dir results/5fold_full_cv_balanced
echo "----------------------------------------"
echo "Finished at: $(date)"
echo "Job completed successfully!"
