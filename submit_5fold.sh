#!/bin/bash
#SBATCH --job-name=5fold_full
#SBATCH --account=def-majewski
#SBATCH --time=24:00:00              # 24 hours for full dataset
#SBATCH --cpus-per-task=4            # More CPUs for data loading
#SBATCH --mem=128G                   # Much more memory for full dataset
#SBATCH --gres=gpu:1                 # GPU recommended for full dataset
#SBATCH --output=5fold_full_%j.out
#SBATCH --error=5fold_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


# module load python/3.11.5
# module load cuda/11.8
# module load scipy-stack

# Activate your virtual environment
source ~/deephistone/myproject/bin/activate

pip install sympy

# Move to your working directory
cd ~/deephistone

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    nvidia-smi
fi
echo "----------------------------------------"

# Run the training script
python train_5fold.py

echo "----------------------------------------"
echo "Finished at: $(date)"
echo "Job completed successfully!"