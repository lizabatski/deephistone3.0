#!/bin/bash
#SBATCH --job-name=gpu_test_100samples
#SBATCH --account=def-majewski
#SBATCH --time=00:30:00                  
#SBATCH --cpus-per-task=6               
#SBATCH --mem=32G 
#SBATCH --gres=gpu:1                        
#SBATCH --output=gpu_test_100samples_%j.out
#SBATCH --error=gpu_test_100samples_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# Activate your virtual environment
source ~/deephistone/myproject/bin/activate

# Move to your working directory
cd ~/deephistone

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Running MINI dataset (100 samples)"
echo "----------------------------------------"

# Run the training script
python train_5fold_balance.py

echo "----------------------------------------"
echo "Finished at: $(date)"
echo "Job completed successfully!"