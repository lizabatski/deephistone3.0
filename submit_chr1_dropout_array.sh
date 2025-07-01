#!/bin/bash
#SBATCH --job-name=chr1_dropout_sweep
#SBATCH --account=def-majewski
#SBATCH --time=4:00:00              
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=logs/chr1_dropout_%A_%a.out
#SBATCH --error=logs/chr1_dropout_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# Activate virtual environment
source ~/deephistone/myproject/bin/activate

# Move to project directory
cd ~/deephistone

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Define dropout rates
DROPOUT_RATES=(0.0 0.2)
DROPOUT=${DROPOUT_RATES[$SLURM_ARRAY_TASK_ID]}

# Auto-generate output directory
OUTDIR=results/chr1_dropout_${DROPOUT}

echo "----------------------------------------"
echo "Running dropout=${DROPOUT}"
python train_5fold.py --dropout $DROPOUT --output_dir $OUTDIR
echo "----------------------------------------"
echo "Finished at: $(date)"
