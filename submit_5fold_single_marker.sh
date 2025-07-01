#!/bin/bash
#SBATCH --account=def-majewski          
#SBATCH --job-name=histone_marker_train
#SBATCH --output=logs/marker_%A_%a.out
#SBATCH --error=logs/marker_%A_%a.err
#SBATCH --mail-user=your.email@domain.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                     
#SBATCH --array=0-6                      

# Load modules (adjust to your environment)
module load python/3.9
source ~/your_venv/bin/activate          # Or load your conda env

# Define output directory root
ROOT_OUTDIR=results/single_marker_runs

# Run training for the marker corresponding to this task index
python train_single_marker.py \
    --marker_index $SLURM_ARRAY_TASK_ID \
    --output_dir $ROOT_OUTDIR/marker_${SLURM_ARRAY_TASK_ID}
