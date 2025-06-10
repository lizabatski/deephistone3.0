#!/bin/bash
#SBATCH --job-name=merge_deephistone
#SBATCH --account=def-majewski
#SBATCH --time=06:00:00              # 6 hours
#SBATCH --cpus-per-task=4            # 4 cores
#SBATCH --mem=120G                   # 120GB RAM (crucial!)
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

echo "Starting dataset merge job: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Available memory: 120GB"

# Load modules
module load python/3.11.5 scipy-stack

# Change to project directory
cd /home/ekourb/deephistone

# Create directories
mkdir -p logs data/final

# Check available memory
echo "Memory info:"
free -h

# Check input files in the correct location
echo "Input files in ../data/converted:"
ls -lh ../data/converted/*expected_format.npz

# Count input files
file_count=$(ls ../data/converted/*expected_format.npz 2>/dev/null | wc -l)
echo "Found $file_count converted datasets"

# Run the merge with correct input directory
echo "Starting merge process..."
python3 deephistone_conversion/merge_datasets.py \
    --input-dir data/converted \
    --output data/converted/E003_all_markers_merged.npz

# Check output
echo "Merge completed: $(date)"
echo "Output file info:"
ls -lh data/final/E003_all_markers_merged.npz

# Verify merged dataset
echo "Quick verification:"
python3 -c "
import numpy as np
try:
    data = np.load('data/final/E003_all_markers_merged.npz')
    print(f'Final dataset shape:')
    for key in data.files:
        print(f'  {key}: {data[key].shape}')
    print(f'Total positive labels: {int(data[\"label\"].sum()):,}')

    # Check file size
    import os
    size_gb = os.path.getsize('data/final/E003_all_markers_merged.npz') / (1024**3)
    print(f'File size: {size_gb:.2f} GB')

except Exception as e:
    print(f'Error verifying output: {e}')
"

echo "Job completed: $(date)"
