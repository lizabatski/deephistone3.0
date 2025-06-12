#!/bin/bash
#SBATCH --job-name=convert_datasets
#SBATCH --account=def-majewski
#SBATCH --time=02:00:00             # Increased from 1 hour
#SBATCH --mem=64G                   # Increased from 16G for 572MB files
#SBATCH --cpus-per-task=4
#SBATCH --array=1-7
#SBATCH --output=logs/convert_%A_%a.out
#SBATCH --error=logs/convert_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to the correct directory
cd /home/ekourb/deephistone/deephistone_conversion

# Define the datasets to process
EPIGENOME_IDS=("E005" "E005" "E005" "E005" "E005" "E005" "E005")
MARKERS=("H3K4me1" "H3K4me3" "H3K36me3" "H3K27me3" "H3K9me3" "H3K9ac" "H3K27ac")

# Get the current task's parameters
EPIGENOME_ID=${EPIGENOME_IDS[$SLURM_ARRAY_TASK_ID-1]}
MARKER=${MARKERS[$SLURM_ARRAY_TASK_ID-1]}

echo "Starting conversion job $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Converting: $EPIGENOME_ID - $MARKER"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Started at: $(date)"

# Load required modules (FIXED)
module load python/3.11.5
module load scipy-stack

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Print system information
echo "Python version: $(python --version)"
echo "Available memory: $(free -h)"
echo "CPU count: $(nproc)"
echo "Current directory: $(pwd)"

# Run the conversion
echo "Starting conversion..."
python convert_single.py "$EPIGENOME_ID" "$MARKER"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully for $EPIGENOME_ID-$MARKER"
    echo "Finished at: $(date)"
else
    echo "Conversion failed for $EPIGENOME_ID-$MARKER"
    echo "Failed at: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
