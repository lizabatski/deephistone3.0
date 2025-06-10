#!/bin/bash
#SBATCH --account=def-yourpi  # Replace with your account
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5
#SBATCH --job-name=e003_histones

# Load modules
module load python/3.9 scipy-stack
source ~/your_env/bin/activate  # Replace with your environment

# Define your 5 markers
declare -a MARKERS=("H3K36me3" "H3K27me3" "H3K9me3" "H3K27ac" "H3K9ac")
MARKER=${MARKERS[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing E003-$MARKER on $(hostname) at $(date)"

# Run your existing code
python -c "
from your_deephistone_script import run_single_combination, setup_logging
logger = setup_logging()
output_path, success = run_single_combination('E003', '$MARKER', logger)
print('SUCCESS' if success else 'FAILED')
"
