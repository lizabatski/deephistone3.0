#!/bin/bash
#SBATCH --job-name=deephistone_parallel
#SBATCH --account=def-majewski
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=1-7                   
#SBATCH --output=logs/deephistone_%A_%a.out
#SBATCH --error=logs/deephistone_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL    
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


EPIGENOME="E003"  


echo "=== $EPIGENOME Parallel Processing ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname), Start: $(date)"

# load modules
module load python/3.9 scipy-stack

# 7 markers in correct order
MARKERS=("H3K4me1" "H3K4me3" "H3K36me3" "H3K27me3" "H3K9me3" "H3K9ac" "H3K27ac")
MARKER=${MARKERS[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing $EPIGENOME-$MARKER (${SLURM_ARRAY_TASK_ID}/7)"

python -c "
from deephistone_pipeline_all import run_single_combination, setup_logging
import time, sys

start_time = time.time()
logger = setup_logging()

try:
    output_path, success = run_single_combination('$EPIGENOME', '$MARKER', logger)
    elapsed = time.time() - start_time
        
    if success:
        print(f'SUCCESS: $EPIGENOME-$MARKER completed in {elapsed/3600:.2f} hours')
        print(f'Output: {output_path}')
    else:
        print(f'FAILED: $EPIGENOME-$MARKER')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: $EPIGENOME-$MARKER - {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "=== Completed $EPIGENOME-$MARKER at $(date) ==="