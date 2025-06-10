#!/bin/bash
#SBATCH --job-name=deephistone_E003_parallel
#SBATCH --account=def-majewski
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=1-5                   # 5 jobs for 5 markers
#SBATCH --output=logs/deephistone_E003_%A_%a.out
#SBATCH --error=logs/deephistone_E003_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL   
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

echo "=== E003 Parallel Processing ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname), Start: $(date)"

# Load modules
module load python/3.9 scipy-stack

# Your 5 markers
MARKERS=("H3K36me3" "H3K27me3" "H3K9me3" "H3K27ac" "H3K9ac")
MARKER=${MARKERS[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing E003-$MARKER"

python -c "
from deephistone_pipeline_all import run_single_combination, setup_logging
import time, sys

start_time = time.time()
logger = setup_logging()

try:
    output_path, success = run_single_combination('E003', '$MARKER', logger)
    elapsed = time.time() - start_time
    
    if success:
        print(f' SUCCESS: E003-$MARKER completed in {elapsed/3600:.2f} hours')
        print(f'Output: {output_path}')
    else:
        print(f' FAILED: E003-$MARKER')
        sys.exit(1)
except Exception as e:
    print(f' ERROR: E003-$MARKER - {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "=== Completed E003-$MARKER at $(date) ==="
