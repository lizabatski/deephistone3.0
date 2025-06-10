#!/bin/bash
#SBATCH --job-name=deephistone_E003_v2
#SBATCH --account=def-majewski
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/deephistone_E003_v2_%j.out
#SBATCH --error=logs/deephistone_E003_v2_%j.err

# Load modules
module load python/3.11.5
module load scipy-stack

# immediate output
export PYTHONUNBUFFERED=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/ekourb/deephistone

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "Files check:"
ls -la raw/E003-H3K4me3.narrowPeak
ls -la raw/hg19.fa
echo "=========================================="


python -u run_deephistone_hpc.py E003

echo "Job completed at: $(date)"

