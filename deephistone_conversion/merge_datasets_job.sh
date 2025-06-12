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



EPIGENOME="E005"  


echo "Starting $EPIGENOME dataset merge job: $(date)"
echo "Job ID: $SLURM_JOB_ID"


module load python/3.11.5 scipy-stack


cd /home/ekourb/deephistone

mkdir -p logs data/converted

# check input files
echo "Input files for $EPIGENOME:"
ls -lh data/converted/${EPIGENOME}/${EPIGENOME}_*_expected_format.npz

# count files
file_count=$(ls data/converted/${EPIGENOME}/${EPIGENOME}_*_expected_format.npz 2>/dev/null | wc -l)
echo "Found $file_count $EPIGENOME converted datasets"

if [ $file_count -ne 7 ]; then
    echo "ERROR: Expected 7 files, found $file_count"
    exit 1
fi

# cun the merge
echo "Starting $EPIGENOME merge process..."
python3 deephistone_conversion/merge_datasets.py \
    --input-dir data/converted/${EPIGENOME} \
    --output data/converted/${EPIGENOME}_all_markers_merged.npz

# check result
if [ $? -eq 0 ]; then
    echo "Merge completed successfully: $(date)"
    ls -lh data/converted/${EPIGENOME}_all_markers_merged.npz
else
    echo "Merge failed: $(date)"
    exit 1
fi

echo "Job completed: $(date)"