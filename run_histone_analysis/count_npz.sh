#!/bin/bash
#SBATCH --job-name=histone_count_E005
#SBATCH --output=histone_count_E005_%j.out
#SBATCH --error=histone_count_E005_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@domain.com

# Load required modules
module load python/3.9
module load numpy

# Set up environment
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Configuration
EPIGENOME_ID="E005"  # Change this to analyze different epigenomes

# Define paths - ADJUST THESE TO YOUR ENVIRONMENT
SCRIPT_DIR="$HOME/deephistone/run_histone_analysis"    # Directory containing your Python scripts
DATA_DIR="$HOME/deephistone/data/converted/${EPIGENOME_ID}"  # Directory containing NPZ files
RESULTS_DIR="$HOME/deephistone/results/histone_counts"       # Output directory

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Change to script directory
cd $SCRIPT_DIR

# Run the analysis using the wrapper script
echo "Starting histone site counting analysis for $EPIGENOME_ID..."
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"

python run_histone_analysis.py \
    --epigenome $EPIGENOME_ID \
    --data_dir $DATA_DIR \
    --output_dir $RESULTS_DIR \
    --verbose

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo "Output files created in: $RESULTS_DIR"
    echo ""
    echo "Generated files:"
    ls -la $RESULTS_DIR/histone_counts_${EPIGENOME_ID}*
else
    echo "Analysis failed with exit code: $?"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"