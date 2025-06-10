#!/bin/bash
#SBATCH --job-name=deephistone_E003_final
#SBATCH --account=def-majewski
#SBATCH --time=15:00:00                
#SBATCH --cpus-per-task=12            # 12 cores for better performance
#SBATCH --mem=32G                     # 32GB memory
#SBATCH --output=logs/deephistone_E003_final_%j.out
#SBATCH --error=logs/deephistone_E003_final_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL   
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca  

# Print job information
echo "============================================================"
echo "DEEPHISTONE PIPELINE - E003 FULL GENOME PROCESSING"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 32GB"
echo "Account: def-majewski"
echo "Expected runtime: 10-14 hours"
echo "============================================================"

# Load required modules
echo "Loading modules..."
module load python/3.11.5      
module load scipy-stack

# Verify modules loaded
echo "Loaded modules:"
module list

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to project directory
cd /home/ekourb/deephistone
echo "Working directory: $(pwd)"

# Verify key files exist
echo ""
echo "Verifying required files..."

# Check pipeline file
if [ -f "deephistone_pipeline_all.py" ]; then
    echo "✓ Pipeline file: deephistone_pipeline_all.py"
else
    echo "✗ ERROR: deephistone_pipeline_all.py not found!"
    exit 1
fi

# Check genome file
if [ -f "raw/hg19.fa" ]; then
    genome_size=$(du -sh raw/hg19.fa | cut -f1)
    echo "✓ Genome file: raw/hg19.fa ($genome_size)"
else
    echo "✗ ERROR: raw/hg19.fa not found!"
    exit 1
fi

# Check sample ChIP-seq file
if [ -f "raw/E003-H3K4me3.narrowPeak" ]; then
    chip_size=$(du -sh raw/E003-H3K4me3.narrowPeak | cut -f1)
    echo "✓ Sample ChIP file: raw/E003-H3K4me3.narrowPeak ($chip_size)"
else
    echo "✗ ERROR: raw/E003-H3K4me3.narrowPeak not found!"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs
echo "✓ Created data/ and logs/ directories"

# Show disk space
echo ""
echo "Disk space check:"
df -h . | tail -1

echo ""
echo "============================================================"
echo "STARTING DEEPHISTONE PIPELINE"
echo "Processing E003 across 7 histone markers on full genome"
echo "============================================================"

# Run the pipeline
python3 -u << 'EOF'
import sys
import os
import time
import traceback

def main():
    print("Setting up pipeline environment...")
    
    # Ensure we're in the right directory
    os.chdir('/home/ekourb/deephistone')
    
    # Add scripts directory to path if it exists
    if os.path.exists('scripts'):
        sys.path.insert(0, 'scripts')
        print("✓ Added scripts/ to Python path")
    
    # Import the pipeline
    try:
        print("Importing pipeline modules...")
        from deephistone_pipeline_all import (
            DeepHistoneConfig, 
            setup_logging, 
            run_single_combination,
            validate_epigenome_files
        )
        print("✓ Pipeline modules imported successfully")
    except ImportError as e:
        print(f"✗ ERROR: Could not import pipeline: {e}")
        print("Available files:")
        os.system("ls -la *.py")
        return False
    
    # Configure pipeline for full genome processing
    print("\nConfiguring pipeline...")
    config = DeepHistoneConfig()
    config.TEST_MODE = False          # Full genome processing
    config.N_PROCESSES = 12           # Use all allocated cores
    config.SKIP_EXISTING = True       # Skip if output already exists
    config.CONTINUE_ON_ERROR = True   # Continue if one marker fails
    
    # Update the global configuration
    import deephistone_pipeline_all
    deephistone_pipeline_all.config = config
    
    print("✓ Pipeline configured for full genome processing")
    print(f"  Processes: {config.N_PROCESSES}")
    print(f"  Skip existing: {config.SKIP_EXISTING}")
    print(f"  Continue on error: {config.CONTINUE_ON_ERROR}")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting DeepHistone processing for E003")
    
    # Define processing targets
    epigenome_id = "E003"
    markers = [
        "H3K4me3", "H3K4me1", "H3K36me3", "H3K27me3", 
        "H3K9me3", "H3K27ac", "H3K9ac"
    ]
    
    print(f"\nProcessing targets:")
    print(f"  Epigenome: {epigenome_id}")
    print(f"  Markers: {markers}")
    print(f"  Total combinations: {len(markers)}")
    
    # Validate input files
    print(f"\nValidating input files for {epigenome_id}...")
    try:
        files_valid, missing_files = validate_epigenome_files(epigenome_id)
        if files_valid:
            print("✓ All required files found")
        else:
            print(f"✗ ERROR: Missing files: {missing_files}")
            return False
    except Exception as e:
        print(f"✗ ERROR: File validation failed: {e}")
        return False
    
    # Track processing results
    successful = []
    failed = []
    skipped = []
    total_start_time = time.time()
    
    # Process each marker
    for i, marker in enumerate(markers, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING MARKER {i}/{len(markers)}: {epigenome_id}-{marker}")
        print(f"{'='*80}")
        
        marker_start_time = time.time()
        
        try:
            # Check if output already exists
            output_path = config.get_output_path(epigenome_id, marker)
            if os.path.exists(output_path):
                print(f" Output already exists: {output_path}")
                size_mb = os.path.getsize(output_path) / (1024*1024)
                print(f"   File size: {size_mb:.1f} MB")
                skipped.append((marker, output_path))
                continue
            
            # Run the pipeline for this marker
            print(f" Starting processing...")
            result_path, success = run_single_combination(epigenome_id, marker, logger)
            
            marker_elapsed = time.time() - marker_start_time
            
            if success and result_path:
                successful.append((marker, result_path))
                
                # Get output file information
                if os.path.exists(result_path):
                    size_mb = os.path.getsize(result_path) / (1024*1024)
                    
                    # Quick validation of output
                    try:
                        import numpy as np
                        data = np.load(result_path, allow_pickle=True)
                        n_samples = len(data['sequences'])
                        n_positive = int(data['labels'].sum())
                        n_negative = len(data['labels']) - n_positive
                        
                        print(f"\n SUCCESS: {marker}")
                        print(f"   Processing time: {marker_elapsed/60:.1f} minutes")
                        print(f"   Output file: {result_path}")
                        print(f"   File size: {size_mb:.1f} MB")
                        print(f"   Total samples: {n_samples:,}")
                        print(f"   Positive samples: {n_positive:,}")
                        print(f"   Negative samples: {n_negative:,}")
                        print(f"   Class ratio: {n_negative/n_positive:.1f}:1 (neg:pos)")
                        
                        logger.info(f"SUCCESS: {marker} - {n_samples:,} samples in {marker_elapsed:.1f}s")
                        
                    except Exception as e:
                        print(f"\n SUCCESS: {marker}")
                        print(f"   Processing time: {marker_elapsed/60:.1f} minutes")
                        print(f"   File size: {size_mb:.1f} MB")
                        print(f"   Could not validate file contents: {e}")
                        
                        logger.warning(f"Could not validate {marker}: {e}")
                else:
                    print(f"\n SUCCESS: {marker}")
                    print(f"   Processing time: {marker_elapsed/60:.1f} minutes")
                    print(f"    Output file not found at expected location")
                    
            else:
                failed.append(marker)
                print(f"\n FAILED: {marker}")
                print(f"   Processing time: {marker_elapsed/60:.1f} minutes")
                print(f"   Check logs for error details")
                
                logger.error(f"FAILED: {marker} after {marker_elapsed:.1f}s")
                
        except Exception as e:
            marker_elapsed = time.time() - marker_start_time
            failed.append(marker)
            print(f"\n CRASHED: {marker}")
            print(f"   Processing time: {marker_elapsed/60:.1f} minutes")
            print(f"   Error: {e}")
            
            logger.error(f"CRASHED: {marker}: {e}")
            traceback.print_exc()
        
        # Progress summary
        elapsed_total = time.time() - total_start_time
        remaining_markers = len(markers) - i
        
        if i > 0:
            avg_time_per_marker = elapsed_total / i
            estimated_remaining = avg_time_per_marker * remaining_markers
            
            print(f"\n PROGRESS SUMMARY:")
            print(f"   Completed: {i}/{len(markers)} markers")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            print(f"   Skipped: {len(skipped)}")
            print(f"   Elapsed time: {elapsed_total/3600:.1f} hours")
            print(f"   Estimated remaining: {estimated_remaining/3600:.1f} hours")
            print(f"   Estimated total: {(elapsed_total + estimated_remaining)/3600:.1f} hours")
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"FINAL PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time/3600:.2f} hours")
    print(f"Markers processed: {len(markers)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped (already existed): {len(skipped)}")
    
    # Successful datasets
    if successful:
        print(f"\n SUCCESSFUL DATASETS:")
        total_output_size = 0
        for marker, path in successful:
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                total_output_size += size_mb
                print(f"   {marker:8}: {size_mb:6.1f} MB - {path}")
            else:
                print(f"   {marker:8}: Unknown size - {path}")
        
        print(f"\n   Total output size: {total_output_size:.1f} MB ({total_output_size/1024:.2f} GB)")
    
    # Skipped datasets
    if skipped:
        print(f"\n SKIPPED DATASETS (already existed):")
        for marker, path in skipped:
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"   {marker:8}: {size_mb:6.1f} MB - {path}")
    
    # Failed datasets
    if failed:
        print(f"\n FAILED DATASETS:")
        for marker in failed:
            print(f"   {marker}")
        print(f"\n   Check the error logs above for details on failures.")
    
    # Overall result
    success_rate = len(successful) / len(markers) * 100
    
    if len(failed) == 0:
        print(f"\n ALL PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"   Success rate: {success_rate:.1f}%")
        logger.info(f"All processing completed successfully")
        return True
    else:
        print(f"\n PROCESSING COMPLETED WITH SOME FAILURES")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Failed markers: {failed}")
        logger.warning(f"Processing completed with {len(failed)} failures")
        return False

# Run the main function
if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        if success:
            print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print(f"\nPIPELINE COMPLETED WITH SOME ISSUES")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n PIPELINE CRASHED WITH FATAL ERROR:")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

EOF

# Job completion message
echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"

# Show final output directory contents
echo "Final output directory contents:"
ls -la data/

echo "Log files created:"
ls -la logs/deephistone_E003_final_${SLURM_JOB_ID}.*

echo "============================================================"
echo "DeepHistone processing job finished"
echo "============================================================"
