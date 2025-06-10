#!/bin/bash
# test_local.sh - Local test version (no SLURM, runs immediately)

echo "============================================================"
echo "DEEPHISTONE LOCAL TEST - NO SLURM SUBMISSION"
echo "============================================================"
echo "This tests the pipeline locally before submitting to cluster"
echo "Testing started at: $(date)"

# Load modules (same as cluster)
echo "Loading modules..."
module load python/3.9
module load scipy-stack

# Set environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2  # Use fewer cores for local test

# Change to project directory
cd /home/ekourb/deephistone
echo "Working in: $(pwd)"

# Check key files (same checks as cluster script)
echo ""
echo "Verifying required files..."

if [ -f "deephistone_pipeline_all.py" ]; then
    echo "✓ Pipeline file: deephistone_pipeline_all.py"
else
    echo "✗ ERROR: deephistone_pipeline_all.py not found!"
    exit 1
fi

if [ -f "raw/hg19.fa" ]; then
    genome_size=$(du -sh raw/hg19.fa | cut -f1)
    echo "✓ Genome file: raw/hg19.fa ($genome_size)"
else
    echo "✗ ERROR: raw/hg19.fa not found!"
    exit 1
fi

if [ -f "raw/E003-H3K4me3.narrowPeak" ]; then
    chip_size=$(du -sh raw/E003-H3K4me3.narrowPeak | cut -f1)
    echo "✓ Sample ChIP file: raw/E003-H3K4me3.narrowPeak ($chip_size)"
else
    echo "✗ ERROR: raw/E003-H3K4me3.narrowPeak not found!"
    exit 1
fi

# Create directories
mkdir -p data logs
echo "✓ Created directories"

echo ""
echo "============================================================"
echo "TESTING PIPELINE IMPORT AND BASIC FUNCTIONALITY"
echo "============================================================"

# Test the Python pipeline (abbreviated version)
python3 -u << 'EOF'
import sys
import os
import time

print("Testing pipeline setup...")

# Setup
os.chdir('/home/ekourb/deephistone')
if os.path.exists('scripts'):
    sys.path.insert(0, 'scripts')

# Test imports
try:
    print("Testing imports...")
    from deephistone_pipeline_all import (
        DeepHistoneConfig, 
        setup_logging, 
        run_single_combination,
        validate_epigenome_files
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test configuration
try:
    print("Testing configuration...")
    config = DeepHistoneConfig()
    config.TEST_MODE = True  # Use chr22 for local test
    config.TEST_CHROMOSOME = "chr22"
    config.N_PROCESSES = 2  # Use fewer cores
    config.SKIP_EXISTING = True
    
    # Update global config
    import deephistone_pipeline_all
    deephistone_pipeline_all.config = config
    
    print("✓ Configuration successful")
    print(f"  TEST_MODE: {config.TEST_MODE}")
    print(f"  N_PROCESSES: {config.N_PROCESSES}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test file validation
try:
    print("Testing file validation...")
    files_valid, missing_files = validate_epigenome_files("E003")
    if files_valid:
        print("✓ All files validated")
    else:
        print(f"✗ Missing files: {missing_files}")
        sys.exit(1)
except Exception as e:
    print(f"✗ File validation failed: {e}")
    sys.exit(1)

# Test ONE marker processing (chr22 only - should be fast)
print("\n" + "="*50)
print("RUNNING QUICK PIPELINE TEST (E003-H3K4me3 on chr22)")
print("="*50)
print("This should take 1-2 minutes...")

try:
    start_time = time.time()
    logger = setup_logging()
    
    result_path, success = run_single_combination("E003", "H3K4me3", logger)
    
    elapsed = time.time() - start_time
    
    if success and result_path:
        print(f"\n🎉 LOCAL TEST SUCCESSFUL!")
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Output: {result_path}")
        
        if os.path.exists(result_path):
            size_mb = os.path.getsize(result_path) / (1024*1024)
            print(f"File size: {size_mb:.1f} MB")
            
            # Quick validation
            try:
                import numpy as np
                data = np.load(result_path, allow_pickle=True)
                n_samples = len(data['sequences'])
                n_pos = int(data['labels'].sum())
                n_neg = len(data['labels']) - n_pos
                
                print(f"Samples: {n_samples:,} ({n_pos:,} pos, {n_neg:,} neg)")
                print(f"Ratio: {n_neg/n_pos:.1f}:1")
                
            except Exception as e:
                print(f"Warning: Could not validate contents: {e}")
        
        print(f"\n✅ PIPELINE IS READY FOR CLUSTER SUBMISSION!")
        print(f"Estimated full genome time: ~{elapsed * 50 / 60:.0f} minutes per marker")
        
    else:
        print(f"\n❌ LOCAL TEST FAILED after {elapsed:.1f} seconds")
        print("Check the error messages above")
        sys.exit(1)
        
except Exception as e:
    print(f"\n💥 LOCAL TEST CRASHED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("LOCAL TEST COMPLETE")
print("="*50)

EOF

echo ""
echo "============================================================"
echo "LOCAL TEST COMPLETED AT: $(date)"
echo "============================================================"

echo ""
echo "If the test above passed, you can now submit to the cluster:"
echo "  sbatch submit_final.sh"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/deephistone_E003_final_*.out"
