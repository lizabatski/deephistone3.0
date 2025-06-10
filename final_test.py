#!/usr/bin/env python3
"""Final pipeline test - one marker on chr22"""

import sys
import os
import time

print("=== FINAL PIPELINE TEST ===")
print("Testing E003-H3K4me3 on chr22 (should take 1-2 minutes)")

# Setup
os.chdir('/home/ekourb/deephistone')
if os.path.exists('scripts'):
    sys.path.insert(0, 'scripts')

# Import and configure
from deephistone_pipeline_all import *

config = DeepHistoneConfig()
config.TEST_MODE = True
config.TEST_CHROMOSOME = "chr22"
config.N_PROCESSES = 2

# Update global config
import deephistone_pipeline_all
deephistone_pipeline_all.config = config

print(f"✅ Configuration: {config.TEST_CHROMOSOME} mode")

# Run the test
print("\n🚀 Running pipeline test...")
start_time = time.time()

try:
    logger = setup_logging()
    result_path, success = run_single_combination("E003", "H3K4me3", logger)
    
    elapsed = time.time() - start_time
    
    if success and result_path:
        # Validate the output
        import numpy as np
        data = np.load(result_path, allow_pickle=True)
        
        n_samples = len(data['sequences'])
        n_pos = int(data['labels'].sum())
        n_neg = len(data['labels']) - n_pos
        file_size = os.path.getsize(result_path) / (1024*1024)
        
        print(f"\n🎉 TEST SUCCESSFUL!")
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Output: {result_path}")
        print(f"File size: {file_size:.1f} MB")
        print(f"Samples: {n_samples:,} ({n_pos:,} pos, {n_neg:,} neg)")
        print(f"Ratio: {n_neg/n_pos:.1f}:1 (neg:pos)")
        
        # Estimates for full genome
        full_genome_time = elapsed * 50  # chr22 is ~2% of genome
        full_genome_size = file_size * 50
        
        print(f"\n📊 FULL GENOME ESTIMATES:")
        print(f"Time per marker: ~{full_genome_time/60:.0f} minutes")
        print(f"Total time (7 markers): ~{full_genome_time*7/3600:.1f} hours")
        print(f"File size per marker: ~{full_genome_size:.0f} MB")
        print(f"Total output: ~{full_genome_size*7/1024:.1f} GB")
        
        print(f"\n✅ PIPELINE IS READY FOR COMPUTE CANADA!")
        
    else:
        print(f"\n❌ TEST FAILED after {elapsed:.1f} seconds")
        print("Check the error messages above")
        sys.exit(1)
        
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n💥 TEST CRASHED after {elapsed:.1f} seconds")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
