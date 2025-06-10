#!/usr/bin/env python3

import sys
import os
import time

print("=== DEEPHISTONE QUICK TEST ===")

# Change to project directory
os.chdir('/home/ekourb/deephistone')
print(f"Working in: {os.getcwd()}")

# Add scripts to path if needed
if os.path.exists('scripts'):
    sys.path.insert(0, 'scripts')

# Import pipeline
try:
    from deephistone_pipeline_all import *
    print("✓ Pipeline imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Configure for testing
config = DeepHistoneConfig()
config.TEST_MODE = True
config.TEST_CHROMOSOME = "chr22"
config.N_PROCESSES = 2

# Update global config
import deephistone_pipeline_all
deephistone_pipeline_all.config = config

print("✓ Configuration set for chr22 test")

# Test file validation
files_valid, missing = validate_epigenome_files("E003")
if not files_valid:
    print(f"✗ Missing files: {missing}")
    sys.exit(1)
print("✓ All files found")

# Run quick test
print("\nRunning E003-H3K4me3 test on chr22...")
start_time = time.time()

logger = setup_logging()
result_path, success = run_single_combination("E003", "H3K4me3", logger)

elapsed = time.time() - start_time

if success:
    print(f" TEST PASSED in {elapsed:.1f} seconds!")
    if result_path and os.path.exists(result_path):
        size_mb = os.path.getsize(result_path) / (1024*1024)
        print(f"Output: {size_mb:.1f} MB")
    print(" Pipeline is ready for full submission!")
else:
    print(f"TEST FAILED after {elapsed:.1f} seconds")
    sys.exit(1)
