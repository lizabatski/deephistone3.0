#!/usr/bin/env python3
"""Simple environment test for DeepHistone"""

import sys
import os

print("=== SIMPLE DEEPHISTONE TEST ===")

# Test basic imports
print("\n1. Testing basic imports...")
try:
    import numpy as np
    import pandas as pd
    print("✅ numpy, pandas")
except ImportError as e:
    print(f"❌ Scientific packages: {e}")
    print("Run: module load scipy-stack")
    sys.exit(1)

try:
    from tqdm import tqdm
    from pyfaidx import Fasta
    print("✅ tqdm, pyfaidx")
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("Run: pip install --user tqdm pyfaidx")
    sys.exit(1)

# Test pipeline import
print("\n2. Testing pipeline import...")
os.chdir('/home/ekourb/deephistone')
if os.path.exists('scripts'):
    sys.path.insert(0, 'scripts')

try:
    from deephistone_pipeline_all import DeepHistoneConfig
    print("✅ Pipeline imported")
except ImportError as e:
    print(f"❌ Pipeline import failed: {e}")
    sys.exit(1)

# Test configuration
print("\n3. Testing configuration...")
try:
    config = DeepHistoneConfig()
    print(f"✅ Config created")
    print(f"   Markers: {len(config.ALL_MARKERS)}")
    print(f"   Epigenomes: {len(config.VALID_EPIGENOMES)}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test files
print("\n4. Testing key files...")
key_files = [
    "raw/hg19.fa",
    "raw/E003-H3K4me3.narrowPeak",
    "deephistone_pipeline_all.py"
]

missing = []
for f in key_files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ {f}")
        missing.append(f)

if missing:
    print(f"\n❌ Missing {len(missing)} files")
    sys.exit(1)

print("\n🎉 ALL TESTS PASSED!")
print("Environment is ready for DeepHistone pipeline")
