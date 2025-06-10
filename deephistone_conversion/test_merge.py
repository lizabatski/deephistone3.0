#!/usr/bin/env python3
"""
Quick test for merge script before submitting SLURM job
"""

import numpy as np
import os
import glob

def test_merge_setup():
    """Test that merge will work before submitting job"""
    
    print("MERGE SETUP TEST")
    print("="*50)
    
    # Test 1: Check input directory
    input_dir = "../data/converted"
    print(f"1. Checking input directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    else:
        print(f"✅ Input directory exists")
    
    # Test 2: Find converted files
    pattern = os.path.join(input_dir, "*expected_format.npz")
    files = glob.glob(pattern)
    
    print(f"\n2. Looking for files: {pattern}")
    print(f"Found {len(files)} files:")
    
    if not files:
        print(f"❌ No converted files found!")
        return False
    
    total_size_gb = 0
    markers = []
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        size_gb = os.path.getsize(filepath) / (1024**3)
        total_size_gb += size_gb
        
        # Extract marker
        try:
            parts = filename.replace('_expected_format.npz', '').split('_')
            if len(parts) >= 2:
                marker = parts[1]
                markers.append(marker)
                print(f"  ✅ {marker:8}: {filename} ({size_gb:.2f} GB)")
            else:
                print(f"  ⚠️  Could not parse: {filename}")
        except:
            print(f"  ❌ Error parsing: {filename}")
    
    print(f"\nTotal input size: {total_size_gb:.2f} GB")
    
    # Test 3: Check for all 7 markers
    expected_markers = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']
    found_markers = set(markers)
    missing_markers = set(expected_markers) - found_markers
    
    print(f"\n3. Checking for all 7 markers:")
    print(f"Expected: {expected_markers}")
    print(f"Found: {sorted(list(found_markers))}")
    
    if missing_markers:
        print(f"⚠️  Missing markers: {missing_markers}")
    else:
        print(f"✅ All 7 markers found!")
    
    # Test 4: Quick file format check
    print(f"\n4. Testing file format (first file):")
    test_file = files[0]
    
    try:
        data = np.load(test_file)
        print(f"Testing: {os.path.basename(test_file)}")
        
        expected_keys = ['keys', 'dna', 'dnase', 'label']
        for key in expected_keys:
            if key in data.files:
                shape = data[key].shape
                print(f"  ✅ {key:6}: {shape}")
            else:
                print(f"  ❌ Missing key: {key}")
                return False
        
        n_samples = len(data['keys'])
        print(f"  Samples: {n_samples:,}")
        
    except Exception as e:
        print(f"❌ Error reading test file: {e}")
        return False
    
    # Test 5: Estimate memory requirements
    print(f"\n5. Memory requirement estimate:")
    
    # Rough estimate: each dataset needs to be loaded in memory
    estimated_memory_gb = total_size_gb * 2  # Loading + processing overhead
    print(f"Estimated memory needed: {estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > 100:
        print(f"⚠️  High memory usage - make sure SLURM job has enough RAM")
    else:
        print(f"✅ Memory usage looks reasonable")
    
    # Test 6: Check output directory
    output_dir = "data/final"
    print(f"\n6. Checking output directory: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output directory ready")
    except Exception as e:
        print(f"❌ Cannot create output directory: {e}")
        return False
    
    # Test 7: Check disk space
    print(f"\n7. Checking disk space:")
    import shutil
    
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    estimated_output_gb = total_size_gb * 1.5  # Rough estimate for merged file
    
    print(f"Available space: {free_space_gb:.1f} GB")
    print(f"Estimated output: {estimated_output_gb:.1f} GB")
    
    if free_space_gb < estimated_output_gb * 2:  # 2x safety margin
        print(f"⚠️  Low disk space - might need more room")
    else:
        print(f"✅ Sufficient disk space")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Files found: {len(files)}")
    print(f"Total input size: {total_size_gb:.2f} GB")
    print(f"Missing markers: {len(missing_markers)}")
    print(f"Estimated memory: {estimated_memory_gb:.1f} GB")
    print(f"Available disk: {free_space_gb:.1f} GB")
    
    if len(files) >= 6 and estimated_memory_gb < 150:  # Reasonable thresholds
        print(f"\n🎉 READY TO SUBMIT MERGE JOB!")
        return True
    else:
        print(f"\n⚠️  ISSUES FOUND - fix before submitting")
        return False

if __name__ == "__main__":
    success = test_merge_setup()
    
    if success:
        print(f"\nNext steps:")
        print(f"1. sbatch merge_datasets_job.sh")
        print(f"2. Monitor: squeue -u $USER")
        print(f"3. Watch: tail -f logs/merge_*.out")
    else:
        print(f"\nFix the issues above before submitting the job!")
