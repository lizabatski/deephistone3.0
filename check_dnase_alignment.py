#!/usr/bin/env python3
"""
Simple DNase Alignment Check for E005
"""

import os
import numpy as np
from collections import defaultdict
from deephistone_pipeline_all import config, extract_dnase_openness_scores

def quick_dnase_check(epigenome_id="E005", n_test_regions=1000):
    """
    Quick check if DNase data is properly aligned with histone modifications
    """
    print(f"{'='*60}")
    print(f"DNASE ALIGNMENT CHECK FOR {epigenome_id}")
    print(f"{'='*60}")
    
    # 1. Check DNase file exists
    dnase_file = config.get_dnase_path(epigenome_id)
    print(f"DNase file: {dnase_file}")
    
    if not os.path.exists(dnase_file):
        print(f"DNase file not found!")
        return "MISSING"
    
    # 2. Count DNase peaks
    dnase_peaks = 0
    peak_scores = []
    
    print(f"Counting DNase peaks...")
    with open(dnase_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7:
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                
                # Skip sex chromosomes 
                if chrom in ['chrX', 'chrY']:
                    continue
                
                # Get fold enrichment score
                try:
                    fold_enrichment = float(cols[6])
                except:
                    fold_enrichment = 1.0
                
                dnase_peaks += 1
                peak_scores.append(fold_enrichment)
                
            except (ValueError, IndexError):
                continue
    
    print(f"Found {dnase_peaks:,} DNase peaks")
    
    if dnase_peaks == 0:
        print(f"No valid DNase peaks found!")
        return "NO_PEAKS"
    
    # 3. Basic DNase statistics
    print(f"\nDNASE STATISTICS:")
    print(f"   Total peaks: {dnase_peaks:,}")
    print(f"   Score range: {min(peak_scores):.2f} - {max(peak_scores):.2f}")
    print(f"   Mean score: {np.mean(peak_scores):.2f}")
    print(f"   Median score: {np.median(peak_scores):.2f}")
    
    # 4. Test alignment with histone regions
    print(f"\nTESTING ALIGNMENT WITH HISTONE REGIONS:")
    
    # Use some known histone regions - you can get these from your current processing
    # For now, let's create some test regions around typical gene locations
    test_regions = [
        ('chr1', 1000000, 1001000),   # Random test regions
        ('chr1', 2000000, 2001000),
        ('chr2', 1000000, 1001000),
        ('chr2', 2000000, 2001000),
        ('chr3', 1000000, 1001000),
    ]
    
    print(f"   Testing {len(test_regions)} regions...")
    
    # Extract DNase scores for test regions
    try:
        openness_scores = extract_dnase_openness_scores(epigenome_id, test_regions)
        
        # Analyze coverage
        total_positions = len(test_regions) * 1000
        non_zero_positions = sum(np.count_nonzero(scores) for scores in openness_scores)
        coverage_pct = (non_zero_positions / total_positions) * 100
        
        print(f"   Coverage: {coverage_pct:.1f}% ({non_zero_positions:,}/{total_positions:,} positions)")
        
        # Analyze scores
        all_scores = []
        for scores in openness_scores:
            all_scores.extend(scores[scores > 0])
        
        if len(all_scores) > 0:
            print(f"   Non-zero scores: {len(all_scores):,}")
            print(f"   Score range: {min(all_scores):.2f} - {max(all_scores):.2f}")
            print(f"   Mean score: {np.mean(all_scores):.2f}")
        else:
            print(f"   No non-zero scores found in test regions!")
            return "NO_SIGNAL"
        
        # 5. Diagnosis
        print(f"\n DIAGNOSIS:")
        
        if coverage_pct < 5:
            print(f"POOR ALIGNMENT: Very low coverage ({coverage_pct:.1f}%)")
            print(f"   This will severely hurt model performance!")
            status = "POOR"
        elif coverage_pct < 15:
            print(f" MODERATE ALIGNMENT: Low coverage ({coverage_pct:.1f}%)")
            print(f"   This might explain suboptimal performance")
            status = "MODERATE"
        elif coverage_pct > 80:
            print(f" SUSPICIOUS: Very high coverage ({coverage_pct:.1f}%)")
            print(f"   This seems unusually high - check for issues")
            status = "SUSPICIOUS"
        else:
            print(f"GOOD ALIGNMENT: Reasonable coverage ({coverage_pct:.1f}%)")
            status = "GOOD"
        
        # Additional checks
        if max(peak_scores) < 2.0:
            print(f" WARNING: Low DNase enrichment scores (max={max(peak_scores):.2f})")
        
        if min(peak_scores) < 0:
            print(f" WARNING: Negative DNase scores found")
        
        return status
        
    except Exception as e:
        print(f"Error testing alignment: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"

def check_dnase_in_merged_dataset(data_file):
    """
    Check DNase data quality in your merged dataset
    """
    print(f"\n{'='*60}")
    print(f"CHECKING DNASE IN MERGED DATASET")
    print(f"{'='*60}")
    
    try:
        with np.load(data_file) as data:
            dnase_data = data['dnase'][:]
            
            print(f" DNase data shape: {dnase_data.shape}")
            print(f"   Min: {dnase_data.min():.3f}")
            print(f"   Max: {dnase_data.max():.3f}")
            print(f"   Mean: {dnase_data.mean():.3f}")
            print(f"   Std: {dnase_data.std():.3f}")
            
            zero_pct = (dnase_data == 0).mean() * 100
            print(f"   Zero values: {zero_pct:.1f}%")
            
            if zero_pct > 85:
                print(f" CRITICAL: {zero_pct:.1f}% of DNase values are zero!")
                print(f"   This indicates severe DNase alignment problems")
                return "CRITICAL"
            elif zero_pct > 70:
                print(f" WARNING: {zero_pct:.1f}% of DNase values are zero")
                print(f"   This might explain poor performance")
                return "WARNING"
            else:
                print(f" DNase data looks reasonable ({zero_pct:.1f}% zeros)")
                return "GOOD"
    
    except Exception as e:
        print(f"Error checking merged dataset: {e}")
        return "ERROR"

if __name__ == "__main__":
    print("DNase Alignment Diagnostic")
    print("=" * 30)
    
    # Check raw DNase alignment
    alignment_status = quick_dnase_check("E005")
    
    # Check merged dataset (update path as needed)
    merged_file = "data/final/E005_all_markers_merged.npz"
    if os.path.exists(merged_file):
        dataset_status = check_dnase_in_merged_dataset(merged_file)
    else:
        print(f"\n  Merged dataset not found: {merged_file}")
        dataset_status = "NOT_FOUND"
    
    print(f"\n{'='*60}")
    print(f"FINAL ASSESSMENT")
    print(f"{'='*60}")
    print(f"Raw DNase alignment: {alignment_status}")
    print(f"Merged dataset DNase: {dataset_status}")
    
    if alignment_status in ["POOR", "NO_SIGNAL"] or dataset_status in ["CRITICAL", "WARNING"]:
        print(f"\n CONCLUSION: DNase alignment issues likely causing poor performance!")
        print(f"   Solutions:")
        print(f"   1. Check DNase file format and coordinates")
        print(f"   2. Verify DNase peak calling parameters")
        print(f"   3. Consider using different DNase processing")
        print(f"   4. Train DNA-only model to compare")
    elif alignment_status == "GOOD" and dataset_status == "GOOD":
        print(f"\ CONCLUSION: DNase alignment looks good")
        print(f"   Problem likely in model architecture or training")
    else:
        print(f"\n CONCLUSION: Mixed results - investigate further")