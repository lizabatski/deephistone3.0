#!/usr/bin/env python3
"""
Comprehensive DNase Alignment Debugging Script
"""

import os
import numpy as np
from collections import defaultdict
from deephistone_pipeline_all import config

def debug_dnase_file_format(epigenome_id="E005"):
    """
    Debug DNase file format and content
    """
    print(f"{'='*60}")
    print(f"DEBUGGING DNASE FILE FORMAT")
    print(f"{'='*60}")
    
    dnase_file = config.get_dnase_path(epigenome_id)
    print(f"DNase file: {dnase_file}")
    
    if not os.path.exists(dnase_file):
        print(f"ERROR: File not found!")
        return
    
    # 1. Check file format
    print(f"\n1. FILE FORMAT ANALYSIS:")
    with open(dnase_file, 'r') as f:
        lines = [f.readline().strip() for _ in range(10)]
    
    for i, line in enumerate(lines):
        if line:
            cols = line.split('\t')
            print(f"   Line {i+1}: {len(cols)} columns - {line[:100]}{'...' if len(line) > 100 else ''}")
    
    # 2. Check chromosomes
    print(f"\n2. CHROMOSOME ANALYSIS:")
    chroms = defaultdict(int)
    total_peaks = 0
    score_column_6 = []
    score_column_5 = []
    
    with open(dnase_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num > 100000:  # Limit for speed
                break
                
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7:
                continue
                
            try:
                chrom = cols[0]
                chroms[chrom] += 1
                total_peaks += 1
                
                # Collect scores from different columns
                try:
                    score_column_5.append(float(cols[4]))  # Column 5 (index 4)
                except:
                    pass
                try:
                    score_column_6.append(float(cols[5]))  # Column 6 (index 5)
                except:
                    pass
                    
            except (ValueError, IndexError):
                continue
    
    print(f"   Total peaks analyzed: {total_peaks}")
    print(f"   Chromosomes found: {len(chroms)}")
    for chrom, count in sorted(chroms.items()):
        print(f"     {chrom}: {count} peaks")
    
    # 3. Score analysis
    print(f"\n3. SCORE ANALYSIS:")
    if score_column_5:
        print(f"   Column 5 scores:")
        print(f"     Count: {len(score_column_5)}")
        print(f"     Range: {min(score_column_5):.2f} - {max(score_column_5):.2f}")
        print(f"     Mean: {np.mean(score_column_5):.2f}")
        print(f"     Percentiles: {np.percentile(score_column_5, [25, 50, 75, 90, 95])}")
    
    if score_column_6:
        print(f"   Column 6 scores:")
        print(f"     Count: {len(score_column_6)}")
        print(f"     Range: {min(score_column_6):.2f} - {max(score_column_6):.2f}")
        print(f"     Mean: {np.mean(score_column_6):.2f}")
        print(f"     Percentiles: {np.percentile(score_column_6, [25, 50, 75, 90, 95])}")

def debug_coordinate_overlap(epigenome_id="E005"):
    """
    Debug coordinate overlap between histone and DNase regions
    """
    print(f"\n{'='*60}")
    print(f"DEBUGGING COORDINATE OVERLAP")
    print(f"{'='*60}")
    
    # 1. Load some histone regions
    print(f"1. Loading histone regions...")
    try:
        from deephistone_pipeline_all import load_all_peaks_at_once, scan_genome_for_modification_sites
        
        all_peaks = load_all_peaks_at_once(epigenome_id)
        h3k4me1_sites = scan_genome_for_modification_sites(
            epigenome_id, "H3K4me1", all_peaks=all_peaks, apply_threshold=False
        )
        
        # Take first 20 sites for testing
        test_sites = h3k4me1_sites[:20]
        print(f"   Testing with {len(test_sites)} H3K4me1 sites")
        
        # Expand to 1000bp
        from deephistone_pipeline_all import expand_regions_to_1000bp
        expanded_sites = expand_regions_to_1000bp(test_sites)
        print(f"   Expanded to {len(expanded_sites)} 1000bp regions")
        
    except Exception as e:
        print(f"   ERROR loading histone regions: {e}")
        return
    
    # 2. Load DNase peaks
    print(f"\n2. Loading DNase peaks...")
    dnase_file = config.get_dnase_path(epigenome_id)
    dnase_peaks_by_chrom = defaultdict(list)
    
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
                
                if chrom in ['chrX', 'chrY']:
                    continue
                
                try:
                    fold_enrichment = float(cols[5])  # Try column 6 (index 5)
                except:
                    fold_enrichment = 1.0
                
                fold_enrichment = max(0.0, fold_enrichment)
                dnase_peaks_by_chrom[chrom].append((start, end, fold_enrichment))
                
            except (ValueError, IndexError):
                continue
    
    print(f"   Loaded DNase peaks for {len(dnase_peaks_by_chrom)} chromosomes")
    for chrom, peaks in list(dnase_peaks_by_chrom.items())[:5]:
        print(f"     {chrom}: {len(peaks)} peaks")
    
    # 3. Test overlap
    print(f"\n3. TESTING OVERLAP:")
    overlaps_found = 0
    total_signal = 0
    
    for i, (chrom, region_start, region_end) in enumerate(expanded_sites[:10]):
        print(f"   Region {i+1}: {chrom}:{region_start}-{region_end}")
        
        region_signal = 0
        region_overlaps = 0
        
        if chrom in dnase_peaks_by_chrom:
            for peak_start, peak_end, fold_enrichment in dnase_peaks_by_chrom[chrom]:
                # Check overlap
                if peak_end <= region_start or peak_start >= region_end:
                    continue
                
                overlap_start = max(region_start, peak_start)
                overlap_end = min(region_end, peak_end)
                overlap_length = overlap_end - overlap_start
                
                if overlap_length > 0:
                    region_overlaps += 1
                    region_signal += fold_enrichment * overlap_length
                    
                    print(f"     Overlap {region_overlaps}: DNase peak {peak_start}-{peak_end}, score={fold_enrichment:.2f}, overlap={overlap_length}bp")
                    
                    if region_overlaps >= 3:  # Limit output
                        break
        
        if region_overlaps > 0:
            overlaps_found += 1
            total_signal += region_signal
            print(f"     Total overlaps: {region_overlaps}, signal: {region_signal:.2f}")
        else:
            print(f"     NO OVERLAPS FOUND")
    
    print(f"\nOVERALL RESULTS:")
    print(f"   Regions with overlaps: {overlaps_found}/{len(expanded_sites[:10])}")
    print(f"   Average signal per region: {total_signal/len(expanded_sites[:10]):.2f}")
    
    if overlaps_found == 0:
        print(f"   CRITICAL: No overlaps found - major alignment issue!")
    elif overlaps_found < 5:
        print(f"   WARNING: Very few overlaps - alignment issue likely")
    else:
        print(f"   GOOD: Reasonable overlap found")

def debug_coordinate_systems():
    """
    Debug potential coordinate system mismatches
    """
    print(f"\n{'='*60}")
    print(f"DEBUGGING COORDINATE SYSTEMS")
    print(f"{'='*60}")
    
    # Check if sequences and DNase use same coordinates
    print(f"1. Checking reference genome:")
    print(f"   Sequence FASTA: {config.FASTA_PATH}")
    print(f"   Chromosome sizes: {config.CHROM_SIZES}")
    
    # Check chromosome sizes
    try:
        chrom_sizes = config.get_chrom_sizes()
        print(f"   Loaded {len(chrom_sizes)} chromosomes")
        print(f"   Sample chromosome sizes:")
        for chrom, size in list(chrom_sizes.items())[:5]:
            print(f"     {chrom}: {size:,} bp")
    except Exception as e:
        print(f"   ERROR loading chromosome sizes: {e}")
    
    # Check FASTA file
    try:
        from pyfaidx import Fasta
        genome = Fasta(config.FASTA_PATH)
        print(f"   FASTA chromosomes: {len(genome.keys())}")
        print(f"   Sample FASTA chromosomes: {list(genome.keys())[:10]}")
    except Exception as e:
        print(f"   ERROR loading FASTA: {e}")

def test_different_score_columns(epigenome_id="E005"):
    """
    Test using different score columns for DNase
    """
    print(f"\n{'='*60}")
    print(f"TESTING DIFFERENT SCORE COLUMNS")
    print(f"{'='*60}")
    
    # Get a test region
    test_region = ('chr1', 1000000, 1001000)
    print(f"Testing region: {test_region[0]}:{test_region[1]}-{test_region[2]}")
    
    dnase_file = config.get_dnase_path(epigenome_id)
    
    # Test different columns
    for col_idx, col_name in [(4, "Column 5"), (5, "Column 6"), (6, "Column 7")]:
        print(f"\n{col_name} results:")
        
        signal_total = 0
        peak_count = 0
        
        with open(dnase_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                cols = line.split('\t')
                if len(cols) <= col_idx:
                    continue
                    
                try:
                    chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                    
                    if chrom != test_region[0]:
                        continue
                    
                    # Check overlap with test region
                    if end <= test_region[1] or start >= test_region[2]:
                        continue
                    
                    try:
                        score = float(cols[col_idx])
                        signal_total += score
                        peak_count += 1
                        
                        if peak_count <= 5:  # Show first few
                            print(f"   Peak {peak_count}: {start}-{end}, score={score:.2f}")
                            
                    except ValueError:
                        continue
                        
                except (ValueError, IndexError):
                    continue
        
        print(f"   Total peaks overlapping: {peak_count}")
        print(f"   Total signal: {signal_total:.2f}")
        if peak_count > 0:
            print(f"   Average signal: {signal_total/peak_count:.2f}")

if __name__ == "__main__":
    print("DNase Alignment Debugging")
    print("=" * 40)
    
    epigenome_id = "E005"
    
    # Run all debugging functions
    debug_dnase_file_format(epigenome_id)
    debug_coordinate_systems()
    debug_coordinate_overlap(epigenome_id)
    test_different_score_columns(epigenome_id)
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING COMPLETE")
    print(f"{'='*60}")
    print(f"Next steps based on results:")
    print(f"1. If no overlaps found: Check coordinate system mismatch")
    print(f"2. If few overlaps found: Try different score columns")
    print(f"3. If overlaps found but low signal: Check score normalization")
    print(f"4. If chromosomes missing: Check file format/filtering")