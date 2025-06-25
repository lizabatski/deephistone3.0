#!/usr/bin/env python3
"""
FIXED: Comprehensive Debug Script that actually uses the target epigenome
"""

import sys
import os
from collections import defaultdict

# Import your main pipeline
from deephistone_pipeline_all import *  

def debug_specific_epigenome(target_epigenome):
    """Debug a specific epigenome with proper parameter passing"""
    
    print(f"{'='*80}")
    print(f"DEBUGGING {target_epigenome} - FIXED VERSION")
    print(f"{'='*80}")
    
    # Paper counts for the target epigenome
    paper_counts = {
        'E003': {'H3K4me1': 342503, 'H3K4me3': 213546, 'H3K27me3': 231823, 'H3K36me3': 424644, 'H3K9me3': 322656, 'H3K9ac': 155558, 'H3K27ac': 236273},
        'E005': {'H3K4me1': 433887, 'H3K4me3': 166189, 'H3K27me3': 107589, 'H3K36me3': 456459, 'H3K9me3': 287790, 'H3K9ac': 71753, 'H3K27ac': 258471},
        'E006': {'H3K4me1': 462960, 'H3K4me3': 163290, 'H3K27me3': 130314, 'H3K36me3': 487545, 'H3K9me3': 278273, 'H3K9ac': 168264, 'H3K27ac': 289090},
        'E007': {'H3K4me1': 263177, 'H3K4me3': 159076, 'H3K27me3': 69494, 'H3K36me3': 192670, 'H3K9me3': 149941, 'H3K9ac': 294433, 'H3K27ac': 118545}
    }
    
    if target_epigenome not in paper_counts:
        print(f"❌ {target_epigenome} not in paper data")
        return
    
    target_paper_counts = paper_counts[target_epigenome]
    
    print(f"✅ Processing {target_epigenome} (confirmed)")
    print(f"\n{'Marker':>8} {'Your Pos':>10} {'Your Neg':>10} {'Paper Pos':>10} {'Pos%':>6} {'Diff%':>8}")
    print("-" * 70)
    
    for marker in config.ALL_MARKERS:
        try:
            print(f"🔄 Loading {target_epigenome}-{marker}...")
            
            # IMPORTANT: Make sure we're passing the correct epigenome
            target_sites, negative_sites, all_marker_sites = load_all_histone_markers_for_epigenome(
                target_epigenome, marker  # This should be E005, not E003!
            )
            
            if target_sites is None:
                print(f"{marker:>8}: ❌ FAILED")
                continue
            
            your_pos = len(target_sites)
            your_neg = len(negative_sites)
            paper_pos = target_paper_counts.get(marker, 0)
            
            total = your_pos + your_neg
            pos_pct = (your_pos / total * 100) if total > 0 else 0
            
            diff_pct = ((your_pos - paper_pos) / paper_pos * 100) if paper_pos > 0 else 0
            
            status = "✅" if abs(diff_pct) < 20 else "⚠️" if abs(diff_pct) < 50 else "❌"
            
            print(f"{marker:>8}: {your_pos:>10,} {your_neg:>10,} {paper_pos:>10,} {pos_pct:>5.1f}% {diff_pct:>+6.1f}% {status}")
            
            # Quick chromosome check for this marker
            target_chroms = set()
            for chrom, start, end in target_sites[:100]:  # Check first 100 sites
                target_chroms.add(chrom)
            
            main_autosomes = {f"chr{i}" for i in range(1, 23)}
            non_main = target_chroms - main_autosomes
            
            if non_main:
                print(f"    ⚠️  Non-main chromosomes found: {list(non_main)[:5]}")
            
        except Exception as e:
            print(f"{marker:>8}: ❌ ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print("-" * 70)
    print(f"✅ Analysis complete for {target_epigenome}")

def check_chromosome_contamination():
    """Quick check of chromosome contamination"""
    print(f"\n{'='*60}")
    print(f"CHROMOSOME CONTAMINATION CHECK")
    print(f"{'='*60}")
    
    chrom_sizes = config.get_chrom_sizes()
    
    main_autosomes = {f"chr{i}" for i in range(1, 23)}
    other_chroms = [c for c in chrom_sizes.keys() if c not in main_autosomes]
    
    print(f"📊 Total chromosomes: {len(chrom_sizes)}")
    print(f"📍 Main autosomes: {len([c for c in chrom_sizes.keys() if c in main_autosomes])}")
    print(f"❓ Other chromosomes: {len(other_chroms)}")
    
    if other_chroms:
        print(f"⚠️  Contaminating chromosomes: {other_chroms[:10]}{'...' if len(other_chroms) > 10 else ''}")
        print(f"🚨 RECOMMENDATION: Filter these out for cleaner training!")
    else:
        print(f"✅ Clean autosomal data")

if __name__ == "__main__":
    # Get target epigenome
    if len(sys.argv) > 1:
        target_epigenome = sys.argv[1]
    else:
        target_epigenome = "E005"
    
    print(f"🎯 Target epigenome: {target_epigenome}")
    
    
    check_chromosome_contamination()
    debug_specific_epigenome(target_epigenome)