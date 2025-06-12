#!/usr/bin/env python3
"""
Wrapper script for histone modification site counting analysis
Usage: python run_histone_analysis.py --epigenome E005 --data_dir /path/to/data --output_dir /path/to/results
"""

import argparse
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

from count_npz_sites import count_histone_sites, count_multiple_epigenomes, inspect_npz_file, EPIGENOME_CONFIGS

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def save_results_text(results, epigenome_id, output_file, data_dir):
    """Save results in detailed text format"""
    config = EPIGENOME_CONFIGS[epigenome_id]
    
    with open(output_file, 'w') as f:
        # Header
        f.write(f"Histone Modification Site Counts - {config['name']} ({epigenome_id})\n")
        f.write("=" * 70 + "\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}\n")
        f.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}\n\n")
        
        # Individual marker counts
        total_sites = 0
        f.write("Individual Marker Counts:\n")
        f.write("-" * 30 + "\n")
        for marker in config['histone_markers']:
            count = results.get(marker, 0)
            f.write(f"{marker:>10}: {count:>12,} sites\n")
            total_sites += count
        
        f.write("-" * 30 + "\n")
        f.write(f"{'TOTAL':>10}: {total_sites:>12,} sites\n")
        
        # Comparison with paper if available
        if 'paper_counts' in config:
            paper_counts = config['paper_counts']
            paper_total = sum(paper_counts.values())
            
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"COMPARISON WITH REFERENCE ({epigenome_id}):\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Marker':>10} {'Your Count':>12} {'Ref Count':>12} {'Difference':>12} {'% Diff':>8}\n")
            f.write("-" * 70 + "\n")
            
            for marker in config['histone_markers']:
                your_count = results.get(marker, 0)
                ref_count = paper_counts.get(marker, 0)
                if ref_count > 0:
                    diff = your_count - ref_count
                    diff_pct = (diff / ref_count * 100)
                    f.write(f"{marker:>10} {your_count:>12,} {ref_count:>12,} {diff:>+12,} {diff_pct:>+7.1f}%\n")
                else:
                    f.write(f"{marker:>10} {your_count:>12,} {'N/A':>12} {'N/A':>12} {'N/A':>8}\n")
            
            f.write("-" * 70 + "\n")
            total_diff = total_sites - paper_total
            total_diff_pct = (total_diff / paper_total * 100) if paper_total > 0 else 0
            f.write(f"{'TOTAL':>10} {total_sites:>12,} {paper_total:>12,} {total_diff:>+12,} {total_diff_pct:>+7.1f}%\n")

def save_results_csv(results, epigenome_id, output_file):
    """Save results in CSV format for easy analysis"""
    config = EPIGENOME_CONFIGS[epigenome_id]
    
    with open(output_file, 'w') as f:
        f.write("Epigenome,Marker,Count,Reference_Count,Difference,Percent_Difference\n")
        
        for marker in config['histone_markers']:
            your_count = results.get(marker, 0)
            ref_count = config.get('paper_counts', {}).get(marker, 0)
            diff = your_count - ref_count if ref_count > 0 else 0
            diff_pct = (diff / ref_count * 100) if ref_count > 0 else 0
            
            f.write(f"{epigenome_id},{marker},{your_count},{ref_count},{diff},{diff_pct:.2f}\n")

def analyze_single_epigenome(epigenome_id, data_dir, output_dir, verbose=False):
    """Analyze a single epigenome"""
    print(f"Starting analysis for epigenome: {epigenome_id}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Validate inputs
    if epigenome_id not in EPIGENOME_CONFIGS:
        raise ValueError(f"Unknown epigenome '{epigenome_id}'. Available: {list(EPIGENOME_CONFIGS.keys())}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir)
    
    # Run the analysis
    print("Counting histone modification sites...")
    results = count_histone_sites(data_dir, epigenome_id)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Text format
    txt_file = output_path / f"histone_counts_{epigenome_id}_{timestamp}.txt"
    save_results_text(results, epigenome_id, txt_file, data_dir)
    print(f"Detailed results saved to: {txt_file}")
    
    # CSV format
    csv_file = output_path / f"histone_counts_{epigenome_id}_{timestamp}.csv"
    save_results_csv(results, epigenome_id, csv_file)
    print(f"CSV results saved to: {csv_file}")
    
    # Print summary
    config = EPIGENOME_CONFIGS[epigenome_id]
    total_sites = sum(results.values())
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY:")
    print("=" * 60)
    print(f"Epigenome: {epigenome_id} ({config['name']})")
    print(f"Anatomy: {config.get('anatomy', 'N/A')}")
    print(f"Total histone modification sites: {total_sites:,}")
    print(f"Markers analyzed: {len(config['histone_markers'])}")
    
    return results

def analyze_multiple_epigenomes(epigenome_ids, data_dir, output_dir, verbose=False):
    """Analyze multiple epigenomes"""
    print(f"Starting analysis for {len(epigenome_ids)} epigenomes: {epigenome_ids}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Setup output directory
    output_path = setup_output_directory(output_dir)
    
    # Run the analysis
    all_results = count_multiple_epigenomes(data_dir, epigenome_ids)
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save individual results for each epigenome
    for epigenome_id, results in all_results.items():
        if results:  # Only save if we got results
            txt_file = output_path / f"histone_counts_{epigenome_id}_{timestamp}.txt"
            save_results_text(results, epigenome_id, txt_file, data_dir)
            
            csv_file = output_path / f"histone_counts_{epigenome_id}_{timestamp}.csv"
            save_results_csv(results, epigenome_id, csv_file)
    
    # Save combined CSV
    combined_csv = output_path / f"histone_counts_combined_{timestamp}.csv"
    with open(combined_csv, 'w') as f:
        f.write("Epigenome,Marker,Count,Reference_Count,Difference,Percent_Difference\n")
        
        for epigenome_id, results in all_results.items():
            config = EPIGENOME_CONFIGS.get(epigenome_id, {})
            markers = config.get('histone_markers', [])
            
            for marker in markers:
                your_count = results.get(marker, 0)
                ref_count = config.get('paper_counts', {}).get(marker, 0)
                diff = your_count - ref_count if ref_count > 0 else 0
                diff_pct = (diff / ref_count * 100) if ref_count > 0 else 0
                
                f.write(f"{epigenome_id},{marker},{your_count},{ref_count},{diff},{diff_pct:.2f}\n")
    
    print(f"Combined results saved to: {combined_csv}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze histone modification sites')
    parser.add_argument('--epigenome', '-e', type=str, 
                        help='Epigenome ID to analyze (e.g., E005)')
    parser.add_argument('--epigenomes', '-es', nargs='+', 
                        help='Multiple epigenome IDs to analyze')
    parser.add_argument('--all', action='store_true', 
                        help='Analyze all available epigenomes')
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--list', action='store_true',
                        help='List available epigenomes and exit')
    parser.add_argument('--inspect', type=str,
                        help='Inspect a specific NPZ file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # List available epigenomes
    if args.list:
        print("Available epigenomes:")
        print("-" * 40)
        for epigenome_id, config in EPIGENOME_CONFIGS.items():
            print(f"{epigenome_id:>6}: {config['name']} ({config.get('anatomy', 'N/A')})")
        return
    
    # Inspect specific file
    if args.inspect:
        inspect_npz_file(args.inspect)
        return
    
    # Validate arguments
    if not any([args.epigenome, args.epigenomes, args.all]):
        parser.error("Must specify either --epigenome, --epigenomes, or --all")
    
    try:
        if args.epigenome:
            # Single epigenome analysis
            results = analyze_single_epigenome(args.epigenome, args.data_dir, args.output_dir, args.verbose)
            
        elif args.epigenomes:
            # Multiple specific epigenomes
            results = analyze_multiple_epigenomes(args.epigenomes, args.data_dir, args.output_dir, args.verbose)
            
        elif args.all:
            # All available epigenomes
            all_epigenomes = list(EPIGENOME_CONFIGS.keys())
            results = analyze_multiple_epigenomes(all_epigenomes, args.data_dir, args.output_dir, args.verbose)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()