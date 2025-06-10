                            import numpy as np
import os
from pathlib import Path

def count_histone_sites(npz_directory):
    """
    Count modification sites from npz files for each histone marker
    
    Args:
        npz_directory: Path to directory containing npz files
    """
    
    # The 7 histone markers from the paper
    histone_markers = [
        'H3K4me1', 'H3K4me3', 'H3K27me3', 
        'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
    ]
    
    results = {}
    total_sites = 0
    
    print("Histone Modification Site Counts:")
    print("=" * 40)
    
    for marker in histone_markers:
        # Your specific file naming pattern
        possible_files = [
            f"E003_{marker}_deephistone.npz",  # your exact pattern
            f"{marker}.npz",
            f"{marker}_sites.npz",
            f"E003_{marker}.npz"
        ]
        
        npz_file = None
        for filename in possible_files:
            filepath = Path(npz_directory) / filename
            if filepath.exists():
                npz_file = filepath
                break
        
        if npz_file:
            try:
                # Load the npz file
                data = np.load(npz_file)
                
                # Print what keys are available
                print(f"\n{marker}:")
                print(f"  File: {npz_file.name}")
                print(f"  Available keys: {list(data.keys())}")
                
                # Try to find the data array
                # Common key names for the actual data
                possible_keys = ['data', 'sites', 'positions', 'arr_0', 'peaks']
                
                site_count = None
                for key in possible_keys:
                    if key in data.keys():
                        arr = data[key]
                        site_count = len(arr) if arr.ndim == 1 else arr.shape[0]
                        print(f"  Shape of '{key}': {arr.shape}")
                        print(f"  Number of sites: {site_count:,}")
                        break
                
                if site_count is None:
                    # If no standard key found, try the first array
                    first_key = list(data.keys())[0]
                    arr = data[first_key]
                    site_count = len(arr) if arr.ndim == 1 else arr.shape[0]
                    print(f"  Using first key '{first_key}': {arr.shape}")
                    print(f"  Number of sites: {site_count:,}")
                
                results[marker] = site_count
                total_sites += site_count
                
                # Close the file
                data.close()
                
            except Exception as e:
                print(f"  Error loading {marker}: {e}")
                results[marker] = 0
        else:
            print(f"\n{marker}: File not found")
            print(f"  Looked for: {possible_files}")
            results[marker] = 0
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print("=" * 40)
    
    for marker in histone_markers:
        count = results.get(marker, 0)
        print(f"{marker:>10}: {count:>10,} sites")
    
    print("-" * 40)
    print(f"{'TOTAL':>10}: {total_sites:>10,} sites")
    
    # Compare with paper results for E003
    paper_counts = {
        'H3K4me1': 342503, 'H3K4me3': 213546, 'H3K27me3': 231823,
        'H3K36me3': 424644, 'H3K9me3': 322656, 'H3K9ac': 155558, 'H3K27ac': 236273
    }
    paper_total = 1460537
    
    print("\n" + "=" * 40)
    print("COMPARISON WITH PAPER (E003):")
    print("=" * 40)
    print(f"{'Marker':>10} {'Your Count':>12} {'Paper Count':>12} {'Difference':>12}")
    print("-" * 50)
    
    for marker in histone_markers:
        your_count = results.get(marker, 0)
        paper_count = paper_counts[marker]
        diff = your_count - paper_count
        diff_pct = (diff / paper_count * 100) if paper_count > 0 else 0
        
        print(f"{marker:>10} {your_count:>12,} {paper_count:>12,} {diff:>+8,} ({diff_pct:>+5.1f}%)")
    
    print("-" * 50)
    print(f"{'TOTAL':>10} {total_sites:>12,} {paper_total:>12,} {total_sites-paper_total:>+8,} ({(total_sites-paper_total)/paper_total*100:>+5.1f}%)")
    
    return results

def inspect_npz_file(filepath):
    """
    Detailed inspection of a single npz file
    """
    print(f"\nDetailed inspection of {filepath}:")
    print("=" * 50)
    
    try:
        data = np.load(filepath)
        
        for key in data.keys():
            arr = data[key]
            print(f"\nKey: '{key}'")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Size: {arr.size:,} elements")
            
            if arr.size > 0:
                if arr.dtype.kind in ['i', 'f']:  # numeric data
                    print(f"  Min: {arr.min()}")
                    print(f"  Max: {arr.max()}")
                    if arr.size <= 10:
                        print(f"  Values: {arr}")
                    else:
                        print(f"  First 5: {arr.flat[:5]}")
                        print(f"  Last 5: {arr.flat[-5:]}")
        
        data.close()
        
    except Exception as e:
        print(f"Error inspecting file: {e}")

# Example usage:
if __name__ == "__main__":
    # Your npz files are in data/conversion
    npz_directory = "data/"
    
    # Count sites from all npz files
    results = count_histone_sites(npz_directory)
