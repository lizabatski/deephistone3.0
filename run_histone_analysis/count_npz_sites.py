import numpy as np
import os
from pathlib import Path

# Epigenome configurations - easily extensible
EPIGENOME_CONFIGS = {
    'E003': {
        'name': 'ESC',
        'anatomy': 'ESC',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E003_{marker}_deephistone_expected_format.npz",
            "E003_{marker}_deephistone.npz",
            "{marker}.npz",
            "{marker}_sites.npz",
            "E003_{marker}.npz"
        ],
        'paper_counts': {
            'H3K4me1': 342503, 'H3K4me3': 213546, 'H3K27me3': 231823,
            'H3K36me3': 424644, 'H3K9me3': 322656, 'H3K9ac': 155558, 'H3K27ac': 236273
        }
    },
    'E005': {
        'name': 'ESC_DERIVED',
        'anatomy': 'ESC_DERIVED',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E005_{marker}_deephistone_expected_format.npz",
            "E005_{marker}_deephistone.npz",
            "E005_{marker}.npz",
            "{marker}_E005.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 433887, 'H3K4me3': 166189, 'H3K27me3': 107589,
            'H3K36me3': 456459, 'H3K9me3': 287790, 'H3K9ac': 71753, 'H3K27ac': 258471
        }
    },
    'E006': {
        'name': 'ESC_DERIVED',
        'anatomy': 'ESC_DERIVED',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E006_{marker}_deephistone_expected_format.npz",
            "E006_{marker}_deephistone.npz",
            "E006_{marker}.npz",
            "{marker}_E006.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 462960, 'H3K4me3': 163250, 'H3K27me3': 130314,
            'H3K36me3': 487545, 'H3K9me3': 278273, 'H3K9ac': 168264, 'H3K27ac': 289090
        }
    },
    'E007': {
        'name': 'ESC_DERIVED',
        'anatomy': 'ESC_DERIVED',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E007_{marker}_deephistone_expected_format.npz",
            "E007_{marker}_deephistone.npz",
            "E007_{marker}.npz",
            "{marker}_E007.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 783177, 'H3K4me3': 150076, 'H3K27me3': 692940,
            'H3K36me3': 1926670, 'H3K9me3': 1494941, 'H3K9ac': 294433, 'H3K27ac': 1118346
        }
    },
    'E008': {
        'name': 'ESC',
        'anatomy': 'ESC',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E008_{marker}_deephistone_expected_format.npz",
            "E008_{marker}_deephistone.npz",
            "E008_{marker}.npz",
            "{marker}_E008.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 147743, 'H3K4me3': 208835, 'H3K27me3': 80768,
            'H3K36me3': 336585, 'H3K9me3': 234898, 'H3K9ac': 168324, 'H3K27ac': 129238
        }
    },
    'E017': {
        'name': 'LUNG',
        'anatomy': 'LUNG',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E017_{marker}_deephistone_expected_format.npz",
            "E017_{marker}_deephistone.npz",
            "E017_{marker}.npz",
            "{marker}_E017.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 548594, 'H3K4me3': 191906, 'H3K27me3': 506666,
            'H3K36me3': 682735, 'H3K9me3': 531686, 'H3K9ac': 273512, 'H3K27ac': 382596
        }
    },
    'E114': {
        'name': 'LUNG',
        'anatomy': 'LUNG',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E114_{marker}_deephistone_expected_format.npz",
            "E114_{marker}_deephistone.npz",
            "E114_{marker}.npz",
            "{marker}_E114.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 651428, 'H3K4me3': 362931, 'H3K27me3': 483416,
            'H3K36me3': 585379, 'H3K9me3': 216237, 'H3K9ac': 381391, 'H3K27ac': 430496
        }
    },
    'E117': {
        'name': 'CERVIX',
        'anatomy': 'CERVIX',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E117_{marker}_deephistone_expected_format.npz",
            "E117_{marker}_deephistone.npz",
            "E117_{marker}.npz",
            "{marker}_E117.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 839800, 'H3K4me3': 285181, 'H3K27me3': 406167,
            'H3K36me3': 347919, 'H3K9me3': 95082, 'H3K9ac': 260910, 'H3K27ac': 347565
        }
    },
    'E118': {
        'name': 'LIVER',
        'anatomy': 'LIVER',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E118_{marker}_deephistone_expected_format.npz",
            "E118_{marker}_deephistone.npz",
            "E118_{marker}.npz",
            "{marker}_E118.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 541823, 'H3K4me3': 271863, 'H3K27me3': 205199,
            'H3K36me3': 340891, 'H3K9me3': 245440, 'H3K9ac': 220457, 'H3K27ac': 278768
        }
    },
    'E119': {
        'name': 'BREAST',
        'anatomy': 'BREAST',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E119_{marker}_deephistone_expected_format.npz",
            "E119_{marker}_deephistone.npz",
            "E119_{marker}.npz",
            "{marker}_E119.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 565758, 'H3K4me3': 192740, 'H3K27me3': 260776,
            'H3K36me3': 242863, 'H3K9me3': 376845, 'H3K9ac': 232169, 'H3K27ac': 366810
        }
    },
    'E121': {
        'name': 'MUSCLE',
        'anatomy': 'MUSCLE',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E121_{marker}_deephistone_expected_format.npz",
            "E121_{marker}_deephistone.npz",
            "E121_{marker}.npz",
            "{marker}_E121.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 591321, 'H3K4me3': 244747, 'H3K27me3': 708330,
            'H3K36me3': 483319, 'H3K9me3': 156381, 'H3K9ac': 356341, 'H3K27ac': 486568
        }
    },
    'E122': {
        'name': 'VASCULAR',
        'anatomy': 'VASCULAR',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E122_{marker}_deephistone_expected_format.npz",
            "E122_{marker}_deephistone.npz",
            "E122_{marker}.npz",
            "{marker}_E122.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 582593, 'H3K4me3': 211344, 'H3K27me3': 404019,
            'H3K36me3': 517660, 'H3K9me3': 252718, 'H3K9ac': 260503, 'H3K27ac': 360372
        }
    },
    'E124': {
        'name': 'BLOOD',
        'anatomy': 'BLOOD',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E124_{marker}_deephistone_expected_format.npz",
            "E124_{marker}_deephistone.npz",
            "E124_{marker}.npz",
            "{marker}_E124.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 839651, 'H3K4me3': 431883, 'H3K27me3': 597823,
            'H3K36me3': 1098094, 'H3K9me3': 570919, 'H3K9ac': 268232, 'H3K27ac': 655859
        }
    },
    'E125': {
        'name': 'BRAIN',
        'anatomy': 'BRAIN',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E125_{marker}_deephistone_expected_format.npz",
            "E125_{marker}_deephistone.npz",
            "E125_{marker}.npz",
            "{marker}_E125.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 627409, 'H3K4me3': 258248, 'H3K27me3': 321184,
            'H3K36me3': 400306, 'H3K9me3': 178567, 'H3K9ac': 359857, 'H3K27ac': 424867
        }
    },
    'E127': {
        'name': 'SKIN',
        'anatomy': 'SKIN',
        'histone_markers': [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ],
        'file_patterns': [
            "E127_{marker}_deephistone_expected_format.npz",
            "E127_{marker}_deephistone.npz",
            "E127_{marker}.npz",
            "{marker}_E127.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ],
        'paper_counts': {
            'H3K4me1': 788918, 'H3K4me3': 200273, 'H3K27me3': 466659,
            'H3K36me3': 513074, 'H3K9me3': 186065, 'H3K9ac': 302827, 'H3K27ac': 416314
        }
    }
}

def count_histone_sites(npz_directory, epigenome_id='E003'):
    """
    Count modification sites from npz files for each histone marker
    
    Args:
        npz_directory: Path to directory containing npz files
        epigenome_id: Epigenome identifier (e.g., 'E003', 'E116', 'E114')
    """
    
    if epigenome_id not in EPIGENOME_CONFIGS:
        raise ValueError(f"Unknown epigenome_id '{epigenome_id}'. Available: {list(EPIGENOME_CONFIGS.keys())}")
    
    config = EPIGENOME_CONFIGS[epigenome_id]
    histone_markers = config['histone_markers']
    file_patterns = config['file_patterns']
    
    results = {}
    total_sites = 0
    
    print(f"Histone Modification Site Counts - {config['name']} ({epigenome_id}):")
    print("=" * 60)
    
    for marker in histone_markers:
        # Try different file naming patterns for this epigenome
        possible_files = [pattern.format(marker=marker) for pattern in file_patterns]
        
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
                
                site_count = 0
                if 'label' in data:
                    label = data['label']  # shape: (N, 1, 7)
                    marker_idx = histone_markers.index(marker)
                    site_count = int(np.sum(label[:, 0, marker_idx]))  # count where label == 1
                    print(f"  Number of positive sites in label[:, 0, {marker_idx}]: {site_count:,}")
                else:
                    print("  Warning: No 'label' key found in this file!")
                
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
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    for marker in histone_markers:
        count = results.get(marker, 0)
        print(f"{marker:>10}: {count:>10,} sites")
    
    print("-" * 60)
    print(f"{'TOTAL':>10}: {total_sites:>10,} sites")
    
    # Compare with paper results if available
    if 'paper_counts' in config:
        paper_counts = config['paper_counts']
        paper_total = sum(paper_counts.values())
        
        print("\n" + "=" * 60)
        print(f"COMPARISON WITH PAPER ({epigenome_id}):")
        print("=" * 60)
        print(f"{'Marker':>10} {'Your Count':>12} {'Paper Count':>12} {'Difference':>12}")
        print("-" * 60)
        
        for marker in histone_markers:
            your_count = results.get(marker, 0)
            paper_count = paper_counts.get(marker, 0)
            if paper_count > 0:
                diff = your_count - paper_count
                diff_pct = (diff / paper_count * 100)
                
                print(f"{marker:>10} {your_count:>12,} {paper_count:>12,} {diff:>+8,} ({diff_pct:>+5.1f}%)")
        
        print("-" * 60)
        print(f"{'TOTAL':>10} {total_sites:>12,} {paper_total:>12,} {total_sites-paper_total:>+8,} ({(total_sites-paper_total)/paper_total*100:>+5.1f}%)")
    else:
        print(f"\nNo reference counts available for {epigenome_id}")
    
    return results

def count_multiple_epigenomes(npz_directory, epigenome_ids=None):
    """
    Count sites for multiple epigenomes
    
    Args:
        npz_directory: Path to directory containing npz files
        epigenome_ids: List of epigenome IDs to process. If None, process all available.
    """
    if epigenome_ids is None:
        epigenome_ids = list(EPIGENOME_CONFIGS.keys())
    
    all_results = {}
    
    for epigenome_id in epigenome_ids:
        print(f"\n{'='*80}")
        print(f"Processing {epigenome_id}")
        print(f"{'='*80}")
        
        try:
            results = count_histone_sites(npz_directory, epigenome_id)
            all_results[epigenome_id] = results
        except Exception as e:
            print(f"Error processing {epigenome_id}: {e}")
            all_results[epigenome_id] = {}
    
    # Summary across all epigenomes
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL EPIGENOMES:")
    print(f"{'='*80}")
    
    # Get all unique markers
    all_markers = set()
    for results in all_results.values():
        all_markers.update(results.keys())
    all_markers = sorted(all_markers)
    
    # Print header
    header = f"{'Epigenome':>12}"
    for marker in all_markers:
        header += f" {marker:>10}"
    header += f" {'Total':>12}"
    print(header)
    print("-" * len(header))
    
    # Print results for each epigenome
    for epigenome_id, results in all_results.items():
        config = EPIGENOME_CONFIGS.get(epigenome_id, {})
        name = config.get('name', epigenome_id)[:12]
        row = f"{name:>12}"
        
        total = 0
        for marker in all_markers:
            count = results.get(marker, 0)
            row += f" {count:>10,}"
            total += count
        row += f" {total:>12,}"
        print(row)
    
    return all_results

def add_epigenome_config(epigenome_id, name, histone_markers=None, file_patterns=None, paper_counts=None):
    """
    Add a new epigenome configuration
    
    Args:
        epigenome_id: Unique identifier for the epigenome
        name: Human-readable name
        histone_markers: List of histone markers to look for (default: standard 7)
        file_patterns: List of file naming patterns (default: common patterns)
        paper_counts: Dictionary of expected counts for comparison (optional)
    """
    if histone_markers is None:
        histone_markers = [
            'H3K4me1', 'H3K4me3', 'H3K27me3', 
            'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
        ]
    
    if file_patterns is None:
        file_patterns = [
            f"{epigenome_id}_{{marker}}_deephistone.npz",
            f"{epigenome_id}_{{marker}}.npz",
            "{marker}_" + f"{epigenome_id}.npz",
            "{marker}.npz",
            "{marker}_sites.npz"
        ]
    
    config = {
        'name': name,
        'histone_markers': histone_markers,
        'file_patterns': file_patterns
    }
    
    if paper_counts:
        config['paper_counts'] = paper_counts
    
    EPIGENOME_CONFIGS[epigenome_id] = config
    print(f"Added epigenome configuration for {epigenome_id}: {name}")

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
    npz_directory = "data"
    
    # Count sites for E003 (original behavior)
    results_e003 = count_histone_sites(npz_directory, 'E003')
    
    # Count sites for another epigenome
    # results_e116 = count_histone_sites(npz_directory, 'E116')
    
    # Count sites for multiple epigenomes
    # all_results = count_multiple_epigenomes(npz_directory, ['E003', 'E116'])
    
    # Or count for all configured epigenomes
    # all_results = count_multiple_epigenomes(npz_directory)
    
    # Add a new epigenome configuration
    # add_epigenome_config(
    #     'E129', 
    #     'Bone Marrow Derived Mesenchymal Stem Cell',
    #     paper_counts={'H3K4me1': 150000, 'H3K4me3': 80000}  # if known
    # )
    
    # Optionally inspect a specific file in detail
    # inspect_npz_file("data/conversion/H3K4me1.npz")