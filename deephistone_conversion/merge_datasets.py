#!/usr/bin/env python3
"""
Merge multiple converted datasets into a single multi-task dataset
Usage: python merge_datasets.py [--output merged_dataset.npz]
"""

import numpy as np
import os
import argparse
import time
import logging
from datetime import datetime
import glob


class MergerConfig:
    def __init__(self):
        # Input/output directories
        self.INPUT_DIR = "../data/converted"
        self.DEFAULT_OUTPUT = "../data/merged_dataset.npz"
        
        # The 7 histone markers in order
        self.ALL_MARKERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K9ac','H3K27ac']
        
        # Processing settings
        self.VERIFY_CONSISTENCY = True  # Verify that all datasets have same samples
        self.HANDLE_MISSING_MARKERS = True  # Fill missing markers with zeros


config = MergerConfig()


def setup_logging():
    """Set up logging for dataset merging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/merge_datasets_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def find_converted_datasets():
    """Find all converted dataset files"""
    if not os.path.exists(config.INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {config.INPUT_DIR}")
    
    # Look for files with '_expected_format.npz' suffix
    pattern = os.path.join(config.INPUT_DIR, "*_expected_format.npz")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No converted datasets found in {config.INPUT_DIR}")
    
    # Parse filenames to extract epigenome and marker info
    datasets = []
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            # Expected format: E003_H3K4me1_deephistone[_chr22]_expected_format.npz
            parts = filename.replace('_expected_format.npz', '').split('_')
            
            if len(parts) >= 3:
                epigenome_id = parts[0]
                marker = parts[1]
                
                if marker in config.ALL_MARKERS:
                    datasets.append({
                        'epigenome_id': epigenome_id,
                        'marker': marker,
                        'marker_idx': config.ALL_MARKERS.index(marker),
                        'filepath': filepath,
                        'filename': filename
                    })
                    
        except Exception as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue
    
    return sorted(datasets, key=lambda x: (x['epigenome_id'], x['marker_idx']))


def load_and_verify_dataset(dataset_info, logger):
    """Load and verify a single converted dataset"""
    filepath = dataset_info['filepath']
    epigenome_id = dataset_info['epigenome_id']
    marker = dataset_info['marker']
    
    logger.info(f"Loading {epigenome_id}-{marker} from {os.path.basename(filepath)}")
    
    try:
        data = np.load(filepath)
        
        # Verify expected keys
        expected_keys = ['keys', 'dna', 'dnase', 'label']
        for key in expected_keys:
            if key not in data.files:
                raise ValueError(f"Missing key '{key}' in {filepath}")
        
        # Get shapes and basic info
        n_samples = len(data['keys'])
        dna_shape = data['dna'].shape
        dnase_shape = data['dnase'].shape
        label_shape = data['label'].shape
        
        # Verify shapes
        expected_shapes = {
            'dna': (n_samples, 1, 4, 1000),
            'dnase': (n_samples, 1, 1, 1000),
            'label': (n_samples, 1, 7)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = data[key].shape
            if actual_shape != expected_shape:
                raise ValueError(f"Unexpected shape for {key}: {actual_shape}, expected {expected_shape}")
        
        # Check which marker is active
        active_markers = []
        for i in range(7):
            if np.any(data['label'][:, 0, i] > 0):
                active_markers.append(i)
        
        expected_marker_idx = dataset_info['marker_idx']
        if len(active_markers) != 1 or active_markers[0] != expected_marker_idx:
            print(f"Warning: Expected marker {marker} (idx {expected_marker_idx}), "
                  f"but found active markers at indices {active_markers}")
        
        pos_samples = int(np.sum(data['label'][:, 0, expected_marker_idx]))
        
        logger.info(f"  Loaded {n_samples:,} samples, {pos_samples:,} positive for {marker}")
        
        return {
            'data': data,
            'n_samples': n_samples,
            'pos_samples': pos_samples,
            'info': dataset_info
        }
        
    except Exception as e:
        logger.error(f"Error loading {epigenome_id}-{marker}: {str(e)}")
        raise


def verify_dataset_consistency(loaded_datasets, logger):
    """Verify that all datasets have consistent samples"""
    if not config.VERIFY_CONSISTENCY:
        logger.info("Skipping consistency verification")
        return True
    
    logger.info("Verifying dataset consistency...")
    
    # Check that all datasets have the same number of samples
    sample_counts = [d['n_samples'] for d in loaded_datasets]
    if len(set(sample_counts)) > 1:
        logger.warning(f"Inconsistent sample counts: {sample_counts}")
        if not config.HANDLE_MISSING_MARKERS:
            raise ValueError("Sample counts don't match and HANDLE_MISSING_MARKERS is False")
    
    # Check that genomic keys match (if they exist)
    reference_keys = loaded_datasets[0]['data']['keys']
    n_reference = len(reference_keys)
    
    for i, dataset in enumerate(loaded_datasets[1:], 1):
        keys = dataset['data']['keys']
        if len(keys) != n_reference:
            logger.warning(f"Dataset {i} has {len(keys)} samples vs reference {n_reference}")
            continue
            
        # Check first few and last few keys
        check_indices = list(range(min(10, n_reference))) + list(range(max(0, n_reference-10), n_reference))
        mismatches = 0
        
        for idx in check_indices:
            if idx < len(keys) and keys[idx] != reference_keys[idx]:
                mismatches += 1
        
        if mismatches > 0:
            logger.warning(f"Dataset {i} has {mismatches} key mismatches with reference")
    
    logger.info("Consistency verification completed")
    return True


def merge_datasets(loaded_datasets, output_path, logger):
    """Merge multiple datasets into a single multi-task dataset"""
    start_time = time.time()
    
    logger.info(f"Starting dataset merging for {len(loaded_datasets)} datasets")
    
    # Determine the number of samples (use maximum if inconsistent)
    sample_counts = [d['n_samples'] for d in loaded_datasets]
    n_samples = max(sample_counts) if config.HANDLE_MISSING_MARKERS else sample_counts[0]
    
    logger.info(f"Target merged dataset size: {n_samples:,} samples")
    
    # Initialize merged arrays
    logger.info("Initializing merged arrays...")
    
    # Use the first dataset as reference for keys, dna, and dnase
    reference_data = loaded_datasets[0]['data']
    
    merged_keys = reference_data['keys'][:n_samples] if len(reference_data['keys']) >= n_samples else reference_data['keys']
    merged_dna = reference_data['dna'][:n_samples] if len(reference_data['dna']) >= n_samples else reference_data['dna']
    merged_dnase = reference_data['dnase'][:n_samples] if len(reference_data['dnase']) >= n_samples else reference_data['dnase']
    
    # Initialize empty multi-task labels
    merged_labels = np.zeros((n_samples, 1, 7), dtype=np.float32)
    
    # Merge labels from each dataset
    logger.info("Merging labels from individual datasets...")
    
    for dataset in loaded_datasets:
        data = dataset['data']
        info = dataset['info']
        marker_idx = info['marker_idx']
        marker = info['marker']
        
        # Get labels for this marker
        current_labels = data['label'][:, 0, marker_idx]
        n_current = len(current_labels)
        
        # Copy to merged labels
        copy_samples = min(n_current, n_samples)
        merged_labels[:copy_samples, 0, marker_idx] = current_labels[:copy_samples]
        
        pos_samples = int(np.sum(current_labels[:copy_samples]))
        logger.info(f"  {marker}: {pos_samples:,}/{copy_samples:,} positive samples")
    
    # Verify merged dataset
    logger.info("Verifying merged dataset...")
    
    total_positive_per_marker = np.sum(merged_labels, axis=0)[0]
    logger.info("Positive samples per marker:")
    for i, marker in enumerate(config.ALL_MARKERS):
        pos_count = int(total_positive_per_marker[i])
        pos_percent = (pos_count / n_samples * 100) if n_samples > 0 else 0
        logger.info(f"  {marker}: {pos_count:,} ({pos_percent:.1f}%)")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save merged dataset
    logger.info(f"Saving merged dataset to {output_path}...")
    
    np.savez_compressed(
        output_path,
        keys=merged_keys,
        dna=merged_dna,
        dnase=merged_dnase,
        label=merged_labels
    )
    
    # Verify saved file
    logger.info("Verifying saved merged dataset...")
    verify_data = np.load(output_path)
    
    expected_keys = ['keys', 'dna', 'dnase', 'label']
    for key in expected_keys:
        if key not in verify_data.files:
            raise ValueError(f"Verification failed: missing key {key} in merged file")
    
    # Calculate final statistics
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    elapsed_time = time.time() - start_time
    
    # Summary statistics
    total_positive_samples = int(np.sum(merged_labels))
    total_possible_labels = n_samples * 7
    label_density = (total_positive_samples / total_possible_labels * 100) if total_possible_labels > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"DATASET MERGING COMPLETED")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Total samples: {n_samples:,}")
    print(f"Total markers: {len(config.ALL_MARKERS)}")
    print(f"Total positive labels: {total_positive_samples:,}")
    print(f"Label density: {label_density:.2f}%")
    
    print(f"\nPer-marker statistics:")
    for i, marker in enumerate(config.ALL_MARKERS):
        pos_count = int(total_positive_per_marker[i])
        pos_percent = (pos_count / n_samples * 100) if n_samples > 0 else 0
        print(f"  {marker:>10}: {pos_count:>7,} positive ({pos_percent:>5.1f}%)")
    
    print(f"\nMerged dataset shape verification:")
    print(f"  keys: {verify_data['keys'].shape}")
    print(f"  dna:  {verify_data['dna'].shape}")
    print(f"  dnase: {verify_data['dnase'].shape}")
    print(f"  label: {verify_data['label'].shape}")
    
    logger.info(f"Dataset merging completed successfully: {n_samples:,} samples, {file_size_mb:.1f}MB")
    
    return output_path


def main():
    """Main function for dataset merging"""
    parser = argparse.ArgumentParser(description="Merge converted datasets into a single multi-task dataset")
    parser.add_argument('--output', '-o', default=config.DEFAULT_OUTPUT,
                       help='Output path for merged dataset')
    parser.add_argument('--input-dir', default=config.INPUT_DIR,
                       help='Directory containing converted datasets')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify dataset consistency (default: True)')
    parser.add_argument('--no-verify', dest='verify', action='store_false',
                       help='Skip dataset consistency verification')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.INPUT_DIR = args.input_dir
    config.VERIFY_CONSISTENCY = args.verify
    
    # Setup logging
    logger = setup_logging()
    
    try:
        print(f"\n{'='*70}")
        print(f"DEEPHISTONE DATASET MERGER")
        print(f"{'='*70}")
        print(f"Input directory: {config.INPUT_DIR}")
        print(f"Output file: {args.output}")
        print(f"Verify consistency: {config.VERIFY_CONSISTENCY}")
        print(f"{'='*70}")
        
        # Find converted datasets
        logger.info("Finding converted datasets...")
        datasets = find_converted_datasets()
        
        if not datasets:
            print(f"No converted datasets found in {config.INPUT_DIR}")
            print("Make sure your conversion jobs completed successfully.")
            return
        
        print(f"\nFound {len(datasets)} converted datasets:")
        for dataset in datasets:
            print(f"  {dataset['epigenome_id']}-{dataset['marker']}: {dataset['filename']}")
        
        # Check if we have all 7 markers
        found_markers = set(d['marker'] for d in datasets)
        missing_markers = set(config.ALL_MARKERS) - found_markers
        
        if missing_markers:
            logger.warning(f"Missing markers: {missing_markers}")
            if not config.HANDLE_MISSING_MARKERS:
                raise ValueError(f"Missing markers and HANDLE_MISSING_MARKERS is False: {missing_markers}")
            else:
                print(f"Warning: Missing markers will be filled with zeros: {missing_markers}")
        
        # Load all datasets
        logger.info("Loading datasets...")
        loaded_datasets = []
        
        for i, dataset_info in enumerate(datasets, 1):
            print(f"\nLoading {i}/{len(datasets)}: {dataset_info['epigenome_id']}-{dataset_info['marker']}")
            loaded_dataset = load_and_verify_dataset(dataset_info, logger)
            loaded_datasets.append(loaded_dataset)
        
        # Verify consistency
        if config.VERIFY_CONSISTENCY:
            verify_dataset_consistency(loaded_datasets, logger)
        
        # Merge datasets
        print(f"\nMerging {len(loaded_datasets)} datasets...")
        output_path = merge_datasets(loaded_datasets, args.output, logger)
        
        print(f"\nMerging completed successfully!")
        print(f"Final dataset saved to: {output_path}")
        
        logger.info("Dataset merging completed successfully")
        
    except Exception as e:
        error_msg = f"Error during dataset merging: {str(e)}"
        print(f"ERROR: {error_msg}")
        logger.error(f"{error_msg}")
        raise


if __name__ == "__main__":
    main()
