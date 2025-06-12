#!/usr/bin/env python3
"""
Individual dataset converter for parallel processing on Compute Canada
Usage: python convert_single.py <epigenome_id> <marker>
"""

import numpy as np
import os
import json
import time
import sys
import logging
from datetime import datetime
import traceback


class SingleConverterConfig:
    def __init__(self):
        # Input/output directories
        self.INPUT_DIR = "../data"
        self.OUTPUT_DIR = "../data/converted"
        
        
        # File handling
        self.SKIP_EXISTING = True
        self.CONTINUE_ON_ERROR = True
        
        # The 7 histone markers in order
        self.ALL_MARKERS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
        
        # Test mode suffix (if you used test mode)
        self.TEST_MODE_SUFFIX = ""  # Set to "" if you didn't use test mode


config = SingleConverterConfig()


def setup_logging(epigenome_id, marker):
    """Set up logging for single conversion"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/convert_{epigenome_id}_{marker}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def find_input_file(epigenome_id, marker):
    """Find the input file for specific epigenome and marker"""
    if not os.path.exists(config.INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {config.INPUT_DIR}")
    
    # Look for files matching the pattern
    possible_patterns = [
        f"{epigenome_id}_{marker}_deephistone{config.TEST_MODE_SUFFIX}.npz",
        f"{epigenome_id}_{marker}_deephistone.npz"
    ]
    
    for pattern in possible_patterns:
        full_path = os.path.join(config.INPUT_DIR, pattern)
        if os.path.exists(full_path):
            return full_path, pattern
    
    raise FileNotFoundError(f"No input file found for {epigenome_id}-{marker}. Looked for: {possible_patterns}")


def get_marker_index(marker):
    """Get the index of a marker in the ALL_MARKERS list"""
    try:
        return config.ALL_MARKERS.index(marker)
    except ValueError:
        raise ValueError(f"Unknown marker {marker}. Valid markers: {config.ALL_MARKERS}")


def convert_dataset(epigenome_id, marker, logger):
    """Convert a single dataset to expected format"""
    start_time = time.time()
    
    logger.info(f"Starting conversion: {epigenome_id}-{marker}")
    
    print(f"\n{'='*60}")
    print(f"Converting: {epigenome_id}-{marker}")
    print(f"{'='*60}")
    
    # Find input file
    input_path, input_filename = find_input_file(epigenome_id, marker)
    print(f"Input: {input_filename}")
    
    # Check if output already exists
    output_filename = input_filename.replace('.npz', '_expected_format.npz')
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    
    if config.SKIP_EXISTING and os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_filename}")
        logger.info(f"Skipped {epigenome_id}-{marker} - already exists")
        return output_path, True, "skipped"
    
    # Load pipeline output
    print("Loading pipeline output...")
    data = np.load(input_path, allow_pickle=True)
    
    # Validate required keys
    required_keys = ['sequences', 'openness', 'labels']
    for key in required_keys:
        if key not in data.files:
            raise ValueError(f"Missing required key in input file: {key}")
    
    sequences = data['sequences']
    openness = data['openness'] 
    labels = data['labels']
    
    n_samples = len(sequences)
    print(f"Processing {n_samples:,} samples...")
    
    if n_samples == 0:
        raise ValueError("No samples found in input file")
    
    # Get or create genomic keys
    if 'keys' in data.files:
        keys = data['keys']
        print(f"Using existing genomic keys")
        print(f"  Examples: {keys[:3].tolist()}")
    else:
        keys = np.array([f"region_{i:06d}" for i in range(n_samples)], dtype='U30')
        print(f"Created generic keys")
    
    # Convert DNA sequences to one-hot encoding
    print(f"Converting {n_samples:,} DNA sequences to one-hot...")
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    dna_onehot = np.zeros((n_samples, 1, 4, 1000), dtype=np.float32)
    
    # Process sequences in chunks to show progress
    chunk_size = max(1, n_samples // 10)  # 10 progress updates
    
    for i in range(n_samples):
        if i % chunk_size == 0:
            progress = (i / n_samples) * 100
            print(f"  DNA conversion progress: {progress:.1f}% ({i:,}/{n_samples:,})")
        
        # Handle different sequence formats
        seq = sequences[i]
        if isinstance(seq, str):
            seq_str = seq
        else:
            seq_str = ''.join(seq)  # Convert character array to string
        
        # Convert to one-hot (ensure max 1000bp)
        seq_len = min(len(seq_str), 1000)
        for j in range(seq_len):
            base = seq_str[j].upper()
            base_idx = base_to_idx.get(base, 0)  # Default to 'A' for unknown bases
            dna_onehot[i, 0, base_idx, j] = 1.0
    
    print(f"DNA conversion complete: {dna_onehot.shape}")
    
    # Reshape DNase accessibility data
    print(f"Reshaping {n_samples:,} DNase accessibility arrays...")
    if openness.ndim == 1:
        # If 1D, assume each element is an array
        dnase_arrays = []
        for i in range(n_samples):
            if isinstance(openness[i], np.ndarray):
                dnase_array = openness[i][:1000]  # Take first 1000bp
                if len(dnase_array) < 1000:
                    # Pad with zeros if shorter
                    padded = np.zeros(1000, dtype=np.float32)
                    padded[:len(dnase_array)] = dnase_array
                    dnase_array = padded
                dnase_arrays.append(dnase_array)
            else:
                # If not an array, create zeros
                dnase_arrays.append(np.zeros(1000, dtype=np.float32))
        
        dnase = np.array(dnase_arrays).reshape(n_samples, 1, 1, 1000).astype(np.float32)
    else:
        # If already 2D, reshape directly
        dnase = openness.reshape(n_samples, 1, 1, 1000).astype(np.float32)
    
    print(f"DNase reshape complete: {dnase.shape}")
    
    # Create multi-task labels (7 histone markers)
    print(f"Creating multi-task labels for {n_samples:,} samples...")
    label = np.zeros((n_samples, 1, 7), dtype=np.float32)
    
    # Set the target marker at the correct index
    marker_idx = get_marker_index(marker)
    label[:, 0, marker_idx] = labels.astype(np.float32)
    
    print(f"Multi-task labels complete: {label.shape}")
    print(f"Target marker '{marker}' set at index {marker_idx}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save in expected format
    print(f"Saving {n_samples:,} samples to {output_filename}...")
    
    np.savez_compressed(
        output_path, 
        keys=keys, 
        dna=dna_onehot, 
        dnase=dnase, 
        label=label
    )
    
    # Verify the saved file
    verify_data = np.load(output_path)
    expected_keys = ['keys', 'dna', 'dnase', 'label']
    for key in expected_keys:
        if key not in verify_data.files:
            raise ValueError(f"Verification failed: missing key {key} in output file")
    
    # Calculate statistics
    pos_samples = int(np.sum(labels))
    neg_samples = n_samples - pos_samples
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"CONVERSION COMPLETED: {epigenome_id}-{marker}")
    print(f"{'='*50}")
    print(f"Output file: {output_filename}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Total samples: {n_samples:,}")
    print(f"Positive samples: {pos_samples:,} ({pos_samples/n_samples*100:.1f}%)")
    print(f"Negative samples: {neg_samples:,} ({neg_samples/n_samples*100:.1f}%)")
    print(f"Target marker: {marker} (index {marker_idx})")
    
    # Verify one-hot encoding
    sample_sums = np.sum(dna_onehot[:5], axis=2)  # Check first 5 samples
    print(f"One-hot verification: {sample_sums.min():.1f} to {sample_sums.max():.1f} (should be 1.0)")
    
    logger.info(f"Successfully converted {epigenome_id}-{marker}: {n_samples:,} samples, {file_size_mb:.1f}MB")
    
    return output_path, True, "converted"


def main():
    """Main function for single dataset conversion"""
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <epigenome_id> <marker>")
        print(f"Valid markers: {config.ALL_MARKERS}")
        sys.exit(1)
    
    epigenome_id = sys.argv[1]
    marker = sys.argv[2]
    
    # Validate marker
    if marker not in config.ALL_MARKERS:
        print(f"Error: Invalid marker '{marker}'")
        print(f"Valid markers: {config.ALL_MARKERS}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(epigenome_id, marker)
    
    try:
        output_path, success, status = convert_dataset(epigenome_id, marker, logger)
        
        if success:
            if status == "skipped":
                print(f"\nDataset {epigenome_id}-{marker} was skipped (already exists)")
                logger.info(f"Conversion skipped for {epigenome_id}-{marker}")
            else:
                print(f"\nDataset {epigenome_id}-{marker} converted successfully!")
                logger.info(f"Conversion completed for {epigenome_id}-{marker}")
            
            # Write success marker file
            success_file = f"{output_path}.success"
            with open(success_file, 'w') as f:
                f.write(f"Conversion completed: {datetime.now()}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Output: {output_path}\n")
            
            sys.exit(0)
        else:
            print(f"\nConversion failed for {epigenome_id}-{marker}")
            logger.error(f"Conversion failed for {epigenome_id}-{marker}")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"Error converting {epigenome_id}-{marker}: {str(e)}"
        print(f"ERROR: {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Individual dataset converter for parallel processing on Compute Canada
Usage: python convert_single.py <epigenome_id> <marker>
"""

import numpy as np
import os
import json
import time
import sys
import logging
from datetime import datetime
import traceback


class SingleConverterConfig:
    def __init__(self):
        # Input/output directories
        self.INPUT_DIR = "data"
        self.OUTPUT_DIR = "data/converted"
        
        # File handling
        self.SKIP_EXISTING = True
        self.CONTINUE_ON_ERROR = True
        
        # The 7 histone markers in order
        self.ALL_MARKERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']
        
        # Test mode suffix (if you used test mode)
        self.TEST_MODE_SUFFIX = "_chr22"  # Set to "" if you didn't use test mode


config = SingleConverterConfig()


def setup_logging(epigenome_id, marker):
    """Set up logging for single conversion"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/convert_{epigenome_id}_{marker}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def find_input_file(epigenome_id, marker):
    """Find the input file for specific epigenome and marker"""
    if not os.path.exists(config.INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {config.INPUT_DIR}")
    
    # Look for files matching the pattern
    possible_patterns = [
        f"{epigenome_id}_{marker}_deephistone{config.TEST_MODE_SUFFIX}.npz",
        f"{epigenome_id}_{marker}_deephistone.npz"
    ]
    
    for pattern in possible_patterns:
        full_path = os.path.join(config.INPUT_DIR, pattern)
        if os.path.exists(full_path):
            return full_path, pattern
    
    raise FileNotFoundError(f"No input file found for {epigenome_id}-{marker}. Looked for: {possible_patterns}")


def get_marker_index(marker):
    """Get the index of a marker in the ALL_MARKERS list"""
    try:
        return config.ALL_MARKERS.index(marker)
    except ValueError:
        raise ValueError(f"Unknown marker {marker}. Valid markers: {config.ALL_MARKERS}")


def convert_dataset(epigenome_id, marker, logger):
    """Convert a single dataset to expected format"""
    start_time = time.time()
    
    logger.info(f"Starting conversion: {epigenome_id}-{marker}")
    
    print(f"\n{'='*60}")
    print(f"Converting: {epigenome_id}-{marker}")
    print(f"{'='*60}")
    
    # Find input file
    input_path, input_filename = find_input_file(epigenome_id, marker)
    print(f"Input: {input_filename}")
    
    # Check if output already exists
    output_filename = input_filename.replace('.npz', '_expected_format.npz')
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    
    if config.SKIP_EXISTING and os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_filename}")
        logger.info(f"Skipped {epigenome_id}-{marker} - already exists")
        return output_path, True, "skipped"
    
    # Load pipeline output
    print("Loading pipeline output...")
    data = np.load(input_path, allow_pickle=True)
    
    # Validate required keys
    required_keys = ['sequences', 'openness', 'labels']
    for key in required_keys:
        if key not in data.files:
            raise ValueError(f"Missing required key in input file: {key}")
    
    sequences = data['sequences']
    openness = data['openness'] 
    labels = data['labels']
    
    n_samples = len(sequences)
    print(f"Processing {n_samples:,} samples...")
    
    if n_samples == 0:
        raise ValueError("No samples found in input file")
    
    # Get or create genomic keys
    if 'keys' in data.files:
        keys = data['keys']
        print(f"Using existing genomic keys")
        print(f"  Examples: {keys[:3].tolist()}")
    else:
        keys = np.array([f"region_{i:06d}" for i in range(n_samples)], dtype='U30')
        print(f"Created generic keys")
    
    # Convert DNA sequences to one-hot encoding
    print(f"Converting {n_samples:,} DNA sequences to one-hot...")
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    dna_onehot = np.zeros((n_samples, 1, 4, 1000), dtype=np.float32)
    
    # Process sequences in chunks to show progress
    chunk_size = max(1, n_samples // 10)  # 10 progress updates
    
    for i in range(n_samples):
        if i % chunk_size == 0:
            progress = (i / n_samples) * 100
            print(f"  DNA conversion progress: {progress:.1f}% ({i:,}/{n_samples:,})")
        
        # Handle different sequence formats
        seq = sequences[i]
        if isinstance(seq, str):
            seq_str = seq
        else:
            seq_str = ''.join(seq)  # Convert character array to string
        
        # Convert to one-hot (ensure max 1000bp)
        seq_len = min(len(seq_str), 1000)
        for j in range(seq_len):
            base = seq_str[j].upper()
            base_idx = base_to_idx.get(base, 0)  # Default to 'A' for unknown bases
            dna_onehot[i, 0, base_idx, j] = 1.0
    
    print(f"DNA conversion complete: {dna_onehot.shape}")
    
    # Reshape DNase accessibility data
    print(f"Reshaping {n_samples:,} DNase accessibility arrays...")
    if openness.ndim == 1:
        # If 1D, assume each element is an array
        dnase_arrays = []
        for i in range(n_samples):
            if isinstance(openness[i], np.ndarray):
                dnase_array = openness[i][:1000]  # Take first 1000bp
                if len(dnase_array) < 1000:
                    # Pad with zeros if shorter
                    padded = np.zeros(1000, dtype=np.float32)
                    padded[:len(dnase_array)] = dnase_array
                    dnase_array = padded
                dnase_arrays.append(dnase_array)
            else:
                # If not an array, create zeros
                dnase_arrays.append(np.zeros(1000, dtype=np.float32))
        
        dnase = np.array(dnase_arrays).reshape(n_samples, 1, 1, 1000).astype(np.float32)
    else:
        # If already 2D, reshape directly
        dnase = openness.reshape(n_samples, 1, 1, 1000).astype(np.float32)
    
    print(f"DNase reshape complete: {dnase.shape}")
    
    # Create multi-task labels (7 histone markers)
    print(f"Creating multi-task labels for {n_samples:,} samples...")
    label = np.zeros((n_samples, 1, 7), dtype=np.float32)
    
    # Set the target marker at the correct index
    marker_idx = get_marker_index(marker)
    label[:, 0, marker_idx] = labels.astype(np.float32)
    
    print(f"Multi-task labels complete: {label.shape}")
    print(f"Target marker '{marker}' set at index {marker_idx}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save in expected format
    print(f"Saving {n_samples:,} samples to {output_filename}...")
    
    np.savez_compressed(
        output_path, 
        keys=keys, 
        dna=dna_onehot, 
        dnase=dnase, 
        label=label
    )
    
    # Verify the saved file
    verify_data = np.load(output_path)
    expected_keys = ['keys', 'dna', 'dnase', 'label']
    for key in expected_keys:
        if key not in verify_data.files:
            raise ValueError(f"Verification failed: missing key {key} in output file")
    
    # Calculate statistics
    pos_samples = int(np.sum(labels))
    neg_samples = n_samples - pos_samples
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"CONVERSION COMPLETED: {epigenome_id}-{marker}")
    print(f"{'='*50}")
    print(f"Output file: {output_filename}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Total samples: {n_samples:,}")
    print(f"Positive samples: {pos_samples:,} ({pos_samples/n_samples*100:.1f}%)")
    print(f"Negative samples: {neg_samples:,} ({neg_samples/n_samples*100:.1f}%)")
    print(f"Target marker: {marker} (index {marker_idx})")
    
    # Verify one-hot encoding
    sample_sums = np.sum(dna_onehot[:5], axis=2)  # Check first 5 samples
    print(f"One-hot verification: {sample_sums.min():.1f} to {sample_sums.max():.1f} (should be 1.0)")
    
    logger.info(f"Successfully converted {epigenome_id}-{marker}: {n_samples:,} samples, {file_size_mb:.1f}MB")
    
    return output_path, True, "converted"


def main():
    """Main function for single dataset conversion"""
    if len(sys.argv) != 3:
        print("Usage: python convert_single.py <epigenome_id> <marker>")
        print(f"Valid markers: {config.ALL_MARKERS}")
        sys.exit(1)
    
    epigenome_id = sys.argv[1]
    marker = sys.argv[2]
    
    # Validate marker
    if marker not in config.ALL_MARKERS:
        print(f"Error: Invalid marker '{marker}'")
        print(f"Valid markers: {config.ALL_MARKERS}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(epigenome_id, marker)
    
    try:
        output_path, success, status = convert_dataset(epigenome_id, marker, logger)
        
        if success:
            if status == "skipped":
                print(f"\nDataset {epigenome_id}-{marker} was skipped (already exists)")
                logger.info(f"Conversion skipped for {epigenome_id}-{marker}")
            else:
                print(f"\nDataset {epigenome_id}-{marker} converted successfully!")
                logger.info(f"Conversion completed for {epigenome_id}-{marker}")
            
            # Write success marker file
            success_file = f"{output_path}.success"
            with open(success_file, 'w') as f:
                f.write(f"Conversion completed: {datetime.now()}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Output: {output_path}\n")
            
            sys.exit(0)
        else:
            print(f"\nConversion failed for {epigenome_id}-{marker}")
            logger.error(f"Conversion failed for {epigenome_id}-{marker}")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"Error converting {epigenome_id}-{marker}: {str(e)}"
        print(f"ERROR: {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
