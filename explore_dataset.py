#!/usr/bin/env python3
"""
Quick DeepHistone Dataset Explorer
Explores the contents of DeepHistone .npz dataset files
"""

import numpy as np
import json
import sys
import os
from collections import Counter

def explore_dataset(filepath):
    """Explore a DeepHistone dataset file"""
    
    print("="*60)
    print(f"EXPLORING DATASET: {os.path.basename(filepath)}")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024*1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        # Load the dataset
        print("\n📂 Loading dataset...")
        data = np.load(filepath, allow_pickle=True)
        
        # Show available keys
        print(f"✅ Dataset loaded successfully")
        print(f"Available keys: {list(data.keys())}")
        
        # Basic dataset info
        print(f"\n📊 DATASET OVERVIEW")
        print("-" * 30)
        
        # Check shapes
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                print(f"{key:12}: {data[key].shape}")
            else:
                print(f"{key:12}: {type(data[key])}")
        
        # Analyze sequences
        if 'sequences' in data:
            sequences = data['sequences']
            print(f"\n🧬 SEQUENCE ANALYSIS")
            print("-" * 30)
            print(f"Total samples: {len(sequences):,}")
            print(f"Sequence length: {sequences.shape[1]} bp")
            
            # Check sequence composition
            sample_seq = ''.join(sequences[0]).rstrip('N')
            base_counts = Counter(sample_seq)
            print(f"Sample sequence composition:")
            for base in ['A', 'T', 'G', 'C', 'N']:
                count = base_counts.get(base, 0)
                percent = count / len(sample_seq) * 100 if len(sample_seq) > 0 else 0
                print(f"  {base}: {count:,} ({percent:.1f}%)")
        
        # Analyze labels
        if 'labels' in data:
            labels = data['labels']
            print(f"\n🏷️  LABEL ANALYSIS")
            print("-" * 30)
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            
            for label, count in zip(unique, counts):
                label_name = 'Positive (H3K4me3)' if label == 1 else 'Negative (other marks)'
                percent = count / total * 100
                print(f"{label_name}: {count:,} ({percent:.1f}%)")
            
            if len(unique) == 2:
                neg_count = counts[0] if unique[0] == 0 else counts[1]
                pos_count = counts[1] if unique[1] == 1 else counts[0]
                ratio = neg_count / pos_count if pos_count > 0 else 0
                print(f"Class ratio (neg:pos): {ratio:.1f}:1")
        
        # Analyze openness scores
        if 'openness' in data:
            openness = data['openness']
            print(f"\n🔓 OPENNESS ANALYSIS")
            print("-" * 30)
            print(f"Openness shape: {openness.shape}")
            print(f"Value range: {openness.min():.3f} to {openness.max():.3f}")
            print(f"Mean openness: {openness.mean():.3f}")
            
            # Check how many positions have openness > 0
            non_zero_frac = np.mean(openness > 0) * 100
            print(f"Positions with openness > 0: {non_zero_frac:.1f}%")
        
        # Show genomic locations
        if 'keys' in data:
            keys = data['keys']
            print(f"\n📍 GENOMIC LOCATIONS")
            print("-" * 30)
            print(f"Sample locations (first 5):")
            for i in range(min(5, len(keys))):
                print(f"  {i+1}: {keys[i]}")
            
            # Analyze chromosome distribution
            chromosomes = [key.split(':')[0] for key in keys[:1000]]  # Sample first 1000
            chrom_counts = Counter(chromosomes)
            print(f"\nChromosome distribution (sample):")
            for chrom, count in sorted(chrom_counts.items())[:10]:
                print(f"  {chrom}: {count}")
        
        # Show metadata
        if 'metadata' in data:
            print(f"\n📋 METADATA")
            print("-" * 30)
            try:
                metadata = json.loads(str(data['metadata']))
                for key, value in metadata.items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"{key}: [{value[0]}, {value[1]}, ... {len(value)} items]")
                    else:
                        print(f"{key}: {value}")
            except:
                print(f"Could not parse metadata: {data['metadata']}")
        
        # Show sample data
        print(f"\n🔬 SAMPLE DATA")
        print("-" * 30)
        
        # Show first few samples
        if 'sequences' in data and 'labels' in data:
            sequences = data['sequences']
            labels = data['labels']
            
            for i in range(min(3, len(sequences))):
                seq = ''.join(sequences[i]).rstrip('N')
                label = 'POSITIVE' if labels[i] == 1 else 'NEGATIVE'
                location = data['keys'][i] if 'keys' in data and i < len(data['keys']) else 'Unknown'
                
                print(f"\nSample {i+1} ({label}):")
                print(f"  Location: {location}")
                print(f"  Sequence: {seq[:80]}{'...' if len(seq) > 80 else ''}")
                print(f"  Length: {len(seq)} bp")
                
                if 'openness' in data:
                    openness_sample = data['openness'][i]
                    open_positions = np.sum(openness_sample > 0)
                    max_openness = np.max(openness_sample)
                    print(f"  Openness: {open_positions}/{len(openness_sample)} positions open (max: {max_openness:.2f})")
        
        print(f"\n✅ Dataset exploration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("DeepHistone Dataset Explorer")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for datasets in current directory
        import glob
        
        # Find NPZ files
        npz_files = glob.glob("data/*.npz")
        
        if not npz_files:
            print("No .npz files found in data/ directory")
            print("\nUsage:")
            print("  python3 explore_dataset.py <path_to_dataset.npz>")
            print("\nOr place .npz files in data/ directory")
            return
        
        # Show available files
        print(f"Found {len(npz_files)} dataset files:")
        for i, f in enumerate(npz_files, 1):
            size_mb = os.path.getsize(f) / (1024*1024)
            print(f"  {i}: {os.path.basename(f)} ({size_mb:.1f} MB)")
        
        # Ask user to choose
        while True:
            try:
                choice = input(f"\nChoose file to explore (1-{len(npz_files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return
                
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(npz_files):
                    filepath = npz_files[file_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(npz_files)}")
            except ValueError:
                print("Please enter a valid number or 'q'")
    
    # Explore the selected dataset
    success = explore_dataset(filepath)
    
    if success:
        print(f"\n🎉 Successfully explored: {os.path.basename(filepath)}")
    else:
        print(f"\n❌ Failed to explore: {os.path.basename(filepath)}")

if __name__ == "__main__":
    main()
