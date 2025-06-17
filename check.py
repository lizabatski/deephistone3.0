import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def diagnose_npz_file(data_file):
    """Comprehensive diagnosis of NPZ data file"""
    print("="*60)
    print(f"DIAGNOSING: {data_file}")
    print("="*60)
    
    with np.load(data_file) as f:
        print(f"Keys in file: {list(f.keys())}")
        
        # load
        keys = f['keys'][:]
        dna_data = f['dna'][:]
        dnase_data = f['dnase'][:]
        label_data = f['label'][:]
        
        print(f"\n1. BASIC INFO:")
        print(f"   Total samples: {len(keys)}")
        print(f"   Keys shape: {keys.shape}")
        print(f"   DNA shape: {dna_data.shape}")
        print(f"   DNase shape: {dnase_data.shape}")
        print(f"   Label shape: {label_data.shape}")
        
        # Check data types
        print(f"\n2. DATA TYPES:")
        print(f"   Keys dtype: {keys.dtype}")
        print(f"   DNA dtype: {dna_data.dtype}")
        print(f"   DNase dtype: {dnase_data.dtype}")
        print(f"   Label dtype: {label_data.dtype}")
        
        # Memory usage
        total_size = sum([arr.nbytes for arr in [keys, dna_data, dnase_data, label_data]])
        print(f"\n3. MEMORY USAGE:")
        print(f"   Keys: {keys.nbytes/1e6:.1f} MB")
        print(f"   DNA: {dna_data.nbytes/1e6:.1f} MB") 
        print(f"   DNase: {dnase_data.nbytes/1e6:.1f} MB")
        print(f"   Labels: {label_data.nbytes/1e6:.1f} MB")
        print(f"   Total: {total_size/1e6:.1f} MB ({total_size/1e9:.2f} GB)")
        
        # Check for missing/invalid data
        print(f"\n4. DATA QUALITY CHECKS:")
        
        # DNA data checks
        print(f"   DNA data:")
        print(f"     Min: {dna_data.min()}, Max: {dna_data.max()}")
        print(f"     Contains NaN: {np.isnan(dna_data).any()}")
        print(f"     Contains Inf: {np.isinf(dna_data).any()}")
        print(f"     Unique values: {np.unique(dna_data)}")
        
        # Check if DNA is one-hot encoded properly
        if len(dna_data.shape) == 3:  # (samples, 4, length)
            dna_sums = np.sum(dna_data, axis=1)  # Sum across the 4 bases
            print(f"     One-hot check - should all be 1.0:")
            print(f"       Min sum: {dna_sums.min()}")
            print(f"       Max sum: {dna_sums.max()}")
            print(f"       Mean sum: {dna_sums.mean():.3f}")
        
        # DNase data checks
        print(f"   DNase data:")
        print(f"     Min: {dnase_data.min()}, Max: {dnase_data.max()}")
        print(f"     Mean: {dnase_data.mean():.3f}")
        print(f"     Std: {dnase_data.std():.3f}")
        print(f"     Contains NaN: {np.isnan(dnase_data).any()}")
        print(f"     Contains Inf: {np.isinf(dnase_data).any()}")
        print(f"     Zero values: {(dnase_data == 0).sum()} / {dnase_data.size} ({(dnase_data == 0).mean()*100:.1f}%)")
        
        # Label data checks - CRITICAL for performance
        print(f"   Label data:")
        print(f"     Min: {label_data.min()}, Max: {label_data.max()}")
        print(f"     Contains NaN: {np.isnan(label_data).any()}")
        print(f"     Unique values: {np.unique(label_data)}")
        
        # class imbalance
        print(f"\n5. CLASS BALANCE ANALYSIS (Critical for auPRC):")
        num_markers = label_data.shape[1]
        
        for i in range(num_markers):
            marker_labels = label_data[:, i]
            positive_count = np.sum(marker_labels == 1)
            negative_count = np.sum(marker_labels == 0)
            positive_ratio = positive_count / len(marker_labels)
            
            print(f"   Marker {i}:")
            print(f"     Positive: {positive_count:6d} ({positive_ratio:.1%})")
            print(f"     Negative: {negative_count:6d} ({1-positive_ratio:.1%})")
            print(f"     Ratio (pos:neg): 1:{negative_count/positive_count:.1f}")
            
            # flag severe imbalance
            if positive_ratio < 0.01:
                print(f"      WARNING: Very imbalanced! <1% positive")
            elif positive_ratio < 0.05:
                print(f"      WARNING: Highly imbalanced! <5% positive")
        
        # sample some data
        print(f"\n6. SAMPLE DATA INSPECTION:")
        print(f"   First 5 keys: {keys[:5]}")
        print(f"   Sample DNA shape: {dna_data[0].shape}")
        print(f"   Sample DNase shape: {dnase_data[0].shape}")
        print(f"   Sample labels: {label_data[0]}")
        
        # checking for duplicates
        print(f"\n7. DUPLICATE CHECK:")
        unique_keys = np.unique(keys)
        print(f"   Total keys: {len(keys)}")
        print(f"   Unique keys: {len(unique_keys)}")
        if len(keys) != len(unique_keys):
            print(f"     WARNING: {len(keys) - len(unique_keys)} duplicate keys found!")
        
        # Distribution plots
        print(f"\n8. GENERATING DISTRIBUTION PLOTS...")
        
        # Plot class distributions
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Class balance
        plt.subplot(2, 3, 1)
        positive_ratios = [np.mean(label_data[:, i]) for i in range(num_markers)]
        plt.bar(range(num_markers), positive_ratios)
        plt.title('Positive Class Ratios by Marker')
        plt.xlabel('Marker Index')
        plt.ylabel('Positive Ratio')
        plt.xticks(range(num_markers))
        
        # Subplot 2: DNase distribution
        plt.subplot(2, 3, 2)
        plt.hist(dnase_data.flatten(), bins=50, alpha=0.7)
        plt.title('DNase Signal Distribution')
        plt.xlabel('DNase Signal')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Subplot 3: DNase per sample
        plt.subplot(2, 3, 3)
        sample_dnase_means = np.mean(dnase_data, axis=1)
        plt.hist(sample_dnase_means, bins=50, alpha=0.7)
        plt.title('Mean DNase per Sample')
        plt.xlabel('Mean DNase Signal')
        plt.ylabel('Frequency')
        
        # Subplot 4: Label correlation matrix
        plt.subplot(2, 3, 4)
        label_corr = np.corrcoef(label_data.T)
        sns.heatmap(label_corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Label Correlation Matrix')
        
        # Subplot 5: DNA base composition
        if len(dna_data.shape) == 3:
            plt.subplot(2, 3, 5)
            base_counts = np.mean(dna_data, axis=(0, 2))  # Average across samples and positions
            bases = ['A', 'C', 'G', 'T']
            plt.bar(bases, base_counts)
            plt.title('Average Base Composition')
            plt.ylabel('Frequency')
        
        # Subplot 6: Sample-wise positive counts
        plt.subplot(2, 3, 6)
        sample_positive_counts = np.sum(label_data, axis=1)
        plt.hist(sample_positive_counts, bins=range(num_markers + 2), alpha=0.7)
        plt.title('Positive Markers per Sample')
        plt.xlabel('Number of Positive Markers')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary recommendations
        print(f"\n9. RECOMMENDATIONS:")
        
        # Check for common issues that cause low auPRC
        min_positive_ratio = min(positive_ratios)
        max_positive_ratio = max(positive_ratios)
        
        if min_positive_ratio < 0.01:
            print(f"    SEVERE IMBALANCE: Minimum positive ratio is {min_positive_ratio:.1%}")
            print(f"      This will cause very low auPRC scores!")
            print(f"      Consider: class weights, focal loss, or SMOTE")
        
        if max_positive_ratio > 0.5:
            print(f"    Some markers are very common ({max_positive_ratio:.1%} positive)")
            print(f"      These might be too easy to predict")
        
        if np.std(positive_ratios) > 0.2:
            print(f"    Large variation in class balance across markers")
            print(f"      Consider marker-specific loss weighting")
        
        dnase_zero_ratio = (dnase_data == 0).mean()
        if dnase_zero_ratio > 0.8:
            print(f"    {dnase_zero_ratio:.1%} of DNase values are zero")
            print(f"      This might indicate data preprocessing issues")
        
        if dna_data.dtype != np.float32 and dna_data.dtype != np.float64:
            print(f"    DNA data type is {dna_data.dtype}, should be float")
        
        if label_data.dtype not in [np.int32, np.int64, np.float32, np.float64]:
            print(f"   Label data type is {label_data.dtype}, check encoding")
        
        print(f"\nDiagnosis complete! Check 'data_diagnostics.png' for visualizations.")
        
        return {
            'num_samples': len(keys),
            'num_markers': num_markers,
            'positive_ratios': positive_ratios,
            'dnase_stats': {
                'min': dnase_data.min(),
                'max': dnase_data.max(), 
                'mean': dnase_data.mean(),
                'std': dnase_data.std()
            },
            'has_duplicates': len(keys) != len(unique_keys)
        }

if __name__ == "__main__":
    data_file = 'data/final/E005_all_markers_merged.npz'
    
    try:
        results = diagnose_npz_file(data_file)
        print(f"\nQuick Summary:")
        print(f"Samples: {results['num_samples']}")
        print(f"Markers: {results['num_markers']}")
        print(f"Positive ratios: {[f'{r:.1%}' for r in results['positive_ratios']]}")
        
    except FileNotFoundError:
        print(f"File not found: {data_file}")
        print("Please check the file path!")
    except Exception as e:
        print(f"Error: {e}")