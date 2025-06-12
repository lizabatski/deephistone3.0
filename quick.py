import numpy as np

def randomly_sample_mini_labels(data_file, num_samples=20):
    """Randomly sample and display labels from mini dataset"""
    
    print("RANDOM SAMPLE OF MINI DATASET LABELS")
    print("="*60)
    
    # Load mini dataset
    with np.load(data_file) as f:
        keys = f['keys'][:]
        labels = f['label'][:].squeeze()  # Remove middle dimension if present
        
    print(f"Dataset shape: {labels.shape}")
    print(f"Total samples available: {len(keys)}")
    print()
    
    # Randomly sample indices
    np.random.seed(42)  # For reproducibility
    if num_samples > len(keys):
        num_samples = len(keys)
        
    random_indices = np.random.choice(len(keys), num_samples, replace=False)
    random_indices = np.sort(random_indices)  # Sort for easier reading
    
    histone_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    print(f"Randomly sampled {num_samples} labels:")
    print()
    print("Index  Key                     " + "  ".join([h[:8] for h in histone_names]))
    print("-" * (30 + 10 * len(histone_names)))
    
    for i, idx in enumerate(random_indices):
        key_str = str(keys[idx])[:25].ljust(25)
        label_str = "  ".join([str(int(labels[idx, j])) for j in range(min(7, labels.shape[1]))])
        print(f"{idx:3d}    {key_str} {label_str}")
    
    print()
    
    # Quick stats on the random sample
    print("STATISTICS OF RANDOM SAMPLE:")
    print("-" * 40)
    
    sample_labels = labels[random_indices]
    
    for j, histone in enumerate(histone_names[:min(7, labels.shape[1])]):
        pos_count = np.sum(sample_labels[:, j])
        pct = pos_count / num_samples * 100
        print(f"{histone:10s}: {pos_count:2d}/{num_samples} ({pct:5.1f}%)")
    
    print()
    
    # Check for patterns in this random sample
    print("PATTERN CHECK ON RANDOM SAMPLE:")
    print("-" * 40)
    
    # Samples with all marks
    all_marks = np.where(np.sum(sample_labels, axis=1) == 7)[0]
    print(f"Samples with ALL marks: {len(all_marks)}")
    if len(all_marks) > 0:
        for idx in all_marks:
            orig_idx = random_indices[idx]
            print(f"  Index {orig_idx}: {keys[orig_idx]}")
    
    # Samples with no marks
    no_marks = np.where(np.sum(sample_labels, axis=1) == 0)[0]
    print(f"Samples with NO marks: {len(no_marks)}")
    if len(no_marks) > 0:
        for idx in no_marks[:3]:  # Show first 3
            orig_idx = random_indices[idx]
            print(f"  Index {orig_idx}: {keys[orig_idx]}")
    
    # Check for identical patterns in this sample
    print(f"Samples with 1-6 marks: {num_samples - len(all_marks) - len(no_marks)}")
    
    print()
    
    # Check correlations in random sample
    print("CORRELATIONS IN RANDOM SAMPLE:")
    print("-" * 40)
    
    if len(sample_labels) > 1 and sample_labels.shape[1] >= 7:
        for i in range(7):
            for j in range(i+1, 7):
                # Only calculate correlation if there's variation
                if np.var(sample_labels[:, i]) > 0 and np.var(sample_labels[:, j]) > 0:
                    corr = np.corrcoef(sample_labels[:, i], sample_labels[:, j])[0, 1]
                    status = "❌" if abs(corr) > 0.95 else "✓"
                    print(f"{status} {histone_names[i][:8]} vs {histone_names[j][:8]}: {corr:.3f}")
                else:
                    print(f"? {histone_names[i][:8]} vs {histone_names[j][:8]}: no variation")
    
    return random_indices, sample_labels

def show_specific_indices(data_file, indices):
    """Show labels for specific indices"""
    
    print(f"\nSPECIFIC INDICES: {indices}")
    print("-" * 40)
    
    with np.load(data_file) as f:
        keys = f['keys'][:]
        labels = f['label'][:].squeeze()
        
    histone_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    print("Index  Key                     " + "  ".join([h[:8] for h in histone_names]))
    print("-" * (30 + 10 * len(histone_names)))
    
    for idx in indices:
        if idx < len(keys):
            key_str = str(keys[idx])[:25].ljust(25)
            label_str = "  ".join([str(int(labels[idx, j])) for j in range(min(7, labels.shape[1]))])
            print(f"{idx:3d}    {key_str} {label_str}")
        else:
            print(f"{idx:3d}    INDEX OUT OF RANGE")

if __name__ == "__main__":
    # Random sample from mini dataset
    data_file = 'data/converted/mini_merged.npz'
    
    print("1. RANDOM SAMPLE (20 samples):")
    random_indices, sample_labels = randomly_sample_mini_labels(data_file, 20)
    
    print("\n" + "="*60)
    print("2. FIRST 10 SAMPLES (for comparison):")
    show_specific_indices(data_file, list(range(10)))
    
    print("\n" + "="*60)
    print("3. LAST 10 SAMPLES (for comparison):")
    show_specific_indices(data_file, list(range(90, 100)))
    
    print("\n" + "="*60)
    print("4. MIDDLE SAMPLES (for comparison):")
    show_specific_indices(data_file, list(range(45, 55)))