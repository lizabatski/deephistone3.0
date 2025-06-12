from model import DeepHistone
import copy
import numpy as np
from utils import metrics, model_train, model_eval, model_predict
import torch
import os
from sklearn.model_selection import KFold
import time
import random
from tqdm import tqdm

use_gpu = torch.cuda.is_available()

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Settings
batchsize = 20
data_file = 'data/final/E005_all_markers_merged.npz'

# Extract epigenome name from data file
epigenome_name = os.path.basename(data_file).split('_')[0]
results_dir = f'results/{epigenome_name}_5fold_cv'
os.makedirs(results_dir, exist_ok=True)

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for data file: {data_file}")
print(f"File exists: {os.path.exists(data_file)}")

print(f'Epigenome: {epigenome_name}')
print('Begin loading data...')
start_time = time.time()

with np.load(data_file) as f:
    print("File opened successfully!")
    print(f"Keys in file: {list(f.keys())}")
    
    # doad data as arrays
    keys = f['keys'][:]  # Shape: (1423776,)
    dna_data = f['dna'][:]  # Shape: (1423776, seq_len, 4)
    dnase_data = f['dnase'][:]  # Shape: (1423776, seq_len)  
    label_data = f['label'][:]  # Shape: (1423776, 1, 7)
    
    print(f"Data loaded successfully!")
    print(f"Keys shape: {keys.shape}")
    print(f"DNA shape: {dna_data.shape}")
    print(f"DNase shape: {dnase_data.shape}")
    print(f"Label shape: {label_data.shape}")

load_time = time.time() - start_time
print(f"Data loading completed in {load_time:.2f} seconds")
print(f"Total samples: {len(keys)}")

# Memory usage estimate
total_memory_gb = (dna_data.nbytes + dnase_data.nbytes + label_data.nbytes) / 1e9
print(f"Estimated memory usage: {total_memory_gb:.2f} GB")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")

# Create index array for cross-validation
indices = np.arange(len(keys))

# Initialize 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if use_gpu else "cpu")

# Storage for results across all folds
all_fold_results = {
    'test_auPRC': [],
    'test_auROC': [],
    'test_labels': [],
    'test_predictions': []
}

def create_subset_dicts(subset_indices, keys, dna_data, dnase_data, label_data):
    """Create dictionaries for a subset of data (much smaller than full dataset)"""
    print("Creating subset dictionaries...")
    subset_keys = keys[subset_indices]
    
    # Use tqdm for dictionary creation progress
    dna_dict = {}
    dnase_dict = {}
    label_dict = {}
    
    for i, key in enumerate(tqdm(subset_keys, desc="Building dictionaries", leave=False)):
        dna_dict[key] = dna_data[subset_indices[i]]
        dnase_dict[key] = dnase_data[subset_indices[i]]
        label_dict[key] = label_data[subset_indices[i]]
    
    return subset_keys, dna_dict, dnase_dict, label_dict

# Perform 5-fold cross-validation with progress bar
fold_progress = tqdm(enumerate(kfold.split(indices)), total=5, desc="Cross-Validation Folds", position=0)

for fold_idx, (train_val_idx, test_idx) in fold_progress:
    fold_start_time = time.time()
    fold_progress.set_description(f"Fold {fold_idx + 1}/5")
    
    # Get indices for this fold
    train_val_indices = indices[train_val_idx]
    test_indices = indices[test_idx]
    
    # Further split train_val into train and validation (80% train, 20% validation)
    np.random.shuffle(train_val_indices)
    split_point = int(len(train_val_indices) * 0.8)
    train_indices = train_val_indices[:split_point]
    valid_indices = train_val_indices[split_point:]
    
    tqdm.write(f"\n{'='*50}")
    tqdm.write(f"FOLD {fold_idx + 1}/5")
    tqdm.write(f"{'='*50}")
    tqdm.write(f"Train samples: {len(train_indices)}")
    tqdm.write(f"Validation samples: {len(valid_indices)}")
    tqdm.write(f"Test samples: {len(test_indices)}")
    
    # Create dictionaries for this fold only (much smaller than full dataset)
    fold_indices = np.concatenate([train_indices, valid_indices, test_indices])
    fold_keys, fold_dna_dict, fold_dnase_dict, fold_label_dict = create_subset_dicts(
        fold_indices, keys, dna_data, dnase_data, label_data
    )
    
    # Convert indices to keys for this fold
    train_keys = keys[train_indices]
    valid_keys = keys[valid_indices]
    test_keys = keys[test_indices]
    
    tqdm.write(f"Fold dictionaries created with {len(fold_keys)} samples")
    
    # Initialize model for this fold
    tqdm.write("Initializing model...")
    model = DeepHistone(use_gpu)
    tqdm.write("Model initialized successfully!")
    
    best_model = copy.deepcopy(model)
    best_valid_auPRC = 0
    best_valid_loss = float('inf')
    early_stop_time = 0
    
    tqdm.write(f"Begin training model for fold {fold_idx + 1}...")
    
    # Training loop with progress bar
    epoch_progress = tqdm(range(50), desc=f"Training Fold {fold_idx + 1}", position=1, leave=False)
    
    for epoch in epoch_progress:
        epoch_start_time = time.time()
        
        # Shuffle training data each epoch
        np.random.shuffle(train_keys)
        
        # Train using original utils functions
        train_loss = model_train(train_keys, model, batchsize, fold_dna_dict, fold_dnase_dict, fold_label_dict, device)
        
        # Validate using original utils functions
        valid_loss, valid_lab, valid_pred = model_eval(valid_keys, model, batchsize, fold_dna_dict, fold_dnase_dict, fold_label_dict, device)
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, f'{epigenome_name}_Fold{fold_idx+1}_Valid_Epoch{epoch+1}', valid_loss)
        
        # Save best model based on validation auPRC
        mean_valid_auPRC = np.mean(list(valid_auPRC.values()))
        mean_valid_auROC = np.mean(list(valid_auROC.values()))
        
        epoch_time = time.time() - epoch_start_time
        
        # Update progress bar with current metrics
        epoch_progress.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{valid_loss:.4f}',
            'val_auPRC': f'{mean_valid_auPRC:.4f}',
            'val_auROC': f'{mean_valid_auROC:.4f}',
            'early_stop': f'{early_stop_time}/5'
        })
        
        if mean_valid_auPRC > best_valid_auPRC:
            best_valid_auPRC = mean_valid_auPRC
            best_model = copy.deepcopy(model)
            tqdm.write(f"Epoch {epoch + 1}: New best model! auPRC: {best_valid_auPRC:.4f}")
        
        # Early stopping based on validation loss
        if valid_loss < best_valid_loss:
            early_stop_time = 0
            best_valid_loss = valid_loss
        else:
            model.updateLR(0.1)
            early_stop_time += 1
            if early_stop_time >= 5:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break
    
    epoch_progress.close()
    
    # Test on the held-out fold
    tqdm.write(f"\nTesting fold {fold_idx + 1}...")
    
    # Add progress bar for testing
    with tqdm(desc=f"Testing Fold {fold_idx + 1}", position=1, leave=False) as test_pbar:
        test_lab, test_pred = model_predict(test_keys, best_model, batchsize, fold_dna_dict, fold_dnase_dict, fold_label_dict)
        test_pbar.update(1)
    
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f'{epigenome_name}_Fold{fold_idx+1}_Test')
    
    # Store results for this fold
    all_fold_results['test_auPRC'].append(test_auPRC)
    all_fold_results['test_auROC'].append(test_auROC)
    all_fold_results['test_labels'].append(test_lab)
    all_fold_results['test_predictions'].append(test_pred)
    
    # Save fold-specific results
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    tqdm.write(f"Saving fold {fold_idx + 1} results...")
    with tqdm(desc="Saving results", position=1, leave=False) as save_pbar:
        np.savetxt(os.path.join(fold_dir, f'{epigenome_name}_test_labels.txt'), test_lab, fmt='%d', delimiter='\t')
        save_pbar.update(1)
        np.savetxt(os.path.join(fold_dir, f'{epigenome_name}_test_predictions.txt'), test_pred, fmt='%.4f', delimiter='\t')
        save_pbar.update(1)
        best_model.save_model(os.path.join(fold_dir, f'{epigenome_name}_model.txt'))
        save_pbar.update(1)
    
    fold_time = time.time() - fold_start_time
    tqdm.write(f"\nFold {fold_idx + 1} completed in {fold_time:.2f} seconds ({fold_time/60:.1f} minutes)!")
    tqdm.write(f"Test auPRC: {test_auPRC}")
    tqdm.write(f"Test auROC: {test_auROC}")
    
    # Clean up fold dictionaries to save memory
    del fold_dna_dict, fold_dnase_dict, fold_label_dict

fold_progress.close()

# Calculate and display overall results
print(f"\n{'='*60}")
print(f"{epigenome_name} 5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

# Get all histone marker names
marker_names = list(all_fold_results['test_auPRC'][0].keys())
print(f"Histone markers: {marker_names}")

# Calculate mean and std for each marker
print("\nPer-Marker Results:")
print("-" * 50)
overall_auPRC_means = {}
overall_auROC_means = {}

# Add progress bar for results calculation
for marker in tqdm(marker_names, desc="Calculating results"):
    marker_auPRC = [fold_result[marker] for fold_result in all_fold_results['test_auPRC']]
    marker_auROC = [fold_result[marker] for fold_result in all_fold_results['test_auROC']]
    
    mean_auPRC = np.mean(marker_auPRC)
    std_auPRC = np.std(marker_auPRC)
    mean_auROC = np.mean(marker_auROC)
    std_auROC = np.std(marker_auROC)
    
    overall_auPRC_means[marker] = mean_auPRC
    overall_auROC_means[marker] = mean_auROC
    
    print(f"{marker}:")
    print(f"  auPRC: {mean_auPRC:.4f} ± {std_auPRC:.4f}")
    print(f"  auROC: {mean_auROC:.4f} ± {std_auROC:.4f}")

# Overall average across all markers
overall_mean_auPRC = np.mean(list(overall_auPRC_means.values()))
overall_mean_auROC = np.mean(list(overall_auROC_means.values()))

print(f"\nOverall Average (across all markers):")
print(f"auPRC: {overall_mean_auPRC:.4f}")
print(f"auROC: {overall_mean_auROC:.4f}")

# Save overall results
print(f"\nSaving overall results...")
overall_results = {
    'epigenome': epigenome_name,
    'per_marker_auPRC': overall_auPRC_means,
    'per_marker_auROC': overall_auROC_means,
    'overall_mean_auPRC': overall_mean_auPRC,
    'overall_mean_auROC': overall_mean_auROC,
    'all_fold_auPRC': all_fold_results['test_auPRC'],
    'all_fold_auROC': all_fold_results['test_auROC']
}

with tqdm(desc="Saving files", total=3) as save_pbar:
    np.savez(os.path.join(results_dir, f'{epigenome_name}_overall_results.npz'), **overall_results)
    save_pbar.update(1)
    
    # Save summary as text file
    with open(os.path.join(results_dir, f'{epigenome_name}_summary.txt'), 'w') as f:
        f.write(f"{epigenome_name} 5-Fold Cross-Validation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Per-Marker Results:\n")
        f.write("-" * 20 + "\n")
        for marker in marker_names:
            marker_auPRC = [fold_result[marker] for fold_result in all_fold_results['test_auPRC']]
            marker_auROC = [fold_result[marker] for fold_result in all_fold_results['test_auROC']]
            f.write(f"{marker}:\n")
            f.write(f"  auPRC: {np.mean(marker_auPRC):.4f} ± {np.std(marker_auPRC):.4f}\n")
            f.write(f"  auROC: {np.mean(marker_auROC):.4f} ± {np.std(marker_auROC):.4f}\n")
        
        f.write(f"\nOverall Average:\n")
        f.write(f"auPRC: {overall_mean_auPRC:.4f}\n")
        f.write(f"auROC: {overall_mean_auROC:.4f}\n")
    save_pbar.update(1)
    
    save_pbar.set_description("Files saved successfully")
    save_pbar.update(1)

total_time = time.time() - start_time
print(f"\nResults saved to: {results_dir}/")
print("Files saved:")
print(f"- {epigenome_name}_overall_results.npz (numpy archive with all results)")
print(f"- {epigenome_name}_summary.txt (human-readable summary)")
print("- fold_X/ directories with individual fold results")
print(f"\nFinished {epigenome_name} 5-fold cross-validation!")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.1f} hours)")