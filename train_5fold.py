from model import DeepHistone
import copy
import numpy as np
from utils import metrics, model_train, model_eval, model_predict
import torch
import os
from sklearn.model_selection import KFold
import time

# Settings
batchsize = 20
data_file = 'data/converted/E003_all_markers_merged.npz'

# Extract epigenome name from data file
epigenome_name = os.path.basename(data_file).split('_')[0]  # Extracts 'E003' from filename
results_dir = f'results/{epigenome_name}_5fold_cv'
os.makedirs(results_dir, exist_ok=True)

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for data file: {data_file}")
print(f"File exists: {os.path.exists(data_file)}")
if os.path.exists(data_file):
    file_size_gb = os.path.getsize(data_file) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

print(f'Epigenome: {epigenome_name}')
print('Begin loading data...')
start_time = time.time()

with np.load(data_file) as f:
    print("File opened successfully!")
    print(f"Keys in file: {list(f.keys())}")
    
    indexs = f['keys']
    print(f"Loading {len(indexs)} samples...")
    
    print("Loading DNA sequences...")
    dna_dict = dict(zip(f['keys'], f['dna']))
    print("DNA sequences loaded!")
    
    print("Loading DNase data...")
    dns_dict = dict(zip(f['keys'], f['dnase']))
    print("DNase data loaded!")
    
    print("Loading labels...")
    lab_dict = dict(zip(f['keys'], f['label']))
    print("Labels loaded!")

load_time = time.time() - start_time
print(f"Data loading completed in {load_time:.2f} seconds")
print(f"Total samples: {len(indexs)}")

# Check data shapes
sample_key = list(dna_dict.keys())[0]
print(f"Sample data shapes:")
print(f"  DNA shape: {dna_dict[sample_key].shape}")
print(f"  DNase shape: {dns_dict[sample_key].shape}")
print(f"  Label shape: {lab_dict[sample_key].shape}")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Initialize 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
use_gpu = torch.cuda.is_available()

# Storage for results across all folds
all_fold_results = {
    'test_auPRC': [],
    'test_auROC': [],
    'test_labels': [],
    'test_predictions': []
}

# Perform 5-fold cross-validation
for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(indexs)):
    fold_start_time = time.time()
    print(f"\n{'='*50}")
    print(f"FOLD {fold_idx + 1}/5")
    print(f"{'='*50}")
    
    # Get indices for this fold
    train_val_indices = indexs[train_val_idx]
    test_indices = indexs[test_idx]
    
    # Further split train_val into train and validation (80% train, 20% validation)
    np.random.shuffle(train_val_indices)
    split_point = int(len(train_val_indices) * 0.8)
    train_indices = train_val_indices[:split_point]
    valid_indices = train_val_indices[split_point:]
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(valid_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Initialize model for this fold
    print("Initializing model...")
    model = DeepHistone(use_gpu)
    print("Model initialized successfully!")
    
    best_model = copy.deepcopy(model)
    best_valid_auPRC = 0
    best_valid_loss = float('inf')
    early_stop_time = 0
    
    print(f"Begin training model for fold {fold_idx + 1}...")
    
    # Training loop
    for epoch in range(50):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/50 ---")
        
        # Shuffle training data each epoch
        np.random.shuffle(train_indices)
        
        # Train
        print("Training...")
        train_loss = model_train(train_indices, model, batchsize, dna_dict, dns_dict, lab_dict)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        print("Validating...")
        valid_loss, valid_lab, valid_pred = model_eval(valid_indices, model, batchsize, dna_dict, dns_dict, lab_dict)
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, f'{epigenome_name}_Fold{fold_idx+1}_Valid_Epoch{epoch+1}', valid_loss)
        
        # Save best model based on validation auPRC
        mean_valid_auPRC = np.mean(list(valid_auPRC.values()))
        mean_valid_auROC = np.mean(list(valid_auROC.values()))
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Mean validation auPRC: {mean_valid_auPRC:.4f}")
        print(f"Mean validation auROC: {mean_valid_auROC:.4f}")
        
        if mean_valid_auPRC > best_valid_auPRC:
            best_valid_auPRC = mean_valid_auPRC
            best_model = copy.deepcopy(model)
            print(f"New best model saved! auPRC: {best_valid_auPRC:.4f}")
        
        # Early stopping based on validation loss
        if valid_loss < best_valid_loss:
            early_stop_time = 0
            best_valid_loss = valid_loss
        else:
            model.updateLR(0.1)
            early_stop_time += 1
            print(f"Early stopping counter: {early_stop_time}/5")
            if early_stop_time >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Test on the held-out fold
    print(f"\nBegin testing fold {fold_idx + 1}...")
    test_lab, test_pred = model_predict(test_indices, best_model, batchsize, dna_dict, dns_dict, lab_dict)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f'{epigenome_name}_Fold{fold_idx+1}_Test')
    
    # Store results for this fold
    all_fold_results['test_auPRC'].append(test_auPRC)
    all_fold_results['test_auROC'].append(test_auROC)
    all_fold_results['test_labels'].append(test_lab)
    all_fold_results['test_predictions'].append(test_pred)
    
    # Save fold-specific results
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"Saving fold {fold_idx + 1} results...")
    np.savetxt(os.path.join(fold_dir, f'{epigenome_name}_test_labels.txt'), test_lab, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(fold_dir, f'{epigenome_name}_test_predictions.txt'), test_pred, fmt='%.4f', delimiter='\t')
    best_model.save_model(os.path.join(fold_dir, f'{epigenome_name}_model.txt'))
    
    fold_time = time.time() - fold_start_time
    print(f"\nFold {fold_idx + 1} completed in {fold_time:.2f} seconds ({fold_time/60:.1f} minutes)!")
    print(f"Test auPRC: {test_auPRC}")
    print(f"Test auROC: {test_auROC}")

# Calculate and display overall results
print(f"\n{'='*60}")
print(f"{epigenome_name} 5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

# Get all histone marker names (assuming they're consistent across folds)
marker_names = list(all_fold_results['test_auPRC'][0].keys())
print(f"Histone markers: {marker_names}")

# Calculate mean and std for each marker
print("\nPer-Marker Results:")
print("-" * 50)
overall_auPRC_means = {}
overall_auROC_means = {}

for marker in marker_names:
    # Extract auPRC and auROC values for this marker across all folds
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

# Save as numpy file for easy loading later
np.savez(os.path.join(results_dir, f'{epigenome_name}_overall_results.npz'), **overall_results)

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

total_time = time.time() - start_time
print(f"\nResults saved to: {results_dir}/")
print("Files saved:")
print(f"- {epigenome_name}_overall_results.npz (numpy archive with all results)")
print(f"- {epigenome_name}_summary.txt (human-readable summary)")
print("- fold_X/ directories with individual fold results")
print(f"\nFinished {epigenome_name} 5-fold cross-validation!")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.1f} hours)")