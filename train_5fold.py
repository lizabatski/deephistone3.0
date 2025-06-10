from model import DeepHistone
import copy
import numpy as np
from utils import metrics, model_train, model_eval, model_predict
import torch
import os
from sklearn.model_selection import KFold

# Settings
batchsize = 20
data_file = 'data/converted/E003_all_markers_merged.npz'
results_dir = 'results/5fold_cv'
os.makedirs(results_dir, exist_ok=True)

print('Begin loading data...')
with np.load(data_file) as f:
    indexs = f['keys']
    dna_dict = dict(zip(f['keys'], f['dna']))
    dns_dict = dict(zip(f['keys'], f['dnase']))
    lab_dict = dict(zip(f['keys'], f['label']))

print(f"Total samples: {len(indexs)}")

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
    model = DeepHistone(use_gpu)
    best_model = copy.deepcopy(model)
    best_valid_auPRC = 0
    best_valid_loss = float('inf')
    early_stop_time = 0
    
    print(f"Begin training model for fold {fold_idx + 1}...")
    
    # Training loop
    for epoch in range(50):
        # Shuffle training data each epoch
        np.random.shuffle(train_indices)
        
        # Train
        train_loss = model_train(train_indices, model, batchsize, dna_dict, dns_dict, lab_dict)
        
        # Validate
        valid_loss, valid_lab, valid_pred = model_eval(valid_indices, model, batchsize, dna_dict, dns_dict, lab_dict)
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, f'Fold{fold_idx+1}_Valid_Epoch{epoch+1}', valid_loss)
        
        # Save best model based on validation auPRC
        mean_valid_auPRC = np.mean(list(valid_auPRC.values()))
        if mean_valid_auPRC > best_valid_auPRC:
            best_valid_auPRC = mean_valid_auPRC
            best_model = copy.deepcopy(model)
        
        # Early stopping based on validation loss
        if valid_loss < best_valid_loss:
            early_stop_time = 0
            best_valid_loss = valid_loss
        else:
            model.updateLR(0.1)
            early_stop_time += 1
            if early_stop_time >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Test on the held-out fold
    print(f"Begin testing fold {fold_idx + 1}...")
    test_lab, test_pred = model_predict(test_indices, best_model, batchsize, dna_dict, dns_dict, lab_dict)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f'Fold{fold_idx+1}_Test')
    
    # Store results for this fold
    all_fold_results['test_auPRC'].append(test_auPRC)
    all_fold_results['test_auROC'].append(test_auROC)
    all_fold_results['test_labels'].append(test_lab)
    all_fold_results['test_predictions'].append(test_pred)
    
    # Save fold-specific results
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    np.savetxt(os.path.join(fold_dir, 'test_labels.txt'), test_lab, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(fold_dir, 'test_predictions.txt'), test_pred, fmt='%.4f', delimiter='\t')
    best_model.save_model(os.path.join(fold_dir, 'model.txt'))
    
    print(f"Fold {fold_idx + 1} completed!")
    print(f"Test auPRC: {test_auPRC}")
    print(f"Test auROC: {test_auROC}")

# Calculate and display overall results
print(f"\n{'='*60}")
print("5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

# Get all histone marker names (assuming they're consistent across folds)
marker_names = list(all_fold_results['test_auPRC'][0].keys())

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
overall_results = {
    'per_marker_auPRC': overall_auPRC_means,
    'per_marker_auROC': overall_auROC_means,
    'overall_mean_auPRC': overall_mean_auPRC,
    'overall_mean_auROC': overall_mean_auROC,
    'all_fold_auPRC': all_fold_results['test_auPRC'],
    'all_fold_auROC': all_fold_results['test_auROC']
}

# Save as numpy file for easy loading later
np.savez(os.path.join(results_dir, 'overall_results.npz'), **overall_results)

# Save summary as text file
with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
    f.write("5-Fold Cross-Validation Results\n")
    f.write("=" * 40 + "\n\n")
    
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

print(f"\nResults saved to: {results_dir}/")
print("Files saved:")
print("- overall_results.npz (numpy archive with all results)")
print("- summary.txt (human-readable summary)")
print("- fold_X/ directories with individual fold results")
print("\nFinished 5-fold cross-validation!")