# Refactored DeepHistone training with array slicing (no dictionaries)
import os
import time
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import KFold
from model import DeepHistone
from utils import metrics, model_train, model_eval, model_predict

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Settings
batchsize = 20
data_file = 'data/final/mini_merged.npz'
epigenome_name = os.path.basename(data_file).split('_')[0]
results_dir = f'results/{epigenome_name}_5fold_cv'
os.makedirs(results_dir, exist_ok=True)

# Load data
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for data file: {data_file}")
print(f"File exists: {os.path.exists(data_file)}")
if os.path.exists(data_file):
    file_size_gb = os.path.getsize(data_file) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

print(f"Epigenome: {epigenome_name}")
print("Begin loading data...")
start_time = time.time()

with np.load(data_file) as f:
    print("File opened successfully!")
    print(f"Keys in file: {list(f.keys())}")
    keys = f['keys']
    dna_data = f['dna']
    dnase_data = f['dnase']
    label_data = f['label']

print(f"Loaded {len(keys)} samples")
load_time = time.time() - start_time
print(f"Data loading completed in {load_time:.2f} seconds")

indices = np.arange(len(keys))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

all_fold_results = {
    'test_auPRC': [],
    'test_auROC': [],
    'best_scores': []
}

best_score = -1
best_fold_info = None

for fold_idx, (train_val_idx, test_idx) in tqdm(enumerate(kfold.split(indices), 1), total=5, desc="Cross-Validation Folds"):
    print(f"\n=== Fold {fold_idx}/5 ===")

    np.random.shuffle(train_val_idx)
    split_point = int(0.8 * len(train_val_idx))
    train_indices = train_val_idx[:split_point]
    valid_indices = train_val_idx[split_point:]

    fold_indices = np.concatenate([train_indices, valid_indices, test_idx])
    fold_dna_data = dna_data[fold_indices]
    fold_dnase_data = dnase_data[fold_indices]
    fold_label_data = label_data[fold_indices]

    train_idx = np.arange(len(train_indices))
    valid_idx = np.arange(len(train_indices), len(train_indices) + len(valid_indices))
    test_idx = np.arange(len(train_indices) + len(valid_indices), len(fold_indices))

    model = DeepHistone(use_gpu)
    best_model = copy.deepcopy(model)
    best_valid_auPRC = 0
    best_valid_loss = float('inf')
    early_stop_time = 0

    for epoch in tqdm(range(50), desc=f"Training Fold {fold_idx}", leave=False):
        train_loss = model_train(train_idx, model, batchsize, fold_dna_data, fold_dnase_data, fold_label_data)
        valid_loss, valid_lab, valid_pred = model_eval(valid_idx, model, batchsize, fold_dna_data, fold_dnase_data, fold_label_data)
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, f"{epigenome_name}_Fold{fold_idx}_Valid_Epoch{epoch+1}", valid_loss)

        mean_valid_auPRC = np.mean(list(valid_auPRC.values()))
        if mean_valid_auPRC > best_valid_auPRC:
            best_valid_auPRC = mean_valid_auPRC
            best_model = copy.deepcopy(model)
            early_stop_time = 0
            print(f"Epoch {epoch+1}: New best model! auPRC = {best_valid_auPRC:.4f}")
        else:
            early_stop_time += 1
            model.updateLR(0.1)
            print(f"Epoch {epoch+1}: no improvement (early_stop={early_stop_time})")
            if early_stop_time >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTesting fold {fold_idx}...")
    test_lab, test_pred = model_predict(test_idx, best_model, batchsize, fold_dna_data, fold_dnase_data, fold_label_data)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f"{epigenome_name}_Fold{fold_idx}_Test")

    all_fold_results['test_auPRC'].append(test_auPRC)
    all_fold_results['test_auROC'].append(test_auROC)
    all_fold_results['best_scores'].append((best_valid_auPRC, fold_idx, best_model, test_lab, test_pred))
    torch.cuda.empty_cache()

# Keep only best fold's model
best_valid_auPRC, best_idx, best_model, best_lab, best_pred = max(all_fold_results['best_scores'], key=lambda x: x[0])
best_fold_dir = os.path.join(results_dir, f"fold_{best_idx}_BEST")
os.makedirs(best_fold_dir, exist_ok=True)
np.savetxt(os.path.join(best_fold_dir, 'test_labels.txt'), best_lab, fmt='%d')
np.savetxt(os.path.join(best_fold_dir, 'test_predictions.txt'), best_pred, fmt='%.4f')
best_model.save_model(os.path.join(best_fold_dir, 'model.pt'))

print("\n========== Final Summary ==========")
marker_names = list(all_fold_results['test_auPRC'][0].keys())
for marker in tqdm(marker_names, desc="Calculating results"):
    auprcs = [fold[marker] for fold in all_fold_results['test_auPRC']]
    aurocs = [fold[marker] for fold in all_fold_results['test_auROC']]
    print(f"{marker}: auPRC = {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}, auROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
print("Done.")