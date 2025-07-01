# train_single_marker.py

import time
import argparse
import copy
import os
import numpy as np
import torch
import random
from datetime import datetime
from sklearn.model_selection import KFold
from tqdm import tqdm

from model_single_marker import DeepHistone
from utils import metrics, model_train, model_eval, model_predict, HistoneDataset

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--marker_index', type=int, required=True, help="Histone marker index (0-6)")
parser.add_argument('--output_dir', type=str, default=None, help="Path to save results")
args = parser.parse_args()

# Marker names for reference
marker_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
marker_name = marker_names[args.marker_index]

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Settings
data_file = 'data/final/E005_chr1.npz'
batchsize = 20

# Setup output directory
epigenome_name = os.path.basename(data_file).split('_')[0]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = args.output_dir if args.output_dir else f'results/{epigenome_name}_{marker_name}_cv_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

# Load data
with np.load(data_file) as f:
    keys = f['keys'][:]
    dna_data = f['dna'][:]
    dnase_data = f['dnase'][:]
    label_data = f['label'][:, args.marker_index:args.marker_index+1]  # Only one marker

# Device setup
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Index array for k-fold
indices = np.arange(len(keys))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

all_fold_results = {'test_auPRC': [], 'test_auROC': [], 'test_labels': [], 'test_predictions': []}

def create_subset_dicts(subset_indices):
    subset_keys = keys[subset_indices]
    dna_dict = {k: dna_data[i] for i, k in enumerate(subset_keys)}
    dnase_dict = {k: dnase_data[i] for i, k in enumerate(subset_keys)}
    label_dict = {k: label_data[i] for i, k in enumerate(subset_keys)}
    return subset_keys, dna_dict, dnase_dict, label_dict

for fold_idx, (train_val_idx, test_idx) in tqdm(enumerate(kfold.split(indices)), total=5, desc="Cross-Validation"):
    train_val_indices = indices[train_val_idx]
    test_indices = indices[test_idx]
    split = int(len(train_val_indices) * 0.8)
    train_indices = train_val_indices[:split]
    valid_indices = train_val_indices[split:]

    fold_keys, dna_dict, dnase_dict, label_dict = create_subset_dicts(np.concatenate([train_indices, valid_indices, test_indices]))

    train_dataset = HistoneDataset(keys[train_indices], dna_dict, dnase_dict, label_dict)
    valid_dataset = HistoneDataset(keys[valid_indices], dna_dict, dnase_dict, label_dict)
    test_dataset  = HistoneDataset(keys[test_indices],  dna_dict, dnase_dict, label_dict)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batchsize)

    model = DeepHistone(use_gpu)
    optimizer = model.optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_model = copy.deepcopy(model)
    best_val_auPRC = 0
    best_val_loss = float('inf')
    early_stop_time = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auPRC': [],
        'val_auROC': []
    }

    for epoch in tqdm(range(50), desc=f"Fold {fold_idx+1} Training", leave=False):
        train_loss = model_train(train_loader, model)
        val_loss, val_lab, val_pred = model_eval(valid_loader, model)
        val_auPRC, val_auROC = metrics(val_lab, val_pred, f"Fold{fold_idx+1}_Epoch{epoch+1}", val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auPRC'].append(val_auPRC)
        history['val_auROC'].append(val_auROC)

        if val_auPRC > best_val_auPRC:
            best_val_auPRC = val_auPRC
            best_model = copy.deepcopy(model)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            early_stop_time = 0
            best_val_loss = val_loss
        else:
            early_stop_time += 1
            if early_stop_time >= 10:
                break

    test_lab, test_pred = model_predict(test_loader, best_model)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f"Fold{fold_idx+1}_Test")

    all_fold_results['test_auPRC'].append(test_auPRC)
    all_fold_results['test_auROC'].append(test_auROC)
    all_fold_results['test_labels'].append(test_lab)
    all_fold_results['test_predictions'].append(test_pred)

    fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    np.savetxt(os.path.join(fold_dir, f'{marker_name}_test_labels.txt'), test_lab, fmt='%d')
    np.savetxt(os.path.join(fold_dir, f'{marker_name}_test_predictions.txt'), test_pred, fmt='%.4f')
    best_model.save_model(os.path.join(fold_dir, f'{marker_name}_model.pt'))
    np.save(os.path.join(fold_dir, f'{marker_name}_history.npy'), history)

# Final summary
mean_auPRC = np.mean(all_fold_results['test_auPRC'])
mean_auROC = np.mean(all_fold_results['test_auROC'])
print(f"\n{marker_name} 5-fold Results:\nauPRC: {mean_auPRC:.4f}\nauROC: {mean_auROC:.4f}")
np.savez(os.path.join(results_dir, f'{marker_name}_overall_results.npz'), **all_fold_results)
