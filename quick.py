import torch
import numpy as np
import copy
from model import DeepHistone
from utils import metrics, model_train, model_eval
import time

# Settings
data_file = 'data/converted/E003_all_markers_merged.npz'  # adjust path if needed
batchsize = 8  # small batch size for quick test

print('Loading dataset...')
start = time.time()

# Load entire dataset into memory
with np.load(data_file, mmap_mode=None) as f:
    keys = f['keys']
    dna_all = f['dna']
    dnase_all = f['dnase']
    label_all = f['label']

    print("Loaded full arrays:")
    print("dna_all shape:", dna_all.shape)       # e.g., (N, 4, 1000)
    print("dnase_all shape:", dnase_all.shape)   # e.g., (N, 1, 1000)
    print("label_all shape:", label_all.shape)   # e.g., (N, 7)
    print("keys shape:", keys.shape)             # (N,)

    # Inspect one example entry
    example_index = 0
    print("\nExample entry shapes:")
    print("dna_all[0] shape:", dna_all[example_index].shape)
    print("dnase_all[0] shape:", dnase_all[example_index].shape)
    print("label_all[0] shape:", label_all[example_index].shape)
    print("label_all[0] =", label_all[example_index])

subset_keys = keys[:100]  # small test batch

# Use in-memory arrays to build dictionaries
dna_dict = {k: dna_all[i] for i, k in enumerate(subset_keys)}
dnase_dict = {k: dnase_all[i] for i, k in enumerate(subset_keys)}
label_dict = {k: label_all[i] for i, k in enumerate(subset_keys)}

print("Time to build dicts:", time.time() - start, "seconds")

# Shuffle and split a small sample
np.random.shuffle(subset_keys)
train_keys = subset_keys[:20]
valid_keys = subset_keys[20:30]

# Check for GPU availability
use_gpu = torch.cuda.is_available()
model = DeepHistone(use_gpu)

print('Running quick training and evaluation...')
train_loss = model_train(train_keys, model, batchsize, dna_dict, dnase_dict, label_dict)
valid_loss, valid_lab, valid_pred = model_eval(valid_keys, model, batchsize, dna_dict, dnase_dict, label_dict)
valid_auprc, valid_auroc = metrics(valid_lab, valid_pred, split='Valid', loss=valid_loss)

print('Quick test completed.')
print(f'Validation loss: {valid_loss:.4f}')
print(f'Validation auPRC: {valid_auprc}')
print(f'Validation auROC: {valid_auroc}')
