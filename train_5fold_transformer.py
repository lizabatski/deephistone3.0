import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from model_transformer import NetDeepHistoneTransformer
from utils import metrics, model_train, model_eval, model_predict

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Config
batchsize = 20
data_file = 'data/final/mini_merged.npz'
epigenome = os.path.basename(data_file).split('_')[0]
results_dir = f'results/{epigenome}_5fold_transformer'
os.makedirs(results_dir, exist_ok=True)

# Load data
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for data file: {data_file}")
print(f"File exists: {os.path.exists(data_file)}")

if os.path.exists(data_file):
    file_size_gb = os.path.getsize(data_file) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

print(f"Epigenome: {epigenome}")
print("Begin loading data...")

with np.load(data_file) as f:
    print("File opened successfully!")
    print(f"Keys in file: {list(f.keys())}")

    keys = f['keys'][:]
    dna = f['dna'][:]
    dnase = f['dnase'][:]
    labels = f['label'][:]

    print(f"Data loaded successfully!")
    print(f"Keys shape: {keys.shape}")
    print(f"DNA shape: {dna.shape}")
    print(f"DNase shape: {dnase.shape}")
    print(f"Label shape: {labels.shape}")

# Helper: build sample dictionaries for subset indices
def build_sample_dict(indices):
    print("Building sample dictionaries...")
    sample_dict = {}
    for i in tqdm(indices, desc="Building dict", leave=False):
        sample_dict[keys[i]] = {
            "dna": dna[i],
            "dnase": dnase[i],
            "label": labels[i]
        }
    return sample_dict

# Default transformer config
model_config = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "pooling": "mean",
    "seq_len": 1000,
    "embedding_type": "linear"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_auPRC, all_auROC = [], []

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(keys)):
    print(f"\n{'='*30}\n Fold {fold_idx+1}/5 \n{'='*30}")

    np.random.shuffle(train_val_idx)
    split = int(0.8 * len(train_val_idx))
    train_idx, val_idx = train_val_idx[:split], train_val_idx[split:]

    train_dict = build_sample_dict(train_idx)
    val_dict = build_sample_dict(val_idx)
    test_dict = build_sample_dict(test_idx)

    train_keys = [keys[i] for i in train_idx]
    val_keys = [keys[i] for i in val_idx]
    test_keys = [keys[i] for i in test_idx]

    # Model
    print("Initializing model...")
    model = NetDeepHistoneTransformer(model_config, use_gpu=torch.cuda.is_available())
    print("Model initialized successfully!")
    best_model = model
    best_auprc = 0
    early_stop = 0
    max_patience = 5

    for epoch in range(50):
        np.random.shuffle(train_keys)
        train_loss = model_train(train_keys, model, batchsize, train_dict, device)
        val_loss, val_lab, val_pred = model_eval(val_keys, model, batchsize, val_dict, device)
        val_auPRC, _ = metrics(val_lab, val_pred, f'{epigenome}_fold{fold_idx+1}_val_epoch{epoch+1}', val_loss)

        mean_val_auPRC = np.mean(list(val_auPRC.values()))
        print(f"Epoch {epoch+1} | train loss: {train_loss:.4f} | val auPRC: {mean_val_auPRC:.4f}")

        if mean_val_auPRC > best_auprc:
            best_auprc = mean_val_auPRC
            best_model = model
            early_stop = 0
        else:
            model.updateLR(0.1)
            early_stop += 1
            if early_stop >= max_patience:
                print("Early stopping.")
                break

    # Final test
    print("Testing best model on test fold...")
    test_lab, test_pred = model_predict(test_keys, best_model, batchsize, test_dict)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, f'{epigenome}_fold{fold_idx+1}_test')

    all_auPRC.append(test_auPRC)
    all_auROC.append(test_auROC)

    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_labels.txt'), test_lab, fmt='%d')
    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_preds.txt'), test_pred, fmt='%.4f')

# Summary
print(f"\n{'='*50}\n5-Fold Results for {epigenome}\n{'='*50}")
for i in range(5):
    print(f"Fold {i+1} | auPRC: {np.mean(list(all_auPRC[i].values())):.4f} | auROC: {np.mean(list(all_auROC[i].values())):.4f}")

avg_auPRC = np.mean([np.mean(list(x.values())) for x in all_auPRC])
avg_auROC = np.mean([np.mean(list(x.values())) for x in all_auROC])
print(f"\nOverall Avg auPRC: {avg_auPRC:.4f}")
print(f"Overall Avg auROC: {avg_auROC:.4f}")
