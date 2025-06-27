import time

t0 = time.time()
import os
print(f"[IMPORT] os loaded in {time.time() - t0:.2f} seconds")

t1 = time.time()
import torch
print(f"[IMPORT] torch loaded in {time.time() - t1:.2f} seconds")

t2 = time.time()
import numpy as np
print(f"[IMPORT] numpy loaded in {time.time() - t2:.2f} seconds")

t3 = time.time()
from sklearn.model_selection import KFold
print(f"[IMPORT] sklearn.model_selection.KFold loaded in {time.time() - t3:.2f} seconds")

t4 = time.time()
from tqdm import tqdm
print(f"[IMPORT] tqdm loaded in {time.time() - t4:.2f} seconds")

t5 = time.time()
from model_transformer import NetDeepHistoneTransformer
print(f"[IMPORT] NetDeepHistoneTransformer loaded in {time.time() - t5:.2f} seconds")

t6 = time.time()
from utils_transformer import metrics, model_train, model_eval, model_predict
print(f"[IMPORT] utils_transformer functions loaded in {time.time() - t6:.2f} seconds")

print(f"[INFO] Total import time: {time.time() - t0:.2f} seconds")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Config
batchsize = 20
data_file = 'data/final/E005_all_markers_merged.npz'
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

# Build dictionaries for sample access
print("Indexing sample dictionaries...")
dna_dict = {}
dnase_dict = {}
label_dict = {}
for i in tqdm(range(len(keys)), desc="Indexing"):
    key = keys[i]
    dna_dict[key] = np.expand_dims(dna[i], axis=0)  # (1, 1000, 4)
    dnase_dict[key] = np.expand_dims(dnase[i], axis=0)  # (1, 1000, 1)
    label_dict[key] = np.expand_dims(labels[i], axis=0)  # (1, 7)

# Model configuration
model_config = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "pooling": "mean",
    "seq_len": 1000,
    "embedding_type": "linear",
    "use_transformer_seq": True,
    "use_transformer_dnase": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_auPRC, all_auROC = [], []

print("[DEBUG] model_transformer loaded from:", NetDeepHistoneTransformer.__module__)

all_train_loss, all_val_loss = [], []

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(keys)):
    print(f"\n{'='*30}\n Fold {fold_idx+1}/5 \n{'='*30}")

    # Split train/val
    np.random.shuffle(train_val_idx)
    split = int(0.8 * len(train_val_idx))
    train_idx = train_val_idx[:split]
    val_idx = train_val_idx[split:]
    test_idx = test_idx

    train_keys = [keys[i] for i in train_idx]
    val_keys = [keys[i] for i in val_idx]
    test_keys = [keys[i] for i in test_idx]

    # Initialize model
    print("Initializing model...")
    model = NetDeepHistoneTransformer(model_config).to(device)
    print(f"Model initialized on device: {device}")

    best_model_state = None
    best_val_auprc = 0
    early_stop_counter = 0
    max_patience = 5

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(50):
        np.random.shuffle(train_keys)
        train_loss = model_train(train_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device)
        val_loss, val_lab, val_pred = model_eval(val_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device)
        val_auPRC, _ = metrics(val_lab, val_pred, Type=f"valid_fold{fold_idx+1}_epoch{epoch+1}", loss=val_loss)
        mean_val_auprc = np.mean(list(val_auPRC.values()))

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val auPRC: {mean_val_auprc:.4f}")

        if mean_val_auprc > best_val_auprc:
            best_val_auprc = mean_val_auprc
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= max_patience:
                print("Early stopping triggered.")
                break

    all_train_loss.append(fold_train_losses)
    all_val_loss.append(fold_val_losses)

    # Load best model
    print("Loading best model for test evaluation...")
    model.load_state_dict(best_model_state)

    test_lab, test_pred = model_predict(test_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device)
    test_auPRC, test_auROC = metrics(test_lab, test_pred, Type=f"test_fold{fold_idx+1}")

    all_auPRC.append(test_auPRC)
    all_auROC.append(test_auROC)

    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_labels.txt'), test_lab, fmt='%d')
    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_preds.txt'), test_pred, fmt='%.4f')

# Final report
print(f"\n{'='*50}\n5-Fold Results for {epigenome}\n{'='*50}")
for i in range(5):
    print(f"Fold {i+1} | auPRC: {np.mean(list(all_auPRC[i].values())):.4f} | auROC: {np.mean(list(all_auROC[i].values())):.4f}")

avg_auPRC = np.mean([np.mean(list(fold.values())) for fold in all_auPRC])
avg_auROC = np.mean([np.mean(list(fold.values())) for fold in all_auROC])
print(f"\nOverall Avg auPRC: {avg_auPRC:.4f}")
print(f"Overall Avg auROC: {avg_auROC:.4f}")

np.save(os.path.join(results_dir, "train_loss_per_fold.npy"), np.array(all_train_loss, dtype=object))
np.save(os.path.join(results_dir, "val_loss_per_fold.npy"), np.array(all_val_loss, dtype=object))
print(f"[INFO] Saved loss histories to: {results_dir}")