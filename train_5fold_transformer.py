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
from utils_transformer import metrics, model_train, model_eval, model_predict, TensorBoardLogger
print(f"[IMPORT] utils_transformer functions loaded in {time.time() - t6:.2f} seconds")

t7 = time.time()
from datetime import datetime
print(f"[IMPORT] datetime loaded in {time.time() - t7:.2f} seconds")

print(f"[INFO] Total import time: {time.time() - t0:.2f} seconds")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Config
batchsize = 20
data_file = 'data/final/E005_chr1.npz'
epigenome = os.path.basename(data_file).split('_')[0]
results_dir = f'results/{epigenome}_5fold_transformer_chr1'
os.makedirs(results_dir, exist_ok=True)

# TensorBoard setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_dir = f'runs/{epigenome}_5fold_{timestamp}'
os.makedirs(tensorboard_dir, exist_ok=True)
print(f"[INFO] TensorBoard logs will be saved to: {tensorboard_dir}")
print(f"[INFO] To view logs, run: tensorboard --logdir={tensorboard_dir}")

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
    "batch_size": batchsize,
    "learning_rate": 1e-3,
    "max_epochs": 50,
    "early_stopping_patience": 5,
    "data_file": data_file,
    "epigenome": epigenome
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Initialize summary TensorBoard logger for cross-fold comparisons
summary_logger = TensorBoardLogger(
    log_dir=os.path.join(tensorboard_dir, "summary"),
    model_config=model_config
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_auPRC, all_auROC = [], []
all_train_loss, all_val_loss = [], []
fold_loggers = []

print("[DEBUG] model_transformer loaded from:", NetDeepHistoneTransformer.__module__)

# Store all fold results for final summary
fold_results = {
    'fold_aurocs': [],
    'fold_auprcs': [],
    'fold_train_losses': [],
    'fold_val_losses': []
}

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(keys)):
    print(f"\n{'='*30}\n Fold {fold_idx+1}/5 \n{'='*30}")

    # Initialize fold-specific TensorBoard logger
    fold_logger = TensorBoardLogger(
        log_dir=os.path.join(tensorboard_dir, f"fold_{fold_idx+1}"),
        model_config=model_config
    )
    fold_loggers.append(fold_logger)

    # Split train/val
    np.random.shuffle(train_val_idx)
    split = int(0.8 * len(train_val_idx))
    train_idx = train_val_idx[:split]
    val_idx = train_val_idx[split:]
    test_idx = test_idx

    train_keys = [keys[i] for i in train_idx]
    val_keys = [keys[i] for i in val_idx]
    test_keys = [keys[i] for i in test_idx]

    print(f"[INFO] Fold {fold_idx+1} - Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")

    # Initialize model
    print("Initializing model...")
    model = NetDeepHistoneTransformer(model_config).to(device)
    print(f"Model initialized on device: {device}")
    
    # Log model graph (only for first fold to avoid duplicates)
    if fold_idx == 0:
        print("[INFO] Skipping model graph logging")
        print("[INFO] All other TensorBoard logging (metrics, attention, curves) will work normally")
        best_model_state = None
        best_val_auprc = 0
        early_stop_counter = 0
        max_patience = 5

    fold_train_losses = []
    fold_val_losses = []

    print(f"[INFO] Starting training for fold {fold_idx+1}...")
    
    for epoch in range(50):
        print(f"\n--- Fold {fold_idx+1}, Epoch {epoch+1}/50 ---")
        
        np.random.shuffle(train_keys)
        
        # Training with TensorBoard logging
        train_loss = model_train(
            train_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device,
            tb_logger=fold_logger, epoch=epoch, fold=fold_idx+1
        )
        
        # Validation with TensorBoard logging
        val_loss, val_lab, val_pred = model_eval(
            val_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device,
            tb_logger=fold_logger, epoch=epoch, fold=fold_idx+1
        )
        
        # Calculate validation metrics
        val_auPRC, val_auROC = metrics(val_lab, val_pred, Type=f"valid_fold{fold_idx+1}_epoch{epoch+1}", loss=val_loss)
        mean_val_auprc = np.mean(list(val_auPRC.values()))
        mean_val_auroc = np.mean(list(val_auROC.values()))

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val auPRC: {mean_val_auprc:.4f} | Val auROC: {mean_val_auroc:.4f}")

        # Log to summary logger for cross-fold comparison
        summary_logger.log_scalars({
            f'fold_{fold_idx+1}/train_loss': train_loss,
            f'fold_{fold_idx+1}/val_loss': val_loss,
            f'fold_{fold_idx+1}/val_mean_auprc': mean_val_auprc,
            f'fold_{fold_idx+1}/val_mean_auroc': mean_val_auroc
        }, epoch)

        # Early stopping logic
        if mean_val_auprc > best_val_auprc:
            best_val_auprc = mean_val_auprc
            best_model_state = model.state_dict()
            early_stop_counter = 0
            print(f"[INFO] New best validation auPRC: {best_val_auprc:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= max_patience:
                print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
                break

    # Store fold training history
    all_train_loss.append(fold_train_losses)
    all_val_loss.append(fold_val_losses)
    
    # Log training curves for this fold
    fold_logger.log_training_curves(fold_train_losses, fold_val_losses, fold_idx+1)

    # Load best model for test evaluation
    print("Loading best model for test evaluation...")
    model.load_state_dict(best_model_state)

    # Test evaluation with TensorBoard logging
    test_lab, test_pred = model_predict(
        test_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device,
        tb_logger=fold_logger, fold=fold_idx+1
    )
    
    # Log final training and validation histograms and confusion matrices
    print(f"[INFO] Logging final histograms and confusion matrices for fold {fold_idx+1}...")
    
    # Get final validation predictions for comprehensive logging
    final_val_loss, final_val_lab, final_val_pred = model_eval(
        val_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device
    )
    
    # Get final training predictions for comprehensive logging  
    final_train_lab, final_train_pred = model_predict(
        train_keys, model, batchsize, dna_dict, dnase_dict, label_dict, device
    )
    
    # Log final histograms and confusion matrices for all sets
    if len(final_train_pred) > 0:
        fold_logger.log_histograms(final_train_pred, final_train_lab, 0, prefix=f"fold{fold_idx+1}/train_final")
        fold_logger.log_confusion_matrices(final_train_pred, final_train_lab, 0, prefix=f"fold{fold_idx+1}/train_final")
    
    if len(final_val_pred) > 0:
        fold_logger.log_histograms(final_val_pred, final_val_lab, 0, prefix=f"fold{fold_idx+1}/val_final")
        fold_logger.log_confusion_matrices(final_val_pred, final_val_lab, 0, prefix=f"fold{fold_idx+1}/val_final")
    
    # Test histograms and confusion matrices are already logged in model_predict
    
    # Calculate test metrics
    test_auPRC, test_auROC = metrics(test_lab, test_pred, Type=f"test_fold{fold_idx+1}")

    all_auPRC.append(test_auPRC)
    all_auROC.append(test_auROC)
    
    # Store results for summary
    mean_test_auprc = np.mean(list(test_auPRC.values()))
    mean_test_auroc = np.mean(list(test_auROC.values()))
    
    fold_results['fold_aurocs'].append(mean_test_auroc)
    fold_results['fold_auprcs'].append(mean_test_auprc)
    fold_results['fold_train_losses'].append(fold_train_losses)
    fold_results['fold_val_losses'].append(fold_val_losses)

    # Log final test results to summary
    summary_logger.log_scalars({
        f'fold_{fold_idx+1}/test_mean_auprc': mean_test_auprc,
        f'fold_{fold_idx+1}/test_mean_auroc': mean_test_auroc
    }, 0)  # Use step 0 for final test results

    # Save fold results
    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_labels.txt'), test_lab, fmt='%d')
    np.savetxt(os.path.join(results_dir, f'fold{fold_idx+1}_preds.txt'), test_pred, fmt='%.4f')
    
    print(f"[INFO] Fold {fold_idx+1} completed - Test auPRC: {mean_test_auprc:.4f}, Test auROC: {mean_test_auroc:.4f}")

# Final cross-fold analysis and logging
print(f"\n{'='*60}\n5-Fold Cross-Validation Results for {epigenome}\n{'='*60}")

# Calculate statistics across folds
fold_auprc_mean = np.mean(fold_results['fold_auprcs'])
fold_auprc_std = np.std(fold_results['fold_auprcs'])
fold_auroc_mean = np.mean(fold_results['fold_aurocs'])
fold_auroc_std = np.std(fold_results['fold_aurocs'])

print(f"Cross-fold auPRC: {fold_auprc_mean:.4f} ± {fold_auprc_std:.4f}")
print(f"Cross-fold auROC: {fold_auroc_mean:.4f} ± {fold_auroc_std:.4f}")

for i in range(5):
    print(f"Fold {i+1} | auPRC: {fold_results['fold_auprcs'][i]:.4f} | auROC: {fold_results['fold_aurocs'][i]:.4f}")

# Log final summary statistics
summary_logger.log_scalars({
    'cross_validation/mean_auprc': fold_auprc_mean,
    'cross_validation/std_auprc': fold_auprc_std,
    'cross_validation/mean_auroc': fold_auroc_mean,
    'cross_validation/std_auroc': fold_auroc_std
}, 0)

# Create comprehensive fold comparison plots
print("\n[INFO] Creating cross-fold comparison visualizations...")

# Training curves comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))

# Plot training curves for all folds
plt.subplot(2, 2, 1)
for i, (train_losses, val_losses) in enumerate(zip(fold_results['fold_train_losses'], fold_results['fold_val_losses'])):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label=f'Fold {i+1} Train', alpha=0.7)
    plt.plot(epochs, val_losses, label=f'Fold {i+1} Val', alpha=0.7, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves Across All Folds')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# auPRC across folds
plt.subplot(2, 2, 2)
plt.bar(range(1, 6), fold_results['fold_auprcs'], alpha=0.7, color='skyblue')
plt.axhline(y=fold_auprc_mean, color='red', linestyle='--', label=f'Mean: {fold_auprc_mean:.4f}')
plt.xlabel('Fold')
plt.ylabel('Test auPRC')
plt.title('Test auPRC Across Folds')
plt.legend()
plt.grid(True, alpha=0.3)

# auROC across folds
plt.subplot(2, 2, 3)
plt.bar(range(1, 6), fold_results['fold_aurocs'], alpha=0.7, color='lightcoral')
plt.axhline(y=fold_auroc_mean, color='red', linestyle='--', label=f'Mean: {fold_auroc_mean:.4f}')
plt.xlabel('Fold')
plt.ylabel('Test auROC')
plt.title('Test auROC Across Folds')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot of performance distribution
plt.subplot(2, 2, 4)
performance_data = [fold_results['fold_auprcs'], fold_results['fold_aurocs']]
plt.boxplot(performance_data, labels=['auPRC', 'auROC'])
plt.ylabel('Score')
plt.title('Performance Distribution Across Folds')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
comparison_plot_path = os.path.join(results_dir, 'cross_fold_comparison.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"[INFO] Cross-fold comparison plot saved to: {comparison_plot_path}")

# Convert and log to TensorBoard
import io
from PIL import Image
import torchvision.transforms as transforms

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
image = Image.open(buf)
transform = transforms.ToTensor()
image_tensor = transform(image)

summary_logger.writer.add_image('cross_validation/fold_comparison', image_tensor, 0)
plt.close()
buf.close()

# Calculate and log detailed per-histone statistics
histone_stats = {}
for histone_idx, histone in enumerate(['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']):
    histone_auprcs = [all_auPRC[fold][histone] for fold in range(5)]
    histone_aurocs = [all_auROC[fold][histone] for fold in range(5)]
    
    histone_stats[histone] = {
        'auprc_mean': np.mean(histone_auprcs),
        'auprc_std': np.std(histone_auprcs),
        'auroc_mean': np.mean(histone_aurocs),
        'auroc_std': np.std(histone_aurocs)
    }
    
    # Log per-histone statistics
    summary_logger.log_scalars({
        f'per_histone/{histone}/auprc_mean': histone_stats[histone]['auprc_mean'],
        f'per_histone/{histone}/auprc_std': histone_stats[histone]['auprc_std'],
        f'per_histone/{histone}/auroc_mean': histone_stats[histone]['auroc_mean'],
        f'per_histone/{histone}/auroc_std': histone_stats[histone]['auroc_std']
    }, 0)

print("\nPer-histone performance summary:")
for histone, stats in histone_stats.items():
    print(f"{histone}: auPRC {stats['auprc_mean']:.4f}±{stats['auprc_std']:.4f}, auROC {stats['auroc_mean']:.4f}±{stats['auroc_std']:.4f}")

# Update hyperparameters with final results
final_metrics = {
    'final_mean_auprc': fold_auprc_mean,
    'final_mean_auroc': fold_auroc_mean,
    'final_std_auprc': fold_auprc_std,
    'final_std_auroc': fold_auroc_std
}
summary_logger.update_hparams_metrics(final_metrics)

# Save all results
np.save(os.path.join(results_dir, "train_loss_per_fold.npy"), np.array(all_train_loss, dtype=object))
np.save(os.path.join(results_dir, "val_loss_per_fold.npy"), np.array(all_val_loss, dtype=object))
np.save(os.path.join(results_dir, "fold_results.npy"), fold_results)

# Close all TensorBoard loggers
for logger in fold_loggers:
    logger.close()
summary_logger.close()

print(f"\n[INFO] Experiment completed successfully!")
print(f"[INFO] Results saved to: {results_dir}")
print(f"[INFO] TensorBoard logs saved to: {tensorboard_dir}")
print(f"[INFO] To view TensorBoard, run: tensorboard --logdir={tensorboard_dir}")
print(f"[INFO] Final performance: auPRC {fold_auprc_mean:.4f}±{fold_auprc_std:.4f}, auROC {fold_auroc_mean:.4f}±{fold_auroc_std:.4f}")