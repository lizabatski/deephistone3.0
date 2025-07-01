# %%
# import
print("GOT HERE 1")
import time
start = time.time()
print("Before imports...")
import os
print("os imported")
import numpy as np
print("numpy imported")
import matplotlib.pyplot as plt
print("matplotlib imported")
from sklearn.metrics import precision_recall_curve, average_precision_score
print("sklearn metrics imported")
import argparse
print("argparse imported")
print(f"All imports done in {time.time() - start:.2f} seconds")

print("GOT HERE")

# %% the rest

# --- Configuration ---
epigenome = 'E005'

parser = argparse.ArgumentParser(description='Visualize DeepHistone results.')
parser.add_argument('--epigenome', type=str, default='E005', help='Epigenome ID (e.g., E005)')
parser.add_argument('--results_dir', type=str, help='Path to results directory')

args = parser.parse_args()
epigenome = args.epigenome
results_dir = args.results_dir if args.results_dir else f'results/{epigenome}_5fold_cv'

num_folds = 5
epigenome = 'E005'
marker_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K9ac']

# --- Storage for all folds ---
all_labels = []
all_preds = []

print("Loading test label and prediction files from each fold...")

# --- Load all folds ---
for fold in range(1, num_folds + 1):
    fold_dir = os.path.join(results_dir, f'fold_{fold}')
    label_file = os.path.join(fold_dir, f'{epigenome}_test_labels.txt')
    pred_file = os.path.join(fold_dir, f'{epigenome}_test_predictions.txt')
    
    print(f"  Fold {fold}: loading labels from {label_file}")
    y_true = np.loadtxt(label_file, dtype=int)
    print(f"  Fold {fold}: loading predictions from {pred_file}")
    y_pred = np.loadtxt(pred_file)
    
    all_labels.append(y_true)
    all_preds.append(y_pred)

print("All folds loaded. Concatenating data...")

# Concatenate all folds
labels = np.vstack(all_labels)
preds = np.vstack(all_preds)
print(f"Labels shape: {labels.shape}, Predictions shape: {preds.shape}")

# --- PR Curves ---
print("Generating Precision-Recall curves...")
plt.figure(figsize=(12, 8))
for i, marker in enumerate(marker_names):
    y_true = labels[:, i]
    y_scores = preds[:, i]
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    print(f"  Marker {marker}: AP = {ap:.4f}")
    plt.plot(recall, precision, label=f'{marker} (AP={ap:.3f})')
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (All Markers)')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
prc_path = f'{results_dir}/{epigenome}_PR_Curves.png'
plt.savefig(prc_path)
print(f"Precision-Recall plot saved to {prc_path}")
plt.show()

# --- Label Distribution ---
print("Plotting label distribution...")
plt.figure(figsize=(10, 5))
pos_counts = np.sum(labels, axis=0)
neg_counts = labels.shape[0] - pos_counts

x = np.arange(len(marker_names))
plt.bar(x - 0.2, pos_counts, width=0.4, label='Positive')
plt.bar(x + 0.2, neg_counts, width=0.4, label='Negative')
plt.xticks(x, marker_names, rotation=45)
plt.ylabel('Sample Count')
plt.title('Label Distribution per Histone Marker')
plt.legend()
plt.tight_layout()
label_dist_path = f'{results_dir}/{epigenome}_Label_Distribution.png'
plt.savefig(label_dist_path)
print(f"Label distribution plot saved to {label_dist_path}")
plt.show()

# --- Prediction Score Histograms ---
print("Generating prediction score histograms...")
plt.figure(figsize=(14, 8))
for i, marker in enumerate(marker_names):
    plt.subplot(2, 4, i+1)
    plt.hist(preds[:, i], bins=50, color='skyblue', edgecolor='k')
    plt.title(marker)
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
plt.suptitle('Prediction Score Distributions')
plt.tight_layout(rect=[0, 0, 1, 0.95])
hist_path = f'{results_dir}/{epigenome}_Prediction_Histograms.png'
plt.savefig(hist_path)
print(f"Prediction score histogram saved to {hist_path}")
plt.show()


# --- Training Dynamics Plots ---
print("Generating training loss and validation curves...")

for fold in range(1, num_folds + 1):
    history_path = os.path.join(results_dir, f'fold_{fold}', f'{epigenome}_fold{fold}_history.npy')
    if not os.path.exists(history_path):
        print(f"  Fold {fold}: history file not found, skipping.")
        continue

    history = np.load(history_path, allow_pickle=True).item()
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Dynamics - Fold {fold}', fontsize=16)

    initial_train_loss = history['train_loss'][0]
    train_loss_pct = [(loss / initial_train_loss) * 100 for loss in history['train_loss']]
    axs[0, 0].plot(epochs, train_loss_pct, label='Train Loss (% of initial)', color='blue')
    axs[0, 0].set_ylabel('Loss (% of initial)')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, history['val_auPRC'], label='Val auPRC', color='green')
    axs[1, 0].set_title('Validation auPRC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('auPRC')
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, history['val_auROC'], label='Val auROC', color='purple')
    axs[1, 1].set_title('Validation auROC')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('auROC')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    curve_path = os.path.join(results_dir, f'{epigenome}_Fold{fold}_Training_Curves.png')
    plt.savefig(curve_path)
    print(f"  Fold {fold}: training curve saved to {curve_path}")
    plt.show()


print("All visualizations completed successfully.")
