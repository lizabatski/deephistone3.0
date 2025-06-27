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
print(f"All imports done in {time.time() - start:.2f} seconds")

print("GOT HERE")

# --- Configuration ---
epigenome = 'E005'
results_dir = f'results/{epigenome}_5fold_cv'
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

print("All visualizations completed successfully.")
