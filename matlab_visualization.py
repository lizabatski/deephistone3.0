import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
results_dir = 'results/E005)5fold_transformer_chr1'
histone_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

print("Loading saved results...")

# Load comprehensive results
fold_results = np.load(os.path.join(results_dir, 'fold_results.npy'), allow_pickle=True).item()
train_losses = np.load(os.path.join(results_dir, 'train_loss_per_fold.npy'), allow_pickle=True)
val_losses = np.load(os.path.join(results_dir, 'val_loss_per_fold.npy'), allow_pickle=True)

print(f"Loaded results for {len(fold_results['fold_auprcs'])} folds")

# ============================================================================
# 1. TRAINING CURVES ACROSS ALL FOLDS
# ============================================================================
def plot_training_curves():
    plt.figure(figsize=(15, 10))
    
    # Plot 1: All training curves
    plt.subplot(2, 2, 1)
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f'Fold {i+1} Train', alpha=0.8, linewidth=2)
        plt.plot(epochs, val_loss, label=f'Fold {i+1} Val', alpha=0.8, linewidth=2, linestyle='--')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curves Across All Folds', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average training curve
    plt.subplot(2, 2, 2)
    max_epochs = max(len(tl) for tl in train_losses)
    avg_train_loss = []
    avg_val_loss = []
    std_train_loss = []
    std_val_loss = []
    
    for epoch in range(max_epochs):
        train_at_epoch = [tl[epoch] for tl in train_losses if len(tl) > epoch]
        val_at_epoch = [vl[epoch] for vl in val_losses if len(vl) > epoch]
        
        if train_at_epoch:
            avg_train_loss.append(np.mean(train_at_epoch))
            std_train_loss.append(np.std(train_at_epoch))
        if val_at_epoch:
            avg_val_loss.append(np.mean(val_at_epoch))
            std_val_loss.append(np.std(val_at_epoch))
    
    epochs = range(1, len(avg_train_loss) + 1)
    plt.plot(epochs, avg_train_loss, 'b-', label='Avg Train Loss', linewidth=3)
    plt.fill_between(epochs, 
                     np.array(avg_train_loss) - np.array(std_train_loss),
                     np.array(avg_train_loss) + np.array(std_train_loss),
                     alpha=0.2, color='blue')
    
    epochs_val = range(1, len(avg_val_loss) + 1)
    plt.plot(epochs_val, avg_val_loss, 'r-', label='Avg Val Loss', linewidth=3)
    plt.fill_between(epochs_val,
                     np.array(avg_val_loss) - np.array(std_val_loss),
                     np.array(avg_val_loss) + np.array(std_val_loss),
                     alpha=0.2, color='red')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Average Training Curves with Std Dev', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Performance across folds
    plt.subplot(2, 2, 3)
    folds = range(1, 6)
    x_pos = np.arange(len(folds))
    
    plt.bar(x_pos - 0.2, fold_results['fold_auprcs'], 0.4, label='auPRC', alpha=0.8)
    plt.bar(x_pos + 0.2, fold_results['fold_aurocs'], 0.4, label='auROC', alpha=0.8)
    
    plt.axhline(y=np.mean(fold_results['fold_auprcs']), color='blue', linestyle='--', 
                label=f"Mean auPRC: {np.mean(fold_results['fold_auprcs']):.4f}")
    plt.axhline(y=np.mean(fold_results['fold_aurocs']), color='orange', linestyle='--',
                label=f"Mean auROC: {np.mean(fold_results['fold_aurocs']):.4f}")
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Across Folds', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'Fold {i}' for i in folds])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Training length distribution
    plt.subplot(2, 2, 4)
    training_lengths = [len(tl) for tl in train_losses]
    plt.bar(range(1, 6), training_lengths, alpha=0.8, color='green')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Number of Epochs', fontsize=12)
    plt.title('Training Length per Fold (Early Stopping)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, length in enumerate(training_lengths):
        plt.text(i+1, length + 0.5, str(length), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_training_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 2. DETAILED PERFORMANCE ANALYSIS
# ============================================================================
def plot_detailed_performance():
    # Load individual fold predictions
    all_labels = []
    all_preds = []
    
    for fold in range(1, 6):
        labels = np.loadtxt(os.path.join(results_dir, f'fold{fold}_labels.txt'))
        preds = np.loadtxt(os.path.join(results_dir, f'fold{fold}_preds.txt'))
        all_labels.append(labels)
        all_preds.append(preds)
    
    # Combine all folds
    combined_labels = np.vstack(all_labels)
    combined_preds = np.vstack(all_preds)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot ROC curves for each histone
    for i, histone in enumerate(histone_names):
        ax = axes[i]
        
        # Plot ROC curve for each fold
        for fold in range(5):
            fpr, tpr, _ = roc_curve(all_labels[fold][:, i], all_preds[fold][:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.6, linewidth=1, 
                   label=f'Fold {fold+1} (AUC = {roc_auc:.3f})')
        
        # Plot combined ROC
        fpr_combined, tpr_combined, _ = roc_curve(combined_labels[:, i], combined_preds[:, i])
        roc_auc_combined = auc(fpr_combined, tpr_combined)
        ax.plot(fpr_combined, tpr_combined, 'k-', linewidth=3,
               label=f'Combined (AUC = {roc_auc_combined:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {histone}')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Summary statistics in the last subplot
    ax = axes[-2]
    performance_summary = []
    for i, histone in enumerate(histone_names):
        fold_aucs = []
        for fold in range(5):
            fpr, tpr, _ = roc_curve(all_labels[fold][:, i], all_preds[fold][:, i])
            fold_aucs.append(auc(fpr, tpr))
        performance_summary.append(fold_aucs)
    
    performance_summary = np.array(performance_summary)
    im = ax.imshow(performance_summary, aspect='auto', cmap='RdYlBu_r')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(histone_names)
    ax.set_title('AUROC Heatmap Across Folds')
    
    # Add text annotations
    for i in range(7):
        for j in range(5):
            text = ax.text(j, i, f'{performance_summary[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Remove the last empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'detailed_roc_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 3. CONFUSION MATRICES
# ============================================================================
def plot_confusion_matrices():
    # Load predictions for best fold (highest auPRC)
    best_fold_idx = np.argmax(fold_results['fold_auprcs'])
    best_fold = best_fold_idx + 1
    
    labels = np.loadtxt(os.path.join(results_dir, f'fold{best_fold}_labels.txt'))
    preds = np.loadtxt(os.path.join(results_dir, f'fold{best_fold}_preds.txt'))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, histone in enumerate(histone_names):
        # Convert predictions to binary (threshold = 0.5)
        pred_binary = (preds[:, i] > 0.5).astype(int)
        true_binary = labels[:, i].astype(int)
        
        # Create confusion matrix
        cm = confusion_matrix(true_binary, pred_binary)
        
        # Plot
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title(f'{histone} - Fold {best_fold}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle(f'Confusion Matrices - Best Performing Fold ({best_fold})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 4. PREDICTION SCORE DISTRIBUTIONS
# ============================================================================
def plot_prediction_distributions():
    # Load all predictions
    all_labels = []
    all_preds = []
    
    for fold in range(1, 6):
        labels = np.loadtxt(os.path.join(results_dir, f'fold{fold}_labels.txt'))
        preds = np.loadtxt(os.path.join(results_dir, f'fold{fold}_preds.txt'))
        all_labels.append(labels)
        all_preds.append(preds)
    
    # Combine all folds
    combined_labels = np.vstack(all_labels)
    combined_preds = np.vstack(all_preds)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, histone in enumerate(histone_names):
        ax = axes[i]
        
        # Separate predictions by true label
        positive_preds = combined_preds[combined_labels[:, i] == 1, i]
        negative_preds = combined_preds[combined_labels[:, i] == 0, i]
        
        # Plot histograms
        ax.hist(negative_preds, bins=50, alpha=0.7, label='True Negative', 
               color='red', density=True)
        ax.hist(positive_preds, bins=50, alpha=0.7, label='True Positive', 
               color='blue', density=True)
        
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
                  label='Threshold (0.5)')
        
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{histone} - Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle('Prediction Score Distributions Across All Folds', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================
def print_summary_statistics():
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nCross-Validation Performance:")
    print(f"Mean auPRC: {np.mean(fold_results['fold_auprcs']):.4f} ± {np.std(fold_results['fold_auprcs']):.4f}")
    print(f"Mean auROC: {np.mean(fold_results['fold_aurocs']):.4f} ± {np.std(fold_results['fold_aurocs']):.4f}")
    
    print(f"\nPer-Fold Results:")
    for i in range(5):
        print(f"Fold {i+1}: auPRC = {fold_results['fold_auprcs'][i]:.4f}, auROC = {fold_results['fold_aurocs'][i]:.4f}, Epochs = {len(train_losses[i])}")
    
    print(f"\nTraining Statistics:")
    training_lengths = [len(tl) for tl in train_losses]
    print(f"Average training length: {np.mean(training_lengths):.1f} ± {np.std(training_lengths):.1f} epochs")
    print(f"Early stopping triggered: {sum(1 for tl in training_lengths if len(tl) < 50)}/5 folds")
    
    print(f"\nFiles Generated:")
    print(f"- comprehensive_training_analysis.png")
    print(f"- detailed_roc_analysis.png") 
    print(f"- confusion_matrices.png")
    print(f"- prediction_distributions.png")
    print(f"- cross_fold_comparison.png (already existed)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("Creating comprehensive visualization plots...")
    
    # Generate all plots
    plot_training_curves()
    plot_detailed_performance() 
    plot_confusion_matrices()
    plot_prediction_distributions()
    
    # Print summary
    print_summary_statistics()
    
    print(f"\nAll plots saved to: {results_dir}/")
    print("✅ Complete visualization analysis finished!")