from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image

histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']


class TensorBoardLogger:
    """Enhanced TensorBoard logging for comprehensive experiment tracking"""
    
    def __init__(self, log_dir, model_config=None):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
        # Log hyperparameters
        if model_config:
            self.writer.add_hparams(
                hparam_dict=model_config,
                metric_dict={}  # Will be updated with final metrics
            )
    
    def log_model_graph(self, model, seq_input, dns_input):
        """Log model architecture graph"""
        try:
            self.writer.add_graph(model, (seq_input, dns_input))
        except Exception as e:
            print(f"[WARNING] Could not log model graph: {e}")
    
    def log_scalars(self, metrics_dict, global_step, prefix=""):
        """Log scalar metrics"""
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, global_step)
    
    def log_histograms(self, predictions, labels, global_step, prefix=""):
        """Log prediction score histograms"""
        for i, histone in enumerate(histones):
            # Prediction histograms
            self.writer.add_histogram(
                f"{prefix}/predictions_{histone}" if prefix else f"predictions_{histone}",
                predictions[:, i], global_step
            )
            
            # Separate histograms for positive and negative samples
            pos_mask = labels[:, i] == 1
            neg_mask = labels[:, i] == 0
            
            if pos_mask.sum() > 0:
                self.writer.add_histogram(
                    f"{prefix}/predictions_{histone}_positive" if prefix else f"predictions_{histone}_positive",
                    predictions[pos_mask, i], global_step
                )
            
            if neg_mask.sum() > 0:
                self.writer.add_histogram(
                    f"{prefix}/predictions_{histone}_negative" if prefix else f"predictions_{histone}_negative",
                    predictions[neg_mask, i], global_step
                )
    
    def log_confusion_matrices(self, predictions, labels, global_step, prefix="", threshold=0.5):
        """Log confusion matrices as images"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, histone in enumerate(histones):
            pred_binary = (predictions[:, i] > threshold).astype(int)
            true_binary = labels[:, i].astype(int)
            
            cm = confusion_matrix(true_binary, pred_binary)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{histone} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Remove the last empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        # Convert plot to image and log
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        
        # Convert PIL image to tensor
        import torchvision.transforms as transforms
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        self.writer.add_image(
            f"{prefix}/confusion_matrices" if prefix else "confusion_matrices",
            image_tensor, global_step
        )
        
        plt.close()
        buf.close()
    
    def log_attention_weights(self, attention_weights, global_step, layer_idx=0, head_idx=0, prefix=""):
        """Log attention weight heatmaps for specified layer and head"""
        if attention_weights is None:
            return
        
        # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
        if len(attention_weights.shape) == 4:
            # Take first sample in batch and specified head
            attn_matrix = attention_weights[0, head_idx].detach().cpu().numpy()
        else:
            attn_matrix = attention_weights.detach().cpu().numpy()
        
        # Create attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_matrix, cmap='viridis', cbar=True)
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Convert to image tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        
        import torchvision.transforms as transforms
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        self.writer.add_image(
            f"{prefix}/attention_layer{layer_idx}_head{head_idx}" if prefix else f"attention_layer{layer_idx}_head{head_idx}",
            image_tensor, global_step
        )
        
        plt.close()
        buf.close()
    
    def log_training_curves(self, train_losses, val_losses, fold_idx, prefix=""):
        """Log training curves for a specific fold"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Curves - Fold {fold_idx}')
        ax.legend()
        ax.grid(True)
        
        # Convert to image tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        
        import torchvision.transforms as transforms
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        self.writer.add_image(
            f"{prefix}/training_curves_fold{fold_idx}" if prefix else f"training_curves_fold{fold_idx}",
            image_tensor, 0  # Use 0 as step since this is a summary image
        )
        
        plt.close()
        buf.close()
    
    def update_hparams_metrics(self, final_metrics):
        """Update hyperparameter experiment with final metrics"""
        # This will show final results in the HPARAMS tab
        self.writer.add_hparams({}, final_metrics)
    
    def close(self):
        """Close the writer"""
        self.writer.close()


def loadRegions(regions_indices, dna_dict, dns_dict, label_dict):
    dna_regions = np.concatenate([dna_dict[meta] for meta in regions_indices], axis=0) if dna_dict is not None else []
    dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indices], axis=0) if dns_dict is not None else []
    label_regions = np.concatenate([label_dict[meta] for meta in regions_indices], axis=0).astype(int)
    return dna_regions, dns_regions, label_regions


def model_train(regions, model, batchsize, dna_dict, dns_dict, label_dict, device, tb_logger=None, epoch=0, fold=0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    train_loss = []
    
    all_preds = []
    all_labels = []

    for i in range(0, len(regions), batchsize):
        batch = regions[i:i+batchsize]
        if len(batch) < 2:
            print(f"[INFO] Skipping training batch with size {len(batch)} at index {i}")
            continue

        seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)

        seq_tensor = torch.tensor(seq_batch, dtype=torch.float32).to(device)
        if seq_tensor.dim() == 4 and seq_tensor.shape[1] == 1:
            seq_tensor = seq_tensor.squeeze(1)

        dns_tensor = torch.tensor(dns_batch, dtype=torch.float32).to(device)
        if dns_tensor.dim() == 4 and dns_tensor.shape[1] == 1:
            dns_tensor = dns_tensor.squeeze(1)

        label_tensor = torch.tensor(lab_batch, dtype=torch.float32).to(device)
        if label_tensor.dim() == 3 and label_tensor.shape[1] == 1:
            label_tensor = label_tensor.squeeze(1)

        print(f"[DEBUG][train] seq_tensor: {seq_tensor.shape}, dns_tensor: {dns_tensor.shape}, label_tensor: {label_tensor.shape}")

        optimizer.zero_grad()
        
        # Log attention weights for first batch of first epoch
        save_attention = (tb_logger is not None and i == 0 and epoch == 0)
        
        if hasattr(model, 'forward') and 'save_attention' in model.forward.__code__.co_varnames:
            pred = model(seq_tensor, dns_tensor, save_attention=save_attention, sample_name=f"fold{fold}_epoch{epoch}_batch{i}")
        else:
            pred = model(seq_tensor, dns_tensor)
        
        # Log attention weights if model supports it
        if tb_logger and save_attention and hasattr(model, 'seq_map') and hasattr(model.seq_map, 'encoder_layers'):
            try:
                # Get attention weights from first layer, first head
                first_layer = model.seq_map.encoder_layers[0]
                if hasattr(first_layer, 'attention_weights') and first_layer.attention_weights is not None:
                    tb_logger.log_attention_weights(
                        first_layer.attention_weights, 
                        epoch,
                        layer_idx=0, 
                        head_idx=0, 
                        prefix=f"fold{fold}/train"
                    )
            except Exception as e:
                print(f"[WARNING] Could not log attention weights: {e}")
        
        loss = criterion(pred, label_tensor)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        # Collect predictions and labels for metrics
        all_preds.extend(pred.detach().cpu().numpy())
        all_labels.extend(label_tensor.detach().cpu().numpy())

    # Log training metrics if logger provided
    if tb_logger and all_preds:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate and log AUROC/AUPRC for each histone
        for i, histone in enumerate(histones):
            if len(np.unique(all_labels[:, i])) > 1:  # Check if both classes present
                auroc = ROC(all_labels[:, i], all_preds[:, i])
                auprc = auPR(all_labels[:, i], all_preds[:, i])
                
                tb_logger.log_scalars({
                    f'fold{fold}/train/auroc_{histone}': auroc,
                    f'fold{fold}/train/auprc_{histone}': auprc
                }, epoch)
        
        # Log mean metrics
        mean_auroc = np.mean([ROC(all_labels[:, i], all_preds[:, i]) for i in range(len(histones)) if len(np.unique(all_labels[:, i])) > 1])
        mean_auprc = np.mean([auPR(all_labels[:, i], all_preds[:, i]) for i in range(len(histones)) if len(np.unique(all_labels[:, i])) > 1])
        
        tb_logger.log_scalars({
            f'fold{fold}/train/mean_auroc': mean_auroc,
            f'fold{fold}/train/mean_auprc': mean_auprc,
            f'fold{fold}/train/loss': np.mean(train_loss)
        }, epoch)
        
        # Histograms will be logged at the end of training

    return np.mean(train_loss) if train_loss else 0.0


def model_eval(regions, model, batchsize, dna_dict, dns_dict, label_dict, device, tb_logger=None, epoch=0, fold=0):
    model.eval()
    criterion = nn.BCELoss()
    losses = []
    preds = []
    labels = []

    with torch.no_grad():
        for i in range(0, len(regions), batchsize):
            batch = regions[i:i+batchsize]
            if len(batch) < 2:
                print(f"[INFO] Skipping validation batch with size {len(batch)} at index {i}")
                continue

            seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)

            seq_tensor = torch.tensor(seq_batch, dtype=torch.float32).to(device)
            if seq_tensor.dim() == 4 and seq_tensor.shape[1] == 1:
                seq_tensor = seq_tensor.squeeze(1)

            dns_tensor = torch.tensor(dns_batch, dtype=torch.float32).to(device)
            if dns_tensor.dim() == 4 and dns_tensor.shape[1] == 1:
                dns_tensor = dns_tensor.squeeze(1)

            label_tensor = torch.tensor(lab_batch, dtype=torch.float32).to(device)
            if label_tensor.dim() == 3 and label_tensor.shape[1] == 1:
                label_tensor = label_tensor.squeeze(1)

            print(f"[DEBUG][eval] seq_tensor: {seq_tensor.shape}, dns_tensor: {dns_tensor.shape}, label_tensor: {label_tensor.shape}")

            pred = model(seq_tensor, dns_tensor)
            loss = criterion(pred, label_tensor)

            losses.append(loss.item())
            labels.extend(label_tensor.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    if not losses:
        return (0.0, np.zeros((0, 7)), np.zeros((0, 7)))
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    # Log validation metrics if logger provided
    if tb_logger:
        # Calculate and log AUROC/AUPRC for each histone
        for i, histone in enumerate(histones):
            if len(np.unique(labels[:, i])) > 1:  # Check if both classes present
                auroc = ROC(labels[:, i], preds[:, i])
                auprc = auPR(labels[:, i], preds[:, i])
                
                tb_logger.log_scalars({
                    f'fold{fold}/val/auroc_{histone}': auroc,
                    f'fold{fold}/val/auprc_{histone}': auprc
                }, epoch)
        
        # Log mean metrics
        mean_auroc = np.mean([ROC(labels[:, i], preds[:, i]) for i in range(len(histones)) if len(np.unique(labels[:, i])) > 1])
        mean_auprc = np.mean([auPR(labels[:, i], preds[:, i]) for i in range(len(histones)) if len(np.unique(labels[:, i])) > 1])
        
        tb_logger.log_scalars({
            f'fold{fold}/val/mean_auroc': mean_auroc,
            f'fold{fold}/val/mean_auprc': mean_auprc,
            f'fold{fold}/val/loss': np.mean(losses)
        }, epoch)
        
        # Histograms and confusion matrices will be logged at the end of training

    return (np.mean(losses), labels, preds)


def model_predict(regions, model, batchsize, dna_dict, dns_dict, label_dict, device, tb_logger=None, fold=0):
    model.eval()
    labels = []
    preds = []

    with torch.no_grad():
        for i in range(0, len(regions), batchsize):
            batch = regions[i:i+batchsize]
            if len(batch) < 2:
                print(f"[INFO] Skipping test batch with size {len(batch)} at index {i}")
                continue

            seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)

            seq_tensor = torch.tensor(seq_batch, dtype=torch.float32).to(device)
            if seq_tensor.dim() == 4 and seq_tensor.shape[1] == 1:
                seq_tensor = seq_tensor.squeeze(1)

            dns_tensor = torch.tensor(dns_batch, dtype=torch.float32).to(device)
            if dns_tensor.dim() == 4 and dns_tensor.shape[1] == 1:
                dns_tensor = dns_tensor.squeeze(1)

            label_tensor = torch.tensor(lab_batch, dtype=torch.float32).to(device)
            if label_tensor.dim() == 3 and label_tensor.shape[1] == 1:
                label_tensor = label_tensor.squeeze(1)

            print(f"[DEBUG][predict] seq_tensor: {seq_tensor.shape}, dns_tensor: {dns_tensor.shape}, label_tensor: {label_tensor.shape}")

            pred = model(seq_tensor, dns_tensor)
            labels.extend(label_tensor.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    labels = np.array(labels)
    preds = np.array(preds)
    
    # Log test metrics if logger provided
    if tb_logger and len(labels) > 0:
        # Calculate and log AUROC/AUPRC for each histone
        for i, histone in enumerate(histones):
            if len(np.unique(labels[:, i])) > 1:  # Check if both classes present
                auroc = ROC(labels[:, i], preds[:, i])
                auprc = auPR(labels[:, i], preds[:, i])
                
                tb_logger.log_scalars({
                    f'fold{fold}/test/auroc_{histone}': auroc,
                    f'fold{fold}/test/auprc_{histone}': auprc
                }, 0)  # Use step 0 for test metrics
        
        # Log mean metrics
        mean_auroc = np.mean([ROC(labels[:, i], preds[:, i]) for i in range(len(histones)) if len(np.unique(labels[:, i])) > 1])
        mean_auprc = np.mean([auPR(labels[:, i], preds[:, i]) for i in range(len(histones)) if len(np.unique(labels[:, i])) > 1])
        
        tb_logger.log_scalars({
            f'fold{fold}/test/mean_auroc': mean_auroc,
            f'fold{fold}/test/mean_auprc': mean_auprc
        }, 0)
        
        # Log histograms and confusion matrices for test
        tb_logger.log_histograms(preds, labels, 0, prefix=f"fold{fold}/test")
        tb_logger.log_confusion_matrices(preds, labels, 0, prefix=f"fold{fold}/test")

    return labels, preds


def ROC(label, pred):
    label = np.array(label).reshape(-1)
    pred = np.array(pred).reshape(-1)
    if len(np.unique(label)) == 1:
        print("All labels are the same — ROC undefined.")
        return 0
    return roc_auc_score(label, pred)


def auPR(label, pred):
    label = np.array(label).reshape(-1)
    pred = np.array(pred).reshape(-1)
    if len(np.unique(label)) == 1:
        print("All labels are the same — auPRC undefined.")
        return 0
    precision, recall, _ = precision_recall_curve(label, pred)
    return auc(recall, precision)


def metrics(lab, pred, Type='test', loss=None):
    if Type.lower() == 'valid':
        color = '\033[0;34m'
    elif Type.lower() == 'test':
        color = '\033[0;35m'
    else:
        color = '\033[0;36m'

    auPRC_dict = {}
    auROC_dict = {}

    for i in range(len(histones)):
        auPRC_dict[histones[i]] = auPR(lab[:, i], pred[:, i])
        auROC_dict[histones[i]] = ROC(lab[:, i], pred[:, i])

    mean_auPRC = np.mean(list(auPRC_dict.values()))
    mean_auROC = np.mean(list(auROC_dict.values()))

    print('-' * 25 + f' {Type.upper()} ' + '-' * 25)
    print(f'\033[0;36m{Type}\tTotalMean\tauROC : {mean_auROC:.4f}, auPRC : {mean_auPRC:.4f}' + (f", Loss : {loss:.4f}" if loss is not None else "") + '\033[0m')

    for histone in histones:
        print(color + f"{Type}\t{histone.ljust(10)}\tauROC : {auROC_dict[histone]:.4f}, auPRC : {auPRC_dict[histone]:.4f}\033[0m")

    return auPRC_dict, auROC_dict