from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import numpy as np
from torch.utils.data import Dataset
import torch

class HistoneDataset(Dataset):
    def __init__(self, keys, dna_dict, dnase_dict, label_dict): #initializes the dataset 
        self.keys = keys
        self.dna_dict = dna_dict
        self.dnase_dict = dnase_dict
        self.label_dict = label_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        dna = self.dna_dict[key][0]      # Remove shape (1, 4, 1000) -> (4, 1000)
        dnase = self.dnase_dict[key][0]  # Remove shape (1, 1, 1000) -> (1, 1000)
        label = self.label_dict[key][0]  
        
        #converts arrays so that they are suitable for model input
        return (
            torch.tensor(dna, dtype=torch.float32), 
            torch.tensor(dnase, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

#was from their github repo
def loadRegions(regions_indices, dna_dict, dns_dict, label_dict):
    if dna_dict is not None:
        dna_regions = np.concatenate([dna_dict[meta] for meta in regions_indices], axis=0) #if DNA data exists, stack all DNA arrays (1, 4, 1000) -> (N, 4, 1000)
    else:
        dna_regions = []

    if dns_dict is not None:
        dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indices], axis=0) #(1, 1, 1000) -> (N, 1, 1000)
        #added this in because I just want to check this out
        #dns_regions = np.log2(1 + dns_regions)
    else:
        dns_regions = []

    label_regions = np.concatenate([label_dict[meta] for meta in regions_indices], axis=0).astype(int) #(1, 7) -> (N, 7)
    return dna_regions, dns_regions, label_regions


def model_train(dataloader, model):
    model.forward_fn.train()
    train_loss = []
    print("[INFO] Starting model_train...")

    for i, (seq_batch, dns_batch, lab_batch) in enumerate(dataloader):
        # NaN check
        if torch.isnan(lab_batch).any():
            print(f"[WARNING] NaNs in lab_batch at batch {i}")
            continue

        try:
            loss = model.train_on_batch(seq_batch, dns_batch, lab_batch)
            if loss is None or np.isnan(loss):
                print(f"[WARNING] Loss is None or NaN at batch {i}")
                continue
            train_loss.append(loss)
            if i == 0 or (i + 1) % 100 == 0:
                print(f"[INFO] Batch {i+1}: loss = {loss:.4f}")
        except Exception as e:
            print(f"[ERROR] Exception in train_on_batch at batch {i}: {e}")
            raise

    if len(train_loss) == 0:
        print("[WARNING] No valid batches in training — returning 0.0")
        return 0.0

    return np.mean(train_loss)

#evaluates the model on the validation set so it still computes the loss
def model_eval(dataloader, model):
    model.forward_fn.eval() 
    losses = []
    preds, labels = [], []
    for seq_batch, dns_batch, lab_batch in dataloader:
        loss, pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
        losses.append(loss)
        labels.append(lab_batch.numpy())
        preds.append(pred)
    return (
        np.mean(losses),
        np.concatenate(labels),
        np.concatenate(preds),
    )

#evaluates the model on the test set
def model_predict(dataloader, model):
    model.forward_fn.eval()
    preds, labels = [], []
    for seq_batch, dns_batch, lab_batch in dataloader:
        pred = model.test_on_batch(seq_batch, dns_batch)
        preds.append(pred)
        labels.append(lab_batch.numpy())
    return (
        np.concatenate(labels),
        np.concatenate(preds),
    )


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