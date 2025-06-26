from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import numpy as np
import torch
import torch.nn as nn

histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']


def loadRegions(regions_indices, dna_dict, dns_dict, label_dict):
    dna_regions = np.concatenate([dna_dict[meta] for meta in regions_indices], axis=0) if dna_dict is not None else []
    dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indices], axis=0) if dns_dict is not None else []
    label_regions = np.concatenate([label_dict[meta] for meta in regions_indices], axis=0).astype(int)
    return dna_regions, dns_regions, label_regions


def model_train(regions, model, batchsize, dna_dict, dns_dict, label_dict, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    train_loss = []

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
        pred = model(seq_tensor, dns_tensor)
        loss = criterion(pred, label_tensor)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss) if train_loss else 0.0


def model_eval(regions, model, batchsize, dna_dict, dns_dict, label_dict, device):
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

    return (np.mean(losses), np.array(labels), np.array(preds)) if losses else (0.0, np.zeros((0, 7)), np.zeros((0, 7)))


def model_predict(regions, model, batchsize, dna_dict, dns_dict, label_dict, device):
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

    return np.array(labels), np.array(preds)


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
