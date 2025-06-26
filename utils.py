from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import numpy as np

histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def loadRegions(regions_indices, dna_dict, dns_dict, label_dict):
    if dna_dict is not None:
        dna_regions = np.concatenate([dna_dict[meta] for meta in regions_indices], axis=0)
    else:
        dna_regions = []

    if dns_dict is not None:
        dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indices], axis=0)
        #added this in because I just want to check this out
        dns_regions = np.log2(1 + dns_regions)
    else:
        dns_regions = []

    label_regions = np.concatenate([label_dict[meta] for meta in regions_indices], axis=0).astype(int)
    return dna_regions, dns_regions, label_regions


def model_train(regions, model, batchsize, dna_dict, dns_dict, label_dict, device=None):
    train_loss = []
    for i in range(0, len(regions), batchsize):
        batch = regions[i:i+batchsize]
        if len(batch) < 2:
            print(f"[INFO] Skipping training batch with size {len(batch)} at index {i}")
            continue
        seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)
        loss = model.train_on_batch(seq_batch, dns_batch, lab_batch)
        train_loss.append(loss)
    if len(train_loss) == 0:
        return 0.0
    return np.mean(train_loss)


def model_eval(regions, model, batchsize, dna_dict, dns_dict, label_dict, device=None):
    losses = []
    preds = []
    labels = []
    for i in range(0, len(regions), batchsize):
        batch = regions[i:i+batchsize]
        if len(batch) < 2:
            print(f"[INFO] Skipping validation batch with size {len(batch)} at index {i}")
            continue
        seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)
        loss, pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
        losses.append(loss)
        labels.extend(lab_batch)
        preds.extend(pred)
    if len(losses) == 0:
        return 0.0, np.zeros((0, 7)), np.zeros((0, 7))
    return np.mean(losses), np.array(labels), np.array(preds)


def model_predict(regions, model, batchsize, dna_dict, dns_dict, label_dict, device=None):
    labels = []
    preds = []
    for i in range(0, len(regions), batchsize):
        batch = regions[i:i+batchsize]
        if len(batch) < 2:
            print(f"[INFO] Skipping test batch with size {len(batch)} at index {i}")
            continue
        seq_batch, dns_batch, lab_batch = loadRegions(batch, dna_dict, dns_dict, label_dict)
        pred = model.test_on_batch(seq_batch, dns_batch)
        labels.extend(lab_batch)
        preds.extend(pred)
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