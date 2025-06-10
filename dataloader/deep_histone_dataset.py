from torch.utils.data import Dataset
import numpy as np
import torch

class DeepHistoneDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.seq = data['dna']       # shape (N, 4, 1000)
        self.dns = data['dnase']     # shape (N, 1, 1000)
        self.lab = data['label']     # shape (N, 7)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seq[idx], dtype=torch.float32),
            torch.tensor(self.dns[idx], dtype=torch.float32),
            torch.tensor(self.lab[idx], dtype=torch.float32)
        )
