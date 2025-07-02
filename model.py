import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import  Optimizer
import math 
from torch.nn.parameter import Parameter

class BasicBlock(nn.Module):
    def __init__(self, in_planes, grow_rate, dropout_rate):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, grow_rate, (1,9), 1, (0,4)),
            nn.Dropout2d(dropout_rate)
        )
    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out],1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, grow_rate, dropout_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(in_planes + i*grow_rate, grow_rate, dropout_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):  # <-- Add this
        return self.layer(x)


class ModuleDense(nn.Module):
    def __init__(self, SeqOrDnase='seq', dropout_rate=0.2):
        super(ModuleDense, self).__init__()
        self.SeqOrDnase = SeqOrDnase
        conv_kernel = (4,9) if self.SeqOrDnase == 'seq' else (1,9)
        conv_pad = (0,4)
        in_channels = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, conv_kernel, 1, conv_pad),
            nn.Dropout2d(dropout_rate),
        )

        self.block1 = DenseBlock(3, 128, 128, dropout_rate)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(128 + 3*128),
            nn.ReLU(),
            nn.Conv2d(128 + 3*128, 256, (1,1), 1),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d((1,4)),
        )
        self.block2 = DenseBlock(3, 256, 256, dropout_rate)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(256 + 3*256),
            nn.ReLU(),
            nn.Conv2d(256 + 3*256, 512, (1,1), 1),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d((1,4)),
        )
        self.out_size = 1000 // 4 // 4 * 512

    def forward(self, seq):
            n, h, w = seq.size()
            if self.SeqOrDnase == 'seq':
                seq = seq.view(n, 1, 4, w)
            elif self.SeqOrDnase == 'dnase':
                seq = seq.view(n, 1, 1, w)

            out = self.conv1(seq)
            out = self.block1(out)
            out = self.trans1(out)
            out = self.block2(out)
            out = self.trans2(out)
            out = out.view(out.size(0), -1)
            return out



class NetDeepHistone(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(NetDeepHistone, self).__init__()
        print('DeepHistone(Dense,Dense) is used.')

        self.seq_map = ModuleDense(SeqOrDnase='seq', dropout_rate=dropout_rate)
        self.seq_len = self.seq_map.out_size

        self.dns_map = ModuleDense(SeqOrDnase='dnase', dropout_rate=dropout_rate)
        self.dns_len = self.dns_map.out_size

        combined_len = self.seq_len + self.dns_len

        self.linear_map = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(combined_len, 925),
            nn.BatchNorm1d(925),
            nn.ReLU(),
            nn.Linear(925, 7),
            nn.Sigmoid()
        )

    def forward(self, seq, dns):
        flat_seq = self.seq_map(seq)                  # shape: (N, seq_len)
        dns_out = self.dns_map(dns)                   # shape: (N, dns_len)
        flat_dns = dns_out.view(dns_out.size(0), -1)  # ensure flatten if needed
        combined = torch.cat([flat_seq, flat_dns], dim=1)
        out = self.linear_map(combined)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepHistone:
    def __init__(self, use_gpu, learning_rate=0.001, dropout_rate=0.5):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.forward_fn = NetDeepHistone(dropout_rate=dropout_rate).to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self.forward_fn.parameters(), lr=learning_rate, weight_decay=0)
        self.use_gpu = use_gpu
        print(f"DeepHistone initialized on device: {self.device}")

    def updateLR(self, fold):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= fold

    def train_on_batch(self, seq_batch, dns_batch, lab_batch):
        self.forward_fn.train()
        
        seq_batch = seq_batch.to(self.device)
        dns_batch = dns_batch.to(self.device)
        lab_batch = lab_batch.to(self.device)

        output = self.forward_fn(seq_batch, dns_batch)
        loss = self.criterion(output, lab_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_on_batch(self, seq_batch, dns_batch, lab_batch):
        self.forward_fn.eval()
        with torch.no_grad():
            seq_batch = seq_batch.to(self.device, dtype=torch.float32)
            dns_batch = dns_batch.to(self.device, dtype=torch.float32)
            lab_batch = lab_batch.to(self.device, dtype=torch.float32)

            output = self.forward_fn(seq_batch, dns_batch)
            loss = self.criterion(output, lab_batch)

        return loss.item(), output.detach().cpu().numpy()

    def test_on_batch(self, seq_batch, dns_batch):
        self.forward_fn.eval()
        with torch.no_grad():
            seq_batch = seq_batch.to(self.device, dtype=torch.float32)
            dns_batch = dns_batch.to(self.device, dtype=torch.float32)
            output = self.forward_fn(seq_batch, dns_batch)

        return output.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.forward_fn.state_dict(), path)

    def load_model(self, path):
        self.forward_fn.load_state_dict(torch.load(path, map_location=self.device))

