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
	def __init__(self, in_planes, grow_rate,):
		super(BasicBlock, self).__init__()
		self.block = nn.Sequential(
			nn.BatchNorm2d(in_planes),
			nn.ReLU(),
			nn.Conv2d(in_planes, grow_rate, (1,9), 1, (0,4)),
			#nn.Dropout2d(0.2)
		)
	def forward(self, x):
		out = self.block(x)
		return torch.cat([x, out],1)

class DenseBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, grow_rate,):
		super(DenseBlock, self).__init__()
		layers = []
		for i in range(nb_layers):
			layers.append(BasicBlock(in_planes + i*grow_rate, grow_rate,))
		self.layer = nn.Sequential(*layers)
	def forward(self, x):
		return self.layer(x)


class ModuleDense(nn.Module):
	def __init__(self,SeqOrDnase='seq',):
		super(ModuleDense, self).__init__()
		self.SeqOrDnase = SeqOrDnase
		if self.SeqOrDnase== 'seq':
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(4,9),1,(0,4)),
			#nn.Dropout2d(0.2),
			)
		elif self.SeqOrDnase =='dnase'  :
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(1,9),1,(0,4)),
			#nn.Dropout2d(0.2),
			)	
		self.block1 = DenseBlock(3, 128, 128)	
		self.trans1 = nn.Sequential(
			nn.BatchNorm2d(128+3*128),
			nn.ReLU(),
			nn.Conv2d(128+3*128, 256, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,4)),
		)
		self.block2 = DenseBlock(3,256,256)
		self.trans2 = nn.Sequential(
			nn.BatchNorm2d(256+3*256),
			nn.ReLU(),
			nn.Conv2d(256+3*256, 512, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,4)),
		)
		self.out_size = 1000 // 4 // 4  * 512

	def forward(self, seq):
		n, h, w = seq.size()
		if self.SeqOrDnase=='seq':
			seq = seq.view(n,1,4,w)
		elif self.SeqOrDnase=='dnase':
			seq = seq.view(n,1,1,w)
		out = self.conv1(seq)
		out = self.block1(out)
		out = self.trans1(out)
		out = self.block2(out)
		out = self.trans2(out)
		n, c, h, w = out.size()
		out = out.view(n,c*h*w) 
		return out



class NetDeepHistone(nn.Module):
	def __init__(self, ):
		super(NetDeepHistone, self).__init__()
		print('DeepHistone(Dense,Dense) is used.')
		self.seq_map = ModuleDense(SeqOrDnase='seq',)
		self.seq_len = self.seq_map.out_size
		self.dns_map = ModuleDense(SeqOrDnase='dnase',)
		self.dns_len = self.dns_map.out_size	
		combined_len = self.dns_len + self.seq_len 
		self.linear_map = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(int(combined_len),925),
			nn.BatchNorm1d(925),
			nn.ReLU(),
			#nn.Dropout(0.1),
			nn.Linear(925,7),
			nn.Sigmoid(),
		)

	def forward(self, seq, dns):
		flat_seq = self.seq_map(seq)	
		n, h, w = dns.size()
		dns = self.dns_map(dns) 
		flat_dns = dns.view(n,-1)
		combined = torch.cat([flat_seq, flat_dns], 1)
		out = self.linear_map(combined)
		return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepHistone:
    def __init__(self, use_gpu, learning_rate=0.001):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.forward_fn = NetDeepHistone().to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self.forward_fn.parameters(), lr=learning_rate, weight_decay=0)
        self.use_gpu = use_gpu
        print(f"DeepHistone initialized on device: {self.device}")

    def updateLR(self, fold):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= fold

    def train_on_batch(self, seq_batch, dns_batch, lab_batch):
        self.forward_fn.train()
        seq_batch = torch.tensor(seq_batch, device=self.device, dtype=torch.float32)
        dns_batch = torch.tensor(dns_batch, device=self.device, dtype=torch.float32)
        lab_batch = torch.tensor(lab_batch, device=self.device, dtype=torch.float32)

        output = self.forward_fn(seq_batch, dns_batch)
        loss = self.criterion(output, lab_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_on_batch(self, seq_batch, dns_batch, lab_batch):
        self.forward_fn.eval()
        with torch.no_grad():
            seq_batch = torch.tensor(seq_batch, device=self.device, dtype=torch.float32)
            dns_batch = torch.tensor(dns_batch, device=self.device, dtype=torch.float32)
            lab_batch = torch.tensor(lab_batch, device=self.device, dtype=torch.float32)

            output = self.forward_fn(seq_batch, dns_batch)
            loss = self.criterion(output, lab_batch)

        return loss.item(), output.detach().cpu().numpy()

    def test_on_batch(self, seq_batch, dns_batch):
        self.forward_fn.eval()
        with torch.no_grad():
            seq_batch = torch.tensor(seq_batch, device=self.device, dtype=torch.float32)
            dns_batch = torch.tensor(dns_batch, device=self.device, dtype=torch.float32)
            output = self.forward_fn(seq_batch, dns_batch)

        return output.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.forward_fn.state_dict(), path)

    def load_model(self, path):
        self.forward_fn.load_state_dict(torch.load(path, map_location=self.device))

