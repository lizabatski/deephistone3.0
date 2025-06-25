import torch
import torch.nn as nn
import math
from model import ModuleDense


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000): #seguence is max 1000
        super().__init__()
        pe = torch.zeros(max_len, d_model) #creates a tensor of shape (max_len, d_model) will be a matrix that will hold positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #transforms a 1D array into column vector
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  #fills even-index columns with sine values
        pe[:, 1::2] = torch.cos(position * div_term) #fills odd-index columns with cosine values
        # makes 'pe' part of the model but NOT a trainable paramter
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (N, L, C) where N is batch size, L is sequence length, C is feature dimension
        return x + self.pe[:, :x.size(1)] 


class EmbeddingStrategy(nn.Module):
    def __init__(self, input_type, config):
        super().__init__()
        in_channels = 4 if input_type == 'seq' else 1
        self.type = config.get('embedding_type', 'linear')
        self.d_model = config['d_model']
        
        #linear embedding
        if self.type == 'linear':
            self.embed = nn.Linear(in_channels, self.d_model)
        
        #conv embedding
        elif self.type == 'conv':
            self.embed = nn.Sequential(
                nn.Conv1d(in_channels, self.d_model, kernel_size=9, padding=4),
                nn.ReLU()
            )

        
        elif self.type == 'lookup':
            self.embed = nn.Embedding(config['vocab_size'], self.d_model)

        elif self.type == 'identity':
            self.embed = nn.Identity()

        else:
            raise ValueError(f"Unknown embedding type: {self.type}")

    def forward(self, x):
        if self.type == 'lookup':
            return self.embed(x)  # x: (N, L) -> (N, L, d_model)

        elif self.type == 'conv':
            x = self.embed(x)  # x: (N, C, L) -> (N, d_model, L)
            return x.permute(0, 2, 1)  # -> (N, L, d_model)

        elif self.type == 'linear':
            return self.embed(x.permute(0, 2, 1))  # (N, L, C) -> (N, L, d_model)

        elif self.type == 'identity':
            return x  
        
class PoolingStrategy(nn.Module):
    def __init__(self, pooling_type: str, d_model: int, seq_len: int = 1000):
        super().__init__()
        self.pooling_type = pooling_type
        self.d_model = d_model
        self.seq_len = seq_len

        if pooling_type == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)
            )
        elif pooling_type == 'cls':
            # maybe add learnable parameters for cls token
            pass

    @property
    def output_dim(self):
        if self.pooling_type in ['mean', 'max', 'cls', 'attention']:
            return self.d_model
        elif self.pooling_type == 'flatten':
            return self.d_model * self.seq_len
        elif self.pooling_type == 'identity':
            return self.d_model
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(self, x):
        # x: (N, L, d_model)
        if self.pooling_type == 'mean':
            return x.mean(dim=1)

        elif self.pooling_type == 'max':
            return x.max(dim=1).values

        elif self.pooling_type == 'cls':
            return x[:, 0, :]

        elif self.pooling_type == 'flatten':
            return x.view(x.size(0), -1)

        elif self.pooling_type == 'attention':
            attn_weights = self.attn_pool(x).squeeze(-1)        # (N, L)
            attn_scores = torch.softmax(attn_weights, dim=1)    # (N, L)
            return torch.sum(x * attn_scores.unsqueeze(-1), dim=1)  # (N, d_model)

        elif self.pooling_type == 'identity':
            return x  # Return full sequence (N, L, d_model)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")



class ModuleTransformer(nn.Module):
    def __init__(self, input_type, config):
        super().__init__()
        self.embedding = EmbeddingStrategy(input_type, config)
        self.pe = PositionalEncoding(d_model=self.embedding.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding.d_model,
            nhead=config['nhead'],
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config['dropout'],
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        self.pooling = PoolingStrategy(
            pooling_type=config['pooling'],
            d_model=self.embedding.d_model,
            seq_len=config.get('seq_len', 1000)
        )
        self.output_dim = self.pooling.output_dim

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.encoder(x)
        return self.pooling(x)
    

class NetDeepHistoneTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- DNA Sequence Encoder ---
        if config.get('use_transformer_seq', False):
            self.seq_map = ModuleTransformer(input_type='seq', config=config)
            self.seq_len = self.seq_map.output_dim
            print("Using Transformer for DNA sequence input")
        else:
            self.seq_map = ModuleDense(SeqOrDnase='seq')
            self.seq_len = self.seq_map.out_size
            print("Using CNN for DNA sequence input")

        # --- DNase Encoder ---
        if config.get('use_transformer_dnase', False):
            self.dns_map = ModuleTransformer(input_type='dnase', config=config)
            self.dns_len = self.dns_map.output_dim
            print("Using Transformer for DNase input")
        else:
            self.dns_map = ModuleDense(SeqOrDnase='dnase')
            self.dns_len = self.dns_map.out_size
            print("Using CNN for DNase input")

        combined_len = self.seq_len + self.dns_len

        # --- Final Classifier ---
        self.linear_map = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_len, 925),
            nn.BatchNorm1d(925),
            nn.ReLU(),
            nn.Linear(925, 7),
            nn.Sigmoid(),
        )

    def forward(self, seq, dns):
        flat_seq = self.seq_map(seq)    # Shape: (N, seq_len or d_model)
        flat_dns = self.dns_map(dns)    # Shape: (N, dns_len or d_model)
        combined = torch.cat([flat_seq, flat_dns], dim=1)
        out = self.linear_map(combined)
        return out