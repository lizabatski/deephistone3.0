import torch
import torch.nn as nn
import math
import os
import numpy as np
from model import ModuleDense


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EmbeddingStrategy(nn.Module):
    def __init__(self, input_type, config):
        super().__init__()
        in_channels = 4 if input_type == 'seq' else 1
        self.type = config.get('embedding_type', 'linear')
        self.d_model = config['d_model']
        
        if self.type == 'linear':
            self.embed = nn.Linear(in_channels, self.d_model)
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
            return self.embed(x)
        elif self.type == 'conv':
            x = self.embed(x)
            return x.permute(0, 2, 1)
        elif self.type == 'linear':
            return self.embed(x.permute(0, 2, 1))
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
        if self.pooling_type == 'mean':
            return x.mean(dim=1)
        elif self.pooling_type == 'max':
            return x.max(dim=1).values
        elif self.pooling_type == 'cls':
            return x[:, 0, :]
        elif self.pooling_type == 'flatten':
            return x.view(x.size(0), -1)
        elif self.pooling_type == 'attention':
            attn_weights = self.attn_pool(x).squeeze(-1)
            attn_scores = torch.softmax(attn_weights, dim=1)
            return torch.sum(x * attn_scores.unsqueeze(-1), dim=1)
        elif self.pooling_type == 'identity':
            return x
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None  # Store all attention weights
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Get attention weights from self-attention
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Store attention weights (batch_size, num_heads, seq_len, seq_len)
        self.attention_weights = attn_weights
        
        # Continue with standard transformer layer computation
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class AttentionVisualizer:
    """Helper class to save and analyze attention maps"""
    
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def save_attention_maps(self, attention_weights_list, batch_idx=0, sample_name="sample"):
        """
        Save attention maps from all layers and heads
        
        Args:
            attention_weights_list: List of attention tensors from each layer
            batch_idx: Which sample in the batch to save
            sample_name: Name for the saved files
        """
        if not self.save_dir:
            return
            
        for layer_idx, attn_weights in enumerate(attention_weights_list):
            if attn_weights is None:
                continue
                
            # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
            attn_batch = attn_weights[batch_idx].detach().cpu().numpy()
            
            # Save each head separately
            for head_idx in range(attn_batch.shape[0]):
                filename = f"{sample_name}_layer{layer_idx}_head{head_idx}_attn.npy"
                filepath = os.path.join(self.save_dir, filename)
                np.save(filepath, attn_batch[head_idx])
                
        print(f"Saved attention maps for {len(attention_weights_list)} layers to {self.save_dir}")
    
    def analyze_attention_patterns(self, attention_weights_list):
        """
        Analyze attention patterns to identify biologically relevant heads/layers
        Returns metrics for each layer and head
        """
        results = {}
        
        for layer_idx, attn_weights in enumerate(attention_weights_list):
            if attn_weights is None:
                continue
                
            # Average across batch: (num_heads, seq_len, seq_len)
            avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()
            
            layer_results = {}
            for head_idx in range(avg_attn.shape[0]):
                head_attn = avg_attn[head_idx]
                
                # Calculate various metrics
                metrics = {
                    'entropy': self._calculate_entropy(head_attn),
                    'diagonal_focus': self._calculate_diagonal_focus(head_attn),
                    'long_range_connections': self._calculate_long_range(head_attn),
                    'sparsity': self._calculate_sparsity(head_attn)
                }
                
                layer_results[f'head_{head_idx}'] = metrics
            
            results[f'layer_{layer_idx}'] = layer_results
            
        return results
    
    def _calculate_entropy(self, attn_matrix):
        """Calculate entropy of attention distribution"""
        # Avoid log(0) by adding small epsilon
        entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10), axis=-1)
        return float(np.mean(entropy))
    
    def _calculate_diagonal_focus(self, attn_matrix):
        """Calculate how much attention focuses on nearby positions"""
        seq_len = attn_matrix.shape[0]
        diagonal_sum = 0
        for i in range(seq_len):
            for j in range(max(0, i-5), min(seq_len, i+6)):  # ±5 positions
                diagonal_sum += attn_matrix[i, j]
        return float(diagonal_sum / seq_len)
    
    def _calculate_long_range(self, attn_matrix):
        """Calculate attention to distant positions"""
        seq_len = attn_matrix.shape[0]
        long_range_sum = 0
        count = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > 50:  # Positions >50bp apart
                    long_range_sum += attn_matrix[i, j]
                    count += 1
        return float(long_range_sum / count if count > 0 else 0)
    
    def _calculate_sparsity(self, attn_matrix):
        """Calculate sparsity (how concentrated the attention is)"""
        # Gini coefficient as sparsity measure
        flat_attn = attn_matrix.flatten()
        flat_attn = np.sort(flat_attn)
        n = len(flat_attn)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * flat_attn)) / (n * np.sum(flat_attn)))


class ModuleTransformer(nn.Module):
    def __init__(self, input_type, config, log_attention_dir=None):
        super().__init__()
        self.config = config
        self.log_attention_dir = log_attention_dir
        self.embedding = EmbeddingStrategy(input_type, config)
        self.pe = PositionalEncoding(d_model=self.embedding.d_model)
        
        # Create custom encoder layers
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=self.embedding.d_model,
                nhead=config['nhead'],
                dim_feedforward=config.get('dim_feedforward', 512),
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_layers'])
        ])
        
        self.pooling = PoolingStrategy(
            pooling_type=config['pooling'],
            d_model=self.embedding.d_model,
            seq_len=config.get('seq_len', 1000)
        )
        
        # Initialize attention visualizer
        self.attention_visualizer = AttentionVisualizer(log_attention_dir)
        
    @property
    def output_dim(self):
        return self.pooling.output_dim
    
    def forward(self, x, save_attention=False, sample_name="sample"):
        x = self.embedding(x)
        x = self.pe(x)
        
        # Store attention weights from each layer
        attention_weights_list = []
        
        # Pass through each encoder layer
        for layer in self.encoder_layers:
            x = layer(x)
            attention_weights_list.append(layer.attention_weights)
        
        # Save attention maps if requested
        if save_attention and self.log_attention_dir:
            self.attention_visualizer.save_attention_maps(
                attention_weights_list, 
                batch_idx=0, 
                sample_name=sample_name
            )
            
            # Analyze attention patterns
            analysis = self.attention_visualizer.analyze_attention_patterns(attention_weights_list)
            print("Attention Analysis:", analysis)
        
        return self.pooling(x)


class NetDeepHistoneTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config is None:
            config = {
                'use_transformer_seq': True,
                'use_transformer_dnase': True,
                'embedding_type': 'linear',    
                'd_model': 128,
                'nhead': 8,
                'dropout': 0.1,
                'num_layers': 4,
                'dim_feedforward': 512,
                'pooling': 'mean',
                'seq_len': 1000,
                'vocab_size': 5               
            }

        # DNA Sequence Encoder
        if config.get('use_transformer_seq', False):
            self.seq_map = ModuleTransformer(
                input_type='seq', 
                config=config,
                log_attention_dir=config.get('attention_save_dir')
            )
            self.seq_len = self.seq_map.output_dim
            print("Using Transformer for DNA sequence input")
        else:
            self.seq_map = ModuleDense(SeqOrDnase='seq')
            self.seq_len = self.seq_map.out_size
            print("Using CNN for DNA sequence input")

        # DNase Encoder
        if config.get('use_transformer_dnase', False):
            self.dns_map = ModuleTransformer(
                input_type='dnase', 
                config=config,
                log_attention_dir=config.get('attention_save_dir')
            )
            self.dns_len = self.dns_map.output_dim
            print("Using Transformer for DNase input")
        else:
            self.dns_map = ModuleDense(SeqOrDnase='dnase')
            self.dns_len = self.dns_map.out_size
            print("Using CNN for DNase input")

        combined_len = self.seq_len + self.dns_len

        # Classifier
        self.linear_map = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_len, 925),
            nn.BatchNorm1d(925),
            nn.ReLU(),
            nn.Linear(925, 7),
            nn.Sigmoid(),
        )

    def forward(self, seq, dns, save_attention=False, sample_name="sample"):
        # Forward pass with optional attention saving
        if hasattr(self.seq_map, 'forward') and 'save_attention' in self.seq_map.forward.__code__.co_varnames:
            flat_seq = self.seq_map(seq, save_attention=save_attention, sample_name=f"{sample_name}_seq")
        else:
            flat_seq = self.seq_map(seq)
            
        if hasattr(self.dns_map, 'forward') and 'save_attention' in self.dns_map.forward.__code__.co_varnames:
            flat_dns = self.dns_map(dns, save_attention=save_attention, sample_name=f"{sample_name}_dnase")
        else:
            flat_dns = self.dns_map(dns)
            
        combined = torch.cat([flat_seq, flat_dns], dim=1)
        out = self.linear_map(combined)
        return out