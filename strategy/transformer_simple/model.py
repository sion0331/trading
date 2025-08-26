import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, d]

    def forward(self, x):  # x: [B,T,d]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TinyTransformer(nn.Module):
    def __init__(self, in_dim=5, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, num_classes=3):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               batch_first=True, dropout=dropout, norm_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):  # x: [B,T,in_dim]
        h = self.input(x)
        h = self.pos(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        return self.head(h_last)
