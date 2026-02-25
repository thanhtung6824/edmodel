import torch
from torch import nn


class EdgeLSTMModel(nn.Module):
    def __init__(self, n_features=16, hidden_size=64, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attn_weight = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        out, _ = self.lstm(x)  # (batch, seq_len, 64)
        out = self.layer_norm(out)  # normalize all timesteps
        attn_scores = self.attn_weight(out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (out * attn_weights).sum(dim=1)  # (batch, 64)
        context = self.dropout(context)
        return self.fc(context)
