import torch
from torch import nn


class SFPModel(nn.Module):
    def __init__(self, n_features=14, hidden_size=64, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attn_weight = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.3)

        # Quality head: binary (profitable or not)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # TP head: positive percentage
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # SL head: positive percentage
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        out, _ = self.lstm(x)               # (batch, seq_len, hidden)
        out = self.layer_norm(out)

        # Attention pooling
        attn_scores = self.attn_weight(out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (out * attn_weights).sum(dim=1)  # (batch, hidden)
        context = self.dropout(context)

        quality_logit = self.quality_head(context).squeeze(-1)  # (batch,)
        tp = self.tp_head(context).squeeze(-1)                  # (batch,)
        sl = self.sl_head(context).squeeze(-1)                  # (batch,)

        return quality_logit, tp, sl
