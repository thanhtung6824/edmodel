"""Lightweight classifier for Liq+Range+SFP strategy.

Predicts P(win) — probability that TP gets hit before SL.
24 features, ~12.5K params — single-head self-attention + FFN.
"""

import torch
from torch import nn


class LiqRangeSFPClassifier(nn.Module):
    def __init__(self, n_features=24, window=30, hidden=32):
        super().__init__()
        self.bar_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(hidden, num_heads=1, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, window, n_features)
        h = self.bar_proj(x)           # (B, window, hidden)
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))
        pooled = h.mean(dim=1)         # (B, hidden) — global context
        last = h[:, -1, :]             # (B, hidden) — signal bar
        mx = h.max(dim=1).values       # (B, hidden) — max activation
        combined = torch.cat([pooled, last, mx], dim=-1)
        return self.head(combined).squeeze(-1)  # raw logit
