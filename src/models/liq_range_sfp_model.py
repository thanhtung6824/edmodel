"""Liq+Range+SFP model: 1 classification gate + 2 regression heads.

Outputs (B, 3):
  [0]: P(profitable) logit — classification gate (apply sigmoid externally)
  [1]: TP1 distance — conservative (softplus, quantile τ=0.3)
  [2]: TP2 distance — aggressive (softplus, quantile τ=0.7)

30 features, ~16K params — single-head self-attention + FFN + 3 heads.
"""

import torch
from torch import nn
import torch.nn.functional as F


class LiqRangeSFPClassifier(nn.Module):
    def __init__(self, n_features=30, window=30, hidden=32):
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
        # Classification gate: P(profitable) logit
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1),
        )
        # Regression heads: TP distances (softplus ensures positive output)
        self.tp1_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1),
        )
        self.tp2_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1),
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
        combined = torch.cat([pooled, last, mx], dim=-1)  # (B, hidden*3)

        cls_logit = self.cls_head(combined)          # (B, 1)
        tp1 = F.softplus(self.tp1_head(combined))    # (B, 1) — positive
        tp2 = F.softplus(self.tp2_head(combined))    # (B, 1) — positive

        return torch.cat([cls_logit, tp1, tp2], dim=-1)  # (B, 3)
