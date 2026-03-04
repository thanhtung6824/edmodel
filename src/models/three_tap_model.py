"""Lightweight classifier for Three-Tap strategy.

Predicts P(win) — probability that TP gets hit before SL.
~1.5K params — sized for ~2000 signals.
"""

import torch
from torch import nn


class ThreeTapClassifier(nn.Module):
    def __init__(self, n_features=18, window=30, hidden=24):
        super().__init__()
        # Shared per-bar feature transform
        self.bar_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
        )
        # Combine mean pool + last bar → predict win probability
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, window, n_features)
        h = self.bar_proj(x)           # (B, window, hidden)
        pooled = h.mean(dim=1)         # (B, hidden) — global context
        last = h[:, -1, :]             # (B, hidden) — signal bar
        combined = torch.cat([pooled, last], dim=-1)
        logit = self.head(combined).squeeze(-1)  # (B,)
        return logit  # raw logit, apply sigmoid outside
