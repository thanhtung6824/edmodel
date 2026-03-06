"""Per-bar Range Detector model.

Predicts P(in_tradeable_range) per bar using a window of recent bars.
~1.3K params — lightweight window-based classifier.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class RangeDetector(nn.Module):
    def __init__(self, n_features=14, hidden=22):
        super().__init__()
        # Per-bar feature projection
        self.bar_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
        )
        # Combine mean pool + last bar -> predict range probability
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, window, n_features)
        h = self.bar_proj(x)           # (B, window, hidden)
        pooled = h.mean(dim=1)         # (B, hidden) — global context
        last = h[:, -1, :]             # (B, hidden) — current bar
        combined = torch.cat([pooled, last], dim=-1)
        logit = self.head(combined).squeeze(-1)  # (B,)
        return logit  # raw logit, apply sigmoid outside


class RangeDetectorDataset(Dataset):
    """Indexes EVERY bar (not just signals), with a sliding window."""

    def __init__(self, scaled_data, labels, window=60):
        self.data = scaled_data
        self.labels = labels
        self.window = window
        # Every bar with enough history is a sample
        self.valid_start = window - 1
        self.length = max(0, len(labels) - self.valid_start)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = idx + self.valid_start
        x = self.data[real_idx - self.window + 1 : real_idx + 1]
        y = self.labels[real_idx]
        return (
            torch.FloatTensor(x),
            torch.FloatTensor([y]).squeeze(),
        )
