"""Liq+Range+SFP model: 1 classification gate + 2 regression heads +
1 time-to-peak head + direction-conditioned FiLM + positional encoding.

Outputs (B, 4):
  [0]: P(profitable) logit — classification gate (apply sigmoid externally)
  [1]: TP1 distance — conservative (softplus, fixed quantile)
  [2]: TP2 distance — aggressive (softplus, fixed quantile)
  [3]: ttp — time-to-peak ∈ [0, 1] (sigmoid)

33 features, FiLM conditioning on asset/TF/direction embeddings.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ConditioningModule(nn.Module):
    """FiLM conditioning: asset/TF/direction embeddings → per-head scale + bias."""

    def __init__(self, n_assets=6, n_tfs=4, asset_dim=8, tf_dim=4, n_heads=4):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_dim)
        self.tf_emb = nn.Embedding(n_tfs, tf_dim)
        self.direction_emb = nn.Embedding(3, 4)  # 0=none, 1=long, 2=short
        self.film = nn.Sequential(
            nn.Linear(asset_dim + tf_dim + 4, 16),  # +4 for direction_emb
            nn.ReLU(),
            nn.Linear(16, n_heads * 2),  # scale + bias per head
        )
        self.n_heads = n_heads

    def forward(self, asset_ids, tf_ids, direction_ids=None):
        """Returns (scale, bias) each of shape (B, n_heads)."""
        a = self.asset_emb(asset_ids)  # (B, asset_dim)
        t = self.tf_emb(tf_ids)        # (B, tf_dim)
        if direction_ids is None:
            direction_ids = torch.zeros_like(asset_ids)
        d = self.direction_emb(direction_ids)  # (B, 4)
        h = self.film(torch.cat([a, t, d], dim=-1))  # (B, n_heads*2)
        raw_scale = h[:, :self.n_heads]
        raw_bias = h[:, self.n_heads:]
        # Constrain: scale ∈ [0.7, 1.3], bias ∈ [-0.1, 0.1]
        scale = 0.7 + 0.6 * torch.sigmoid(raw_scale)
        bias = 0.1 * torch.tanh(raw_bias)
        return scale, bias


class LiqRangeSFPClassifier(nn.Module):
    def __init__(self, n_features=33, window=30, hidden=32):
        super().__init__()
        self.bar_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
        )
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, window, hidden) * 0.02)
        # Transformer layer 1
        self.attn = nn.MultiheadAttention(hidden, num_heads=1, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        # Transformer layer 2
        self.attn2 = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True, dropout=0.1)
        self.attn2_norm = nn.LayerNorm(hidden)
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.ffn2_norm = nn.LayerNorm(hidden)
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
        # Time-to-peak head
        self.ttp_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1),
        )
        # FiLM conditioning on cls/tp1/tp2/ttp (4 heads)
        self.conditioning = ConditioningModule(n_assets=6, n_tfs=4, n_heads=4)

    def forward(self, x, asset_ids=None, tf_ids=None, direction_ids=None):
        # x: (B, window, n_features)
        B, seq_len = x.shape[0], x.shape[1]
        h = self.bar_proj(x)           # (B, window, hidden)
        h = h + self.pos_embed[:, :seq_len, :]  # positional encoding
        # Transformer layer 1
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))
        # Transformer layer 2
        attn_out2, _ = self.attn2(h, h, h)
        h = self.attn2_norm(h + attn_out2)
        h = self.ffn2_norm(h + self.ffn2(h))
        # Pooling
        pooled = h.mean(dim=1)         # (B, hidden) — global context
        last = h[:, -1, :]             # (B, hidden) — signal bar
        mx = h.max(dim=1).values       # (B, hidden) — max activation
        combined = torch.cat([pooled, last, mx], dim=-1)  # (B, hidden*3)

        cls_logit = self.cls_head(combined)          # (B, 1)
        tp1 = F.softplus(self.tp1_head(combined))    # (B, 1) — positive
        tp2 = F.softplus(self.tp2_head(combined))    # (B, 1) — positive
        ttp = torch.sigmoid(self.ttp_head(combined))  # (B, 1) ∈ [0, 1]

        # Apply FiLM to all 4 heads
        if asset_ids is not None and tf_ids is not None:
            scale, bias = self.conditioning(asset_ids, tf_ids, direction_ids)  # (B, 4), (B, 4)
            cls_logit = cls_logit * scale[:, 0:1] + bias[:, 0:1]
            tp1 = tp1 * scale[:, 1:2] + bias[:, 1:2]
            tp2 = tp2 * scale[:, 2:3] + bias[:, 2:3]
            ttp = ttp * scale[:, 3:4] + bias[:, 3:4]

        return torch.cat([cls_logit, tp1, tp2, ttp], dim=-1)  # (B, 4)
