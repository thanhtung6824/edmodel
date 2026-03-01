import torch
from torch import nn


class SFPTransformer(nn.Module):
    def __init__(self, n_features=22, d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1):
        super(SFPTransformer, self).__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)  # 30 bars + 1 CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # IMPORTANT: our data is (batch, seq, feat)
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Aggregate CLS (global context) + last bar (SFP signal)
        self.aggregate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(d_model)

        # TP head: positive percentage
        self.tp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        # SL head: positive percentage
        self.sl_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.input_proj(x)                           # (B,30,d_model)
        cls = self.cls_token.expand(B, -1, -1)           # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)                   # (B,31,d_model)
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)
        combined = torch.cat([x[:, 0, :], x[:, -1, :]], dim=-1)  # CLS + last bar
        combined = self.aggregate(combined)               # (B, d_model)
        combined = self.norm(combined)
        tp = self.tp_head(combined).squeeze(-1)           # (B,)
        sl = self.sl_head(combined).squeeze(-1)           # (B,)
        return tp, sl
