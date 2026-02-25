import torch
from torch import nn


class SFPTransformer(nn.Module):
    def __init__(self, n_features=22, d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1):
        super(SFPTransformer, self).__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
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
        x = self.input_proj(x)  # (B,30,N) -> (B,30,d_model)
        x = self.dropout(x + self.pos_embed)  # add position info
        x = self.encoder(x)  # self-attention
        x = x[:, -1, :]  # take LAST bar only (the SFP candle)
        x = self.norm(x)
        tp = self.tp_head(x).squeeze(-1)  # (B,)
        sl = self.sl_head(x).squeeze(-1)  # (B,)
        return tp, sl
