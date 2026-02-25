import torch
from torch import nn
import torch.nn.functional as F


class SFPLoss(nn.Module):
    def __init__(self, pos_weight=1.0, lambda_tp=1.0, lambda_sl=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.lambda_tp = lambda_tp
        self.lambda_sl = lambda_sl

    def forward(self, quality_logit, tp_pred, sl_pred, quality_target, tp_target, sl_target):
        # Binary classification: profitable or not
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logit, quality_target.float(), pos_weight=self.pos_weight
        )

        # Regression: TP and SL for all SFP samples
        tp_loss = F.smooth_l1_loss(tp_pred, tp_target)
        sl_loss = F.smooth_l1_loss(sl_pred, sl_target)

        total = quality_loss + self.lambda_tp * tp_loss + self.lambda_sl * sl_loss
        return total, quality_loss, tp_loss, sl_loss
