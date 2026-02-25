"""Model loading and inference."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models.sfp_transformer import SFPTransformer
from server.config import WINDOW


def load_model(path: str) -> SFPTransformer:
    """Load trained SFPTransformer model (CPU only for server)."""
    model = SFPTransformer(n_features=22)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_latest(model: SFPTransformer, feat_values: np.ndarray, window: int = WINDOW):
    """Run inference on the latest bar.

    Fits StandardScaler on all feat_values (same pattern as validate_4h.py:171-172),
    takes the last `window`-bar slice, runs forward pass.

    Returns:
        (tp, sl, ratio) for the latest bar, or None if not enough data.
    """
    if len(feat_values) < window:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_values)

    # Take the last window-sized slice
    x = scaled[-window:]  # (30, 22)
    x_t = torch.FloatTensor(x).unsqueeze(0)  # (1, 30, 22)

    with torch.no_grad():
        tp_pred, sl_pred = model(x_t)

    tp = tp_pred.item()
    sl = sl_pred.item()
    ratio = tp / (sl + 1e-6)

    return tp, sl, ratio
