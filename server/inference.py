"""Model loading and inference for Liq+Range+SFP model.

Model outputs (B, 3):
  [0]: P(profitable) logit — classification gate
  [1]: TP1 distance (softplus, conservative quantile)
  [2]: TP2 distance (softplus, aggressive quantile)
"""

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
from server.config import WINDOW, WINDOW_BY_TF, N_FEATURES


def load_scaler(path: str) -> StandardScaler:
    """Load saved StandardScaler from disk."""
    return joblib.load(path)


def load_model(path: str) -> LiqRangeSFPClassifier:
    """Load trained LiqRangeSFPClassifier model (CPU only for server)."""
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=WINDOW, hidden=32)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_bar(
    model: LiqRangeSFPClassifier,
    scaled_features,
    bar_idx: int,
    tf_key: str = "4h",
) -> tuple[float, float, float] | None:
    """Run model on a specific bar.

    Returns (P(profitable), tp1_dist, tp2_dist), or None if not enough data.
    """
    window = WINDOW_BY_TF.get(tf_key, WINDOW)
    if bar_idx < window - 1 or bar_idx >= len(scaled_features):
        return None

    x = scaled_features[bar_idx - window + 1: bar_idx + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)

    with torch.no_grad():
        out = model(x_t)  # (1, 3)

    out = out.squeeze(0)  # (3,)
    p_win = torch.sigmoid(out[0]).item()
    tp1_dist = out[1].item()
    tp2_dist = out[2].item()
    return (p_win, tp1_dist, tp2_dist)
