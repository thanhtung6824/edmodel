"""Model loading and inference for Liq+Range+SFP classifier."""

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
) -> float | None:
    """Run classifier on a specific bar.

    Returns P(win) probability, or None if not enough data.
    """
    window = WINDOW_BY_TF.get(tf_key, WINDOW)
    if bar_idx < window - 1 or bar_idx >= len(scaled_features):
        return None

    x = scaled_features[bar_idx - window + 1: bar_idx + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)

    with torch.no_grad():
        logit = model(x_t)

    return torch.sigmoid(logit).item()
