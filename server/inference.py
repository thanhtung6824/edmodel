"""Model loading and inference."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models.sfp_transformer import SFPTransformer
from src.models.three_tap_model import ThreeTapClassifier
from src.models.range_sfp_model import RangeSFPClassifier
from src.models.range_quality import RangeQualityClassifier
from server.config import WINDOW, THREE_TAP_N_FEATURES, RANGE_SFP_N_FEATURES, RANGE_QUALITY_N_FEATURES


def load_model(path: str) -> SFPTransformer:
    """Load trained SFPTransformer model (CPU only for server)."""
    model = SFPTransformer(n_features=23)
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
        tp_pred, sl_pred, q_logit = model(x_t)

    tp = tp_pred.item()
    sl = sl_pred.item()
    ratio = tp / (sl + 1e-6)
    p_win = torch.sigmoid(q_logit).item()

    return tp, sl, ratio, p_win


def predict_bar(model: SFPTransformer, feat_values: np.ndarray, bar_idx: int, window: int = WINDOW):
    """Run inference on a specific bar index.

    Returns:
        (tp, sl, ratio, p_win) for the bar, or None if not enough data.
    """
    if bar_idx < window - 1 or bar_idx >= len(feat_values):
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_values)

    x = scaled[bar_idx - window + 1: bar_idx + 1]  # (30, 22)
    x_t = torch.FloatTensor(x).unsqueeze(0)  # (1, 30, 22)

    with torch.no_grad():
        tp_pred, sl_pred, q_logit = model(x_t)

    tp = tp_pred.item()
    sl = sl_pred.item()
    ratio = tp / (sl + 1e-6)
    p_win = torch.sigmoid(q_logit).item()

    return tp, sl, ratio, p_win


def load_three_tap_model(path: str) -> ThreeTapClassifier:
    """Load trained ThreeTapClassifier model (CPU only for server)."""
    model = ThreeTapClassifier(n_features=THREE_TAP_N_FEATURES, window=WINDOW, hidden=24)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_three_tap_bar(
    model: ThreeTapClassifier,
    scaled_features: np.ndarray,
    bar_idx: int,
    window: int = WINDOW,
) -> float | None:
    """Run three-tap classifier on a specific bar.

    Returns P(win) probability, or None if not enough data.
    """
    if bar_idx < window - 1 or bar_idx >= len(scaled_features):
        return None

    x = scaled_features[bar_idx - window + 1: bar_idx + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)

    with torch.no_grad():
        logit = model(x_t)

    return torch.sigmoid(logit).item()


def load_range_sfp_model(path: str) -> RangeSFPClassifier:
    """Load trained RangeSFPClassifier model (CPU only for server)."""
    model = RangeSFPClassifier(n_features=RANGE_SFP_N_FEATURES, window=WINDOW, hidden=22)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_range_sfp_bar(
    model: RangeSFPClassifier,
    scaled_features: np.ndarray,
    bar_idx: int,
    window: int = WINDOW,
) -> float | None:
    """Run range-SFP classifier on a specific bar.

    Returns P(win) probability, or None if not enough data.
    """
    if bar_idx < window - 1 or bar_idx >= len(scaled_features):
        return None

    x = scaled_features[bar_idx - window + 1: bar_idx + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)

    with torch.no_grad():
        logit = model(x_t)

    return torch.sigmoid(logit).item()


def load_range_quality_model(path: str) -> RangeQualityClassifier:
    """Load trained RangeQualityClassifier model (CPU only for server)."""
    model = RangeQualityClassifier(n_features=RANGE_QUALITY_N_FEATURES, hidden=16)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_range_quality(
    model: RangeQualityClassifier,
    features: np.ndarray,
) -> np.ndarray:
    """Run range quality classifier on batch of range features.

    Args:
        model: Loaded RangeQualityClassifier
        features: (N, 16) float32 array of range features

    Returns:
        (N,) array of P(valid) probabilities
    """
    if len(features) == 0:
        return np.array([], dtype=np.float32)

    x_t = torch.FloatTensor(features)
    with torch.no_grad():
        logits = model(x_t)
    return torch.sigmoid(logits).numpy()
