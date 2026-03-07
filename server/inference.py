"""Model loading and inference for Liq+Range+SFP model.

Model outputs (B, 4):
  [0]: P(profitable) logit — classification gate
  [1]: TP1 distance (softplus, fixed quantile)
  [2]: TP2 distance (softplus, fixed quantile)
  [3]: ttp — time-to-peak ∈ [0, 1]
"""

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
from server.config import WINDOW, WINDOW_BY_TF, N_FEATURES, HIDDEN_DIM

# Asset/TF ID maps (must match training)
ASSET_ID_MAP = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
TF_ID_MAP = {"15m": 0, "1h": 1, "4h": 2}
DIRECTION_MAP = {"LONG": 1, "SHORT": 2, 1: 1, 2: 2}


def load_scaler(path: str) -> StandardScaler:
    """Load saved StandardScaler from disk."""
    return joblib.load(path)


def load_model(path: str) -> LiqRangeSFPClassifier:
    """Load trained LiqRangeSFPClassifier model (CPU only for server)."""
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=WINDOW, hidden=HIDDEN_DIM)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_bar(
    model: LiqRangeSFPClassifier,
    scaled_features,
    bar_idx: int,
    tf_key: str = "4h",
    asset_id: float = 1.0,
    direction: int = 0,
) -> tuple[float, float, float, float] | None:
    """Run model on a specific bar.

    Args:
        direction: 1 for LONG, 2 for SHORT, 0 for unknown.

    Returns (P(profitable), tp1_dist, tp2_dist, ttp), or None if not enough data.
    """
    window = WINDOW_BY_TF.get(tf_key, WINDOW)
    if bar_idx < window - 1 or bar_idx >= len(scaled_features):
        return None

    x = scaled_features[bar_idx - window + 1: bar_idx + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)

    # Create asset/tf/direction ID tensors
    a_id = torch.LongTensor([ASSET_ID_MAP.get(asset_id, 0)])
    t_id = torch.LongTensor([TF_ID_MAP.get(tf_key, 0)])
    d_id = torch.LongTensor([DIRECTION_MAP.get(direction, 0)])

    with torch.no_grad():
        out = model(x_t, asset_ids=a_id, tf_ids=t_id, direction_ids=d_id)  # (1, 4)

    out = out.squeeze(0)  # (4,)
    p_win = torch.sigmoid(out[0]).item()
    tp1_dist = out[1].item()
    tp2_dist = out[2].item()
    ttp = out[3].item()
    return (p_win, tp1_dist, tp2_dist, ttp)
