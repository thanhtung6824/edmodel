"""Permutation feature importance for Liq+Range+SFP model.

Shuffles each of the 18 features independently on the test set, measures
TP1 EV degradation at P>0.5. Largest degradation = most important feature.

Usage:
    python -m src.feature_importance
    python -m src.feature_importance --threshold 0.7   # evaluate at P>0.7
"""

import sys
import numpy as np
import torch
from torch import nn

from src.models.liq_range_sfp_model import LiqRangeSFPClassifier

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
MODEL_FILE = "best_model_liq_range_sfp.pth"

# Feature names in order (must match build_features output)
FEATURE_NAMES = [
    "range_height_pct", "range_age", "sweep_depth_range",
    "reclaim_strength_range", "weighted_liq_swept", "max_leverage_norm",
    "liq_cascade_depth", "wick_ratio", "zone_sl_dist",
    "trend_strength", "ms_alignment", "asset_id",
    "is_recaptured", "touch_symmetry", "range_position",
    "direction_feat", "htf_trend", "htf_rsi",
]


def compute_ev(model, test_loaders, threshold=0.5):
    """Compute TP1 EV at given P threshold. Returns (ev, n_trades)."""
    model.eval()
    all_cls_prob = []
    all_tp1_pred = []
    all_mfe = []
    all_sl = []
    all_sl_pred = []

    with torch.no_grad():
        for loader in test_loaders.values():
            for x, direction, q, mfe, sl, ttp, asset_id, tf_id, mae in loader:
                x = x.to(device)
                asset_id = asset_id.to(device)
                tf_id = tf_id.to(device)
                direction = direction.to(device)

                out = model(x, asset_ids=asset_id, tf_ids=tf_id, direction_ids=direction)
                all_cls_prob.append(torch.sigmoid(out[:, 0]).cpu())
                all_tp1_pred.append(out[:, 1].cpu())
                all_sl_pred.append(out[:, 4].cpu())
                all_mfe.append(mfe)
                all_sl.append(sl)

    cls_prob = torch.cat(all_cls_prob)
    tp1_pred = torch.cat(all_tp1_pred)
    sl_pred = torch.cat(all_sl_pred)
    mfe = torch.cat(all_mfe)
    sl = torch.cat(all_sl)

    sl_effective = torch.max(sl_pred, sl)

    take = cls_prob > threshold
    n_take = take.sum().item()
    if n_take == 0:
        return 0.0, 0

    mfe_taken = mfe[take]
    sl_taken = sl_effective[take]
    tp1_taken = tp1_pred[take]
    tp1_wr = (mfe_taken >= tp1_taken).float().mean().item() * 100
    avg_tp1 = tp1_taken.mean().item() * 100
    avg_sl = sl_taken.mean().item() * 100
    ev = (tp1_wr / 100) * avg_tp1 - (1 - tp1_wr / 100) * avg_sl
    return ev, n_take


def shuffle_feature_in_loaders(test_loaders, feature_idx):
    """Shuffle a specific feature column across all test samples in-place.

    Returns the original values so they can be restored.
    """
    originals = {}
    for tk, loader in test_loaders.items():
        ds = loader.dataset
        original = ds.data[:, feature_idx].copy()
        originals[tk] = original
        shuffled = original.copy()
        np.random.shuffle(shuffled)
        ds.data[:, feature_idx] = shuffled
    return originals


def restore_feature_in_loaders(test_loaders, feature_idx, originals):
    """Restore original feature values after shuffling."""
    for tk, loader in test_loaders.items():
        ds = loader.dataset
        ds.data[:, feature_idx] = originals[tk]


def main():
    args = sys.argv[1:]
    threshold = 0.5
    if "--threshold" in args:
        idx = args.index("--threshold")
        if idx + 1 < len(args):
            threshold = float(args[idx + 1])

    # Override sys.argv before importing train module (it parses argv at import time)
    sys.argv = [sys.argv[0], "4h", "1h", "15min"]

    print(f"Feature importance at P>{threshold}")
    print(f"Loading data...")

    from src.train_liq_range_sfp import load_data_set, N_FEATURES, WINDOW_BY_TF, device
    train_loaders, test_loaders, n_features = load_data_set()

    WINDOW = max(WINDOW_BY_TF.values())
    model = LiqRangeSFPClassifier(n_features=n_features, window=WINDOW, hidden=48).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    print(f"Model loaded from {MODEL_FILE}")

    # Baseline EV
    baseline_ev, baseline_n = compute_ev(model, test_loaders, threshold)
    print(f"\nBaseline: TP1 EV = {baseline_ev:+.3f}% ({baseline_n} trades at P>{threshold})")

    # Permutation importance for each feature
    results = []
    np.random.seed(42)

    for feat_idx in range(min(n_features, len(FEATURE_NAMES))):
        feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"feature_{feat_idx}"

        originals = shuffle_feature_in_loaders(test_loaders, feat_idx)
        shuffled_ev, shuffled_n = compute_ev(model, test_loaders, threshold)
        restore_feature_in_loaders(test_loaders, feat_idx, originals)

        importance = baseline_ev - shuffled_ev
        results.append((feat_name, feat_idx, baseline_ev, shuffled_ev, importance))

    # Sort by importance (largest degradation = most important)
    results.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'='*70}")
    print(f"Feature Importance Report (sorted by TP1 EV degradation)")
    print(f"{'='*70}")
    print(f"{'Feature':<30} {'Baseline':>10} {'Shuffled':>10} {'Importance':>12} {'Status'}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    dead_features = []
    for name, idx, base, shuffled, imp in results:
        if imp <= 0:
            status = "DEAD"
            dead_features.append(name)
        elif imp < 0.01:
            status = "weak"
        else:
            status = "OK"
        print(f"{name:<30} {base:>+10.3f}% {shuffled:>+10.3f}% {imp:>+12.4f}% {status}")

    if dead_features:
        print(f"\nFeatures with zero/negative importance ({len(dead_features)}):")
        for f in dead_features:
            print(f"  - {f}")
        print("\nConsider removing these features to reduce noise.")
    else:
        print(f"\nAll {n_features} features contribute positively.")


if __name__ == "__main__":
    main()
