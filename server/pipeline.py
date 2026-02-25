"""SFP detection and 22-feature engineering pipeline.

Extracted from src/train_transformer.py:build_features() and
src/validate_4h.py:run_sfp_pipeline() to avoid duplication.
"""

import numpy as np
import pandas as pd

from src.labels.sfp_labels import (
    detect_swings,
    build_swing_level_series,
    compute_swing_level_info,
    detect_sfp,
)


def run_sfp_detection(df: pd.DataFrame):
    """Run SFP detection with n=5 and n=10, merge results.

    Extracted from validate_4h.py:40-73.

    Returns:
        actions: array of 0=no-trade, 1=long, 2=short
        swept_levels: the swing level price that triggered each SFP
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    reclaim_windows = {5: 1, 10: 3}
    results = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl, _, _ = build_swing_level_series(
            highs, lows, sh, sl, n, max_age=150
        )
        actions, swept = detect_sfp(
            highs, lows, closes, opens, active_sh, active_sl,
            reclaim_window=reclaim_windows[n],
        )
        results[n] = (actions, swept)

    # Merge: agreement keeps, conflict → no-trade
    actions_5, swept_5 = results[5]
    actions_10, swept_10 = results[10]
    actions = np.zeros(len(highs), dtype=np.int64)
    swept_levels = np.zeros(len(highs), dtype=np.float64)
    for i in range(len(highs)):
        a5, a10 = actions_5[i], actions_10[i]
        if a5 == a10:
            actions[i] = a5
            swept_levels[i] = swept_5[i] if swept_5[i] > 0 else swept_10[i]
        elif a5 != 0 and a10 == 0:
            actions[i] = a5
            swept_levels[i] = swept_5[i]
        elif a10 != 0 and a5 == 0:
            actions[i] = a10
            swept_levels[i] = swept_10[i]

    return actions, swept_levels


def build_features(df: pd.DataFrame, actions: np.ndarray, tf_hours: float, asset_id: float = 1.0):
    """Build 22 features for one timeframe's data.

    Extracted from src/train_transformer.py:79-168.
    Computes all 22 features in exact column order, applies warmup drop (30 bars),
    clipping, and fillna.

    Returns:
        feat_values: float32 array of shape (N-30, 22)
        actions_trimmed: actions array with warmup dropped
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    swing_levels = {}
    swing_data = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl, active_sh_ages, active_sl_ages = build_swing_level_series(
            highs, lows, sh, sl, n, max_age=150
        )
        nearest_sh = np.array([levels[0] if levels else np.nan for levels in active_sh])
        nearest_sl = np.array([levels[0] if levels else np.nan for levels in active_sl])
        swing_levels[n] = (nearest_sh, nearest_sl)
        swing_data[n] = (active_sh, active_sl, active_sh_ages, active_sl_ages)

    prev_close = df["Close"].shift(1)
    feat = pd.DataFrame()

    feat["Open"] = df["Open"] / prev_close - 1
    feat["High"] = df["High"] / prev_close - 1
    feat["Low"] = df["Low"] / prev_close - 1
    feat["Close"] = df["Close"] / prev_close - 1
    feat["rsi"] = df["rsi"] / 100.0

    vol_avg_20 = df["Volume"].rolling(20).mean()
    feat["vol_rel_20"] = df["Volume"] / (vol_avg_20 + 1e-8)

    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range_safe
    feat["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range_safe

    for n in [5, 10]:
        recent_sh, recent_sl = swing_levels[n]
        sweep_below = np.maximum(0, recent_sl - lows) / (closes + 1e-8)
        sweep_above = np.maximum(0, highs - recent_sh) / (closes + 1e-8)
        feat[f"sweep_below_{n}"] = sweep_below
        feat[f"sweep_above_{n}"] = sweep_above

    direction_feat = np.zeros(len(df), dtype=np.float32)
    direction_feat[actions == 1] = 1.0
    direction_feat[actions == 2] = -1.0
    feat["direction"] = direction_feat

    feat["trend_strength"] = (df["Close"] - df["ema_21"]) / df["Close"]
    feat["bb_width"] = df["bb"] / 100.0

    obv = df["obv"]
    obv_shifted = obv.shift(10)
    feat["obv_slope"] = (obv - obv_shifted) / (obv_shifted.abs() + 1e-8)

    ash, asl, ash_ages, asl_ages = swing_data[5]
    nearest_age, level_confluence = compute_swing_level_info(
        closes, ash, asl, ash_ages, asl_ages, max_age=150
    )
    feat["swing_level_age"] = nearest_age
    feat["level_confluence"] = level_confluence

    reclaim_dist = np.zeros(len(df), dtype=np.float32)
    nearest_sh_5, nearest_sl_5 = swing_levels[5]
    for i in range(len(actions)):
        if actions[i] == 1 and not np.isnan(nearest_sl_5[i]):
            reclaim_dist[i] = (closes[i] - nearest_sl_5[i]) / (closes[i] + 1e-8)
        elif actions[i] == 2 and not np.isnan(nearest_sh_5[i]):
            reclaim_dist[i] = (nearest_sh_5[i] - closes[i]) / (closes[i] + 1e-8)
    feat["reclaim_distance"] = reclaim_dist

    # Context features
    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions_trimmed = actions[drop_n:]

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    for n in [5, 10]:
        feat[f"sweep_below_{n}"] = feat[f"sweep_below_{n}"].clip(0, 0.05)
        feat[f"sweep_above_{n}"] = feat[f"sweep_above_{n}"].clip(0, 0.05)
    feat["obv_slope"] = feat["obv_slope"].clip(-5.0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["reclaim_distance"] = feat["reclaim_distance"].clip(0, 0.05)

    return feat.values.astype(np.float32), actions_trimmed
