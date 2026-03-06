"""Liq+Range+SFP detection and feature engineering pipeline.

Mirrors src/train_liq_range_sfp.py:build_features() exactly.
"""

import numpy as np
import pandas as pd

from src.labels.liq_sfp_labels import generate_labels
from src.labels.range_sfp_labels import detect_market_structure
from src.labels.three_tap_labels import compute_atr


def run_liq_range_sfp_detection(df: pd.DataFrame, tf_key: str):
    """Run Liq+Range+SFP label generation.

    Returns:
        actions: array of 0=no-trade, 1=long, 2=short
        swept_levels: entry prices for each signal bar
        signal_map: dict of bar_idx -> LiqRangeSFPSignal
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None

    actions, _quality, _tp, _sl, swept_levels, signal_map = generate_labels(
        highs, lows, closes, opens,
        volumes=volumes,
        tf_key=tf_key,
    )

    return actions, swept_levels, signal_map


def build_liq_range_sfp_features(
    df: pd.DataFrame,
    actions: np.ndarray,
    signal_map: dict,
    tf_hours: float,
    asset_id: float = 1.0,
):
    """Build 24 Liq+Range+SFP features.

    6 range + 6 liquidation + 6 SFP candle + 6 context.
    Mirrors src/train_liq_range_sfp.py:build_features() exactly.

    Returns:
        feat_values: float32 array (N-30, 24)
        actions_trimmed: actions with warmup dropped
        signal_map_shifted: signal_map with indices shifted by -30
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.ones(n)

    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    feat = pd.DataFrame()

    # --- Range features (6) ---
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_norm = np.zeros(n, dtype=np.float32)
    range_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth_range = np.zeros(n, dtype=np.float32)
    reclaim_strength_range = np.zeros(n, dtype=np.float32)

    # --- Liq features (6) ---
    n_liq_swept_norm = np.zeros(n, dtype=np.float32)
    weighted_liq_swept = np.zeros(n, dtype=np.float32)
    max_leverage_norm = np.zeros(n, dtype=np.float32)
    liq_cascade_depth = np.zeros(n, dtype=np.float32)
    liq_cluster_density = np.zeros(n, dtype=np.float32)
    n_swings_with_liq_norm = np.zeros(n, dtype=np.float32)

    # --- SFP candle features (6) ---
    body_ratio = np.zeros(n, dtype=np.float32)
    wick_ratio = np.zeros(n, dtype=np.float32)
    vol_spike = np.zeros(n, dtype=np.float32)
    close_position = np.zeros(n, dtype=np.float32)
    zone_sl_dist = np.zeros(n, dtype=np.float32)
    zone_tp_dist = np.zeros(n, dtype=np.float32)

    # Precompute 20-bar volume average
    vol_ma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i, sig in signal_map.items():
        r = sig.range_ref
        range_h = r.high - r.low
        entry = sig.swept_level

        # Range features
        range_height_pct[i] = sig.range_height_pct
        range_touches_norm[i] = min(sig.range_touches, 5) / 5.0
        range_concentration[i] = sig.range_concentration
        range_age[i] = sig.range_age
        sweep_depth_range[i] = sig.sweep_depth_range
        reclaim_strength_range[i] = sig.reclaim_strength_range

        # Liq features
        n_liq_swept_norm[i] = min(sig.n_liq_swept, 30) / 30.0
        weighted_liq_swept[i] = min(sig.weighted_liq_swept, 3.0) / 3.0
        max_leverage_norm[i] = sig.max_leverage_swept / 100.0
        local_atr = atr[i] if atr[i] > 0 else 1e-8
        liq_cascade_depth[i] = np.clip(sig.liq_cascade_depth / local_atr, 0, 5)
        liq_cluster_density[i] = sig.liq_cluster_density
        n_swings_with_liq_norm[i] = min(sig.n_swings_with_liq, 10) / 10.0

        # SFP candle features
        candle_range = highs[i] - lows[i]
        if candle_range > 0:
            body_ratio[i] = (closes[i] - opens[i]) / candle_range
            if sig.direction == 1:
                wick_ratio[i] = (r.support.top - lows[i]) / candle_range
                close_position[i] = (closes[i] - lows[i]) / candle_range
            else:
                wick_ratio[i] = (highs[i] - r.resistance.bottom) / candle_range
                close_position[i] = (highs[i] - closes[i]) / candle_range

        vol_spike[i] = volumes[i] / (vol_ma20[i] + 1e-8)

        # Zone SL/TP distances
        if entry > 0:
            if sig.direction == 1:
                zone_sl_dist[i] = (entry - r.support.bottom) / entry
                zone_tp_dist[i] = (r.resistance.top - entry) / entry
            else:
                zone_sl_dist[i] = (r.resistance.top - entry) / entry
                zone_tp_dist[i] = (entry - r.support.bottom) / entry

    feat["range_height_pct"] = range_height_pct
    feat["range_touches_norm"] = range_touches_norm
    feat["range_concentration"] = range_concentration
    feat["range_age"] = range_age
    feat["sweep_depth_range"] = sweep_depth_range
    feat["reclaim_strength_range"] = reclaim_strength_range

    feat["n_liq_swept_norm"] = n_liq_swept_norm
    feat["weighted_liq_swept"] = weighted_liq_swept
    feat["max_leverage_norm"] = max_leverage_norm
    feat["liq_cascade_depth"] = liq_cascade_depth
    feat["liq_cluster_density"] = liq_cluster_density
    feat["n_swings_with_liq"] = n_swings_with_liq_norm

    feat["body_ratio"] = body_ratio
    feat["wick_ratio"] = wick_ratio
    feat["vol_spike"] = vol_spike
    feat["close_position"] = close_position
    feat["zone_sl_dist"] = zone_sl_dist
    feat["zone_tp_dist"] = zone_tp_dist

    # --- Context features (6) ---
    feat["rsi"] = df["rsi"].values / 100.0 if "rsi" in df.columns else 0.5
    feat["trend_strength"] = ((df["Close"] - df["ema_21"]) / df["Close"]).values if "ema_21" in df.columns else 0.0
    feat["ms_alignment"] = np.zeros(n, dtype=np.float32)
    feat["ms_strength"] = ms_strength_arr

    for i, sig in signal_map.items():
        feat.at[i, "ms_alignment"] = sig.ms_alignment

    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_spike"] = feat["vol_spike"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth_range"] = feat["sweep_depth_range"].clip(0, 2.0)
    feat["reclaim_strength_range"] = feat["reclaim_strength_range"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_dist"] = feat["zone_sl_dist"].clip(0, 0.10)
    feat["zone_tp_dist"] = feat["zone_tp_dist"].clip(0, 0.15)

    return feat.values.astype(np.float32), actions, signal_map_shifted
