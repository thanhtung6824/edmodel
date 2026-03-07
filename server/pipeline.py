"""Liq+Range+SFP detection and feature engineering pipeline.

Mirrors src/train_liq_range_sfp.py:build_features() exactly.
"""

import numpy as np
import pandas as pd

from src.labels.liq_sfp_labels import generate_labels
from src.labels.range_sfp_labels import detect_market_structure
from src.labels.three_tap_labels import compute_atr


def _compute_rsi(closes, period=14):
    """Compute RSI from close prices array. Returns array of same length."""
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_gain = np.zeros_like(closes, dtype=np.float64)
    avg_loss = np.zeros_like(closes, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = 50.0  # warmup
    return rsi


def compute_htf_features(highs, lows, closes, n, resample_factor=4):
    """Compute 4 higher-timeframe alignment features by resampling 4:1.

    Returns (htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength) — 4 arrays of length n.
    """
    # Resample OHLC by taking groups of `resample_factor` bars
    rf = resample_factor
    n_groups = n // rf
    remainder = n - n_groups * rf

    htf_trend = np.zeros(n, dtype=np.float32)
    htf_rsi = np.zeros(n, dtype=np.float32) + 0.5  # default neutral
    htf_ms_direction = np.zeros(n, dtype=np.float32)
    htf_ms_strength = np.zeros(n, dtype=np.float32)

    if n_groups < 22:  # need at least 21 bars for EMA21
        return htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength

    # Build resampled arrays
    start = remainder  # skip partial first group
    rs_highs = np.array([highs[start + i * rf: start + (i + 1) * rf].max() for i in range(n_groups)])
    rs_lows = np.array([lows[start + i * rf: start + (i + 1) * rf].min() for i in range(n_groups)])
    rs_closes = np.array([closes[start + (i + 1) * rf - 1] for i in range(n_groups)])

    # EMA21 on resampled closes
    ema = np.zeros(n_groups, dtype=np.float64)
    ema[0] = rs_closes[0]
    alpha = 2.0 / 22.0
    for i in range(1, n_groups):
        ema[i] = alpha * rs_closes[i] + (1 - alpha) * ema[i - 1]

    # HTF trend: (close - EMA21) / close
    htf_trend_rs = np.clip((rs_closes - ema) / (rs_closes + 1e-8), -0.5, 0.5).astype(np.float32)

    # HTF RSI
    htf_rsi_rs = (_compute_rsi(rs_closes, period=14) / 100.0).astype(np.float32)

    # HTF market structure
    htf_ms_dir_rs, htf_ms_str_rs, _, _ = detect_market_structure(rs_highs, rs_lows, n=10)

    # Map resampled values back to original bars (forward-fill)
    for g in range(n_groups):
        bar_start = start + g * rf
        bar_end = start + (g + 1) * rf
        htf_trend[bar_start:bar_end] = htf_trend_rs[g]
        htf_rsi[bar_start:bar_end] = htf_rsi_rs[g]
        htf_ms_direction[bar_start:bar_end] = htf_ms_dir_rs[g]
        htf_ms_strength[bar_start:bar_end] = htf_ms_str_rs[g]

    # Fill remainder at the start with first group's values
    if remainder > 0 and n_groups > 0:
        htf_trend[:start] = htf_trend_rs[0]
        htf_rsi[:start] = htf_rsi_rs[0]
        htf_ms_direction[:start] = htf_ms_dir_rs[0]
        htf_ms_strength[:start] = htf_ms_str_rs[0]

    return htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength


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

    actions, _quality, _mfe, _sl, _ttp, swept_levels, signal_map, _mae = generate_labels(
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
    """Build 18 Liq+Range+SFP features.

    4 range + 3 liquidation + 2 SFP candle + 3 context
    + 3 range fingerprint + 1 direction + 2 HTF.
    Mirrors src/train_liq_range_sfp.py:build_features() exactly.

    Returns:
        feat_values: float32 array (N-30, 18)
        actions_trimmed: actions with warmup dropped
        signal_map_shifted: signal_map with indices shifted by -30
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    atr = compute_atr(highs, lows, closes, period=14)

    feat = pd.DataFrame()

    # --- Range features (4) ---
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth_range = np.zeros(n, dtype=np.float32)
    reclaim_strength_range = np.zeros(n, dtype=np.float32)

    # --- Liq features (3) ---
    weighted_liq_swept = np.zeros(n, dtype=np.float32)
    max_leverage_norm = np.zeros(n, dtype=np.float32)
    liq_cascade_depth = np.zeros(n, dtype=np.float32)

    # --- SFP candle features (2) ---
    wick_ratio = np.zeros(n, dtype=np.float32)
    zone_sl_dist = np.zeros(n, dtype=np.float32)

    for i, sig in signal_map.items():
        r = sig.range_ref
        entry = sig.swept_level

        # Range features
        range_height_pct[i] = sig.range_height_pct
        range_age[i] = sig.range_age
        sweep_depth_range[i] = sig.sweep_depth_range
        reclaim_strength_range[i] = sig.reclaim_strength_range

        # Liq features
        weighted_liq_swept[i] = min(sig.weighted_liq_swept, 3.0) / 3.0
        max_leverage_norm[i] = sig.max_leverage_swept / 100.0
        local_atr = atr[i] if atr[i] > 0 else 1e-8
        liq_cascade_depth[i] = np.clip(sig.liq_cascade_depth / local_atr, 0, 5)

        # SFP candle features
        candle_range = highs[i] - lows[i]
        if candle_range > 0:
            if sig.direction == 1:
                wick_ratio[i] = (r.support.top - lows[i]) / candle_range
            else:
                wick_ratio[i] = (highs[i] - r.resistance.bottom) / candle_range

        if entry > 0:
            if sig.direction == 1:
                zone_sl_dist[i] = (entry - r.support.bottom) / entry
            else:
                zone_sl_dist[i] = (r.resistance.top - entry) / entry

    feat["range_height_pct"] = range_height_pct
    feat["range_age"] = range_age
    feat["sweep_depth_range"] = sweep_depth_range
    feat["reclaim_strength_range"] = reclaim_strength_range

    feat["weighted_liq_swept"] = weighted_liq_swept
    feat["max_leverage_norm"] = max_leverage_norm
    feat["liq_cascade_depth"] = liq_cascade_depth

    feat["wick_ratio"] = wick_ratio
    feat["zone_sl_dist"] = zone_sl_dist

    # --- Context features (3) ---
    feat["trend_strength"] = ((df["Close"] - df["ema_21"]) / df["Close"]).values if "ema_21" in df.columns else 0.0
    feat["ms_alignment"] = np.zeros(n, dtype=np.float32)

    for i, sig in signal_map.items():
        feat.at[i, "ms_alignment"] = sig.ms_alignment

    feat["asset_id"] = asset_id

    # --- Range fingerprint features (3) ---
    is_recaptured_arr = np.zeros(n, dtype=np.float32)
    touch_symmetry_arr = np.zeros(n, dtype=np.float32)
    range_position_arr = np.zeros(n, dtype=np.float32)

    for i, sig in signal_map.items():
        is_recaptured_arr[i] = sig.is_recaptured
        touch_symmetry_arr[i] = sig.touch_symmetry
        range_position_arr[i] = sig.range_position

    feat["is_recaptured"] = is_recaptured_arr
    feat["touch_symmetry"] = touch_symmetry_arr
    feat["range_position"] = range_position_arr

    # --- Direction feature (1) ---
    direction_arr = np.zeros(n, dtype=np.float32)
    for i, sig in signal_map.items():
        direction_arr[i] = 1.0 if sig.direction == 1 else -1.0
    feat["direction_feat"] = direction_arr

    # --- HTF alignment features (2) ---
    htf_trend, htf_rsi, _, _ = compute_htf_features(highs, lows, closes, n)
    feat["htf_trend"] = htf_trend
    feat["htf_rsi"] = htf_rsi

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth_range"] = feat["sweep_depth_range"].clip(0, 2.0)
    feat["reclaim_strength_range"] = feat["reclaim_strength_range"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_dist"] = feat["zone_sl_dist"].clip(0, 0.10)

    return feat.values.astype(np.float32), actions, signal_map_shifted
