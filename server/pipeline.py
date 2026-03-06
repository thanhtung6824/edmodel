"""SFP + Three-Tap + Range-SFP detection and feature engineering pipelines.

SFP: Extracted from src/train_transformer.py:build_features().
Three-Tap: Mirrors src/train_three_tap.py:build_features().
Range-SFP: Mirrors src/train_range_sfp.py:build_features().
"""

import numpy as np
import pandas as pd

from src.labels.sfp_labels import (
    detect_swings,
    build_swing_level_series,
    compute_swing_level_info,
    detect_sfp,
)
from src.labels.three_tap_labels import (
    generate_labels as three_tap_generate_labels,
    compute_atr,
    detect_ranges,
    RANGE_PARAMS,
)
from src.labels.range_sfp_labels import (
    generate_labels as range_sfp_generate_labels,
    detect_market_structure,
)
from src.labels.range_quality_labels import build_range_quality_features


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


def build_features(df: pd.DataFrame, actions: np.ndarray, tf_hours: float, asset_id: float = 1.0, tf_key: str = "4h"):
    """Build 23 features for one timeframe's data.

    Extracted from src/train_transformer.py.
    Computes all 23 features in exact column order, applies warmup drop (30 bars),
    clipping, and fillna.

    Returns:
        feat_values: float32 array of shape (N-30, 23)
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

    # Range boundary feature: is the swept level near a range high/low?
    at_range_boundary = np.zeros(len(df), dtype=np.float32)
    atr_vals = compute_atr(highs, lows, closes, period=14)
    params = RANGE_PARAMS.get(tf_key, RANGE_PARAMS["4h"])
    wp = params["wide"]
    _, wide_all = detect_ranges(
        highs, lows, closes, atr_vals,
        n_swing=wp["n_swing"], min_bars=wp["min_bars"], max_bars=wp["max_bars"],
        min_atr_mult=wp["min_atr_mult"], max_atr_mult=wp["max_atr_mult"],
        tolerance=0.01, touch_tolerance=0.005, min_touches=2,
    )
    n_bars = len(highs)
    active_ranges = [[] for _ in range(n_bars)]
    for r in wide_all:
        for j in range(r.confirmed, min(r.confirmed + 500, n_bars)):
            if highs[j] > r.high * 1.03 or lows[j] < r.low * 0.97:
                break
            active_ranges[j].append(r)
    nearest_sh_5_arr, nearest_sl_5_arr = swing_levels[5]
    for i in range(n_bars):
        if actions[i] == 0:
            continue
        swept = nearest_sl_5_arr[i] if actions[i] == 1 else nearest_sh_5_arr[i]
        if np.isnan(swept) or not active_ranges[i]:
            continue
        for r in active_ranges[i]:
            rh, rl = r.high, r.low
            range_h = rh - rl
            tol = range_h * 0.003
            if swept >= rh - tol or swept <= rl + tol:
                at_range_boundary[i] = 1.0
                break
    feat["at_range_boundary"] = at_range_boundary

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


# ─── Three-Tap Pipeline ─────────────────────────────────────────────


def run_three_tap_detection(df: pd.DataFrame, tf_key: str):
    """Run three-tap label generation with per-TF range detection.

    Returns:
        actions: array of 0=no-trade, 1=long, 2=short
        entry_levels: entry prices for each signal bar
        signal_zones: dict of bar_idx -> DemandZone with metadata
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    actions, _quality, _tp, _sl, entry_levels, signal_zones = three_tap_generate_labels(
        highs, lows, closes, opens,
        precomputed_ranges=None,
        tf_key=tf_key,
        require_mss=True,
        allow_multi_dev=False,
        mss_mode="soft",
    )

    return actions, entry_levels, signal_zones


def build_three_tap_features(
    df: pd.DataFrame,
    actions: np.ndarray,
    signal_zones: dict,
    tf_hours: float,
    asset_id: float = 1.0,
):
    """Build 18 three-tap features matching src/train_three_tap.py.

    Returns:
        feat_values: float32 array (N-30, 18)
        actions_trimmed: actions with warmup dropped
        signal_zones_shifted: signal_zones with indices shifted by -30
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values

    atr = compute_atr(highs, lows, closes, period=14)
    vol_avg_20 = df["Volume"].rolling(20).mean().values

    feat = pd.DataFrame()

    # --- 10 setup quality features (non-zero only at signal bars) ---
    range_height_atr = np.zeros(n, dtype=np.float32)
    range_touches = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    deviation_depth = np.zeros(n, dtype=np.float32)
    zone_width_pct = np.zeros(n, dtype=np.float32)
    structural_rr = np.zeros(n, dtype=np.float32)
    has_fvg = np.zeros(n, dtype=np.float32)
    dev_volume_ratio = np.zeros(n, dtype=np.float32)
    bars_since_dev = np.zeros(n, dtype=np.float32)
    mss_body_ratio = np.zeros(n, dtype=np.float32)

    for i, zone in signal_zones.items():
        range_h = zone._range_high - zone._range_low
        local_atr = atr[i] if atr[i] > 0 else 1e-8

        range_height_atr[i] = range_h / local_atr
        range_touches[i] = zone._range_touches / 10.0
        range_age[i] = (i - zone._range_confirmed) / 100.0

        if range_h > 0:
            if zone.direction == 1:
                deviation_depth[i] = (zone._range_low - zone.deviation_wick) / range_h
            else:
                deviation_depth[i] = (zone.deviation_wick - zone._range_high) / range_h

        zone_w = zone.top - zone.bottom
        zone_width_pct[i] = zone_w / (closes[i] + 1e-8)

        if zone.direction == 1:
            tp_dist = zone.tp_target - zone.top
            sl_dist = zone.top - zone.bottom
        else:
            tp_dist = zone.bottom - zone.tp_target
            sl_dist = zone.top - zone.bottom
        structural_rr[i] = tp_dist / (sl_dist + 1e-8)

        has_fvg[i] = 1.0 if zone._has_fvg else 0.0

        dev_bar = zone._deviation_bar
        if 0 <= dev_bar < n and vol_avg_20[dev_bar] > 0:
            dev_volume_ratio[i] = volumes[dev_bar] / (vol_avg_20[dev_bar] + 1e-8)

        bars_since_dev[i] = (i - zone._deviation_bar) / 30.0

        mss_bar = zone._mss_bar
        if 0 < mss_bar < n:
            candle_range = highs[mss_bar] - lows[mss_bar]
            if candle_range > 0:
                mss_body_ratio[i] = abs(closes[mss_bar] - opens[mss_bar]) / candle_range

    feat["range_height_atr"] = range_height_atr
    feat["range_touches"] = range_touches
    feat["range_age"] = range_age
    feat["deviation_depth"] = deviation_depth
    feat["zone_width_pct"] = zone_width_pct
    feat["structural_rr"] = structural_rr
    feat["has_fvg"] = has_fvg
    feat["dev_volume_ratio"] = dev_volume_ratio
    feat["bars_since_dev"] = bars_since_dev
    feat["mss_body_ratio"] = mss_body_ratio

    # --- 5 price context features ---
    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["rsi"] = df["rsi"] / 100.0
    feat["trend_strength"] = (df["Close"] - df["ema_21"]) / df["Close"]
    feat["vol_rel_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)
    feat["bb_width"] = df["bb"] / 100.0

    # --- Direction + context ---
    direction_feat = np.zeros(n, dtype=np.float32)
    direction_feat[actions == 1] = 1.0
    direction_feat[actions == 2] = -1.0
    feat["direction"] = direction_feat
    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions_trimmed = actions[drop_n:]
    signal_zones_shifted = {k - drop_n: v for k, v in signal_zones.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["structural_rr"] = feat["structural_rr"].clip(0, 10.0)
    feat["dev_volume_ratio"] = feat["dev_volume_ratio"].clip(0, 5.0)
    feat["range_height_atr"] = feat["range_height_atr"].clip(0, 15.0)

    return feat.values.astype(np.float32), actions_trimmed, signal_zones_shifted


# ─── Range-SFP Pipeline ─────────────────────────────────────────────


def filter_ranges_by_quality(all_ranges, df, model, tf_hours, asset_id, threshold=0.5):
    """Filter ranges using the range quality model.

    Returns:
        filtered_ranges: list of ZoneRange that passed the quality threshold
        quality_scores: dict of range id -> P(valid) score
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values

    atr = compute_atr(highs, lows, closes, period=14)
    ms_dir, ms_str, _, _ = detect_market_structure(highs, lows, n=10)
    rsi = df["rsi"].values if "rsi" in df.columns else np.full(len(highs), 50.0)

    features, _ = build_range_quality_features(
        highs, lows, closes, volumes, all_ranges,
        atr, ms_dir, ms_str, rsi, tf_hours, asset_id,
    )

    if len(features) == 0:
        return all_ranges, {}

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    import torch
    x_t = torch.FloatTensor(scaled)
    with torch.no_grad():
        logits = model(x_t)
    probs = torch.sigmoid(logits).numpy()

    filtered = []
    quality_scores = {}
    for i, r in enumerate(all_ranges):
        quality_scores[id(r)] = float(probs[i])
        if probs[i] >= threshold:
            filtered.append(r)

    return filtered, quality_scores


def run_range_sfp_detection(df: pd.DataFrame, tf_key: str, range_quality_model=None,
                            range_quality_threshold=0.5, tf_hours: float = 4.0,
                            asset_id: float = 1.0):
    """Run range-SFP label generation with optional range quality pre-filter.

    If range_quality_model is provided, filters out low-quality ranges
    before SFP boundary matching (post-filter on signal_map).

    Returns:
        actions: array of 0=no-trade, 1=long, 2=short
        swept_levels: entry prices for each signal bar
        signal_map: dict of bar_idx -> RangeSFPSignal
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    actions, _quality, _tp, _sl, swept_levels, signal_map, all_ranges, _active_per_bar = range_sfp_generate_labels(
        highs, lows, closes, opens,
        tf_key=tf_key,
    )

    # Apply range quality filter if model is available
    if range_quality_model is not None and signal_map:
        _, quality_scores = filter_ranges_by_quality(
            all_ranges, df, range_quality_model, tf_hours, asset_id,
            threshold=range_quality_threshold,
        )

        # Filter signal_map: remove signals whose range failed quality gate
        filtered_map = {}
        for bar_idx, sig in signal_map.items():
            r_id = id(sig.range_ref)
            if quality_scores.get(r_id, 0.0) >= range_quality_threshold:
                filtered_map[bar_idx] = sig
            else:
                actions[bar_idx] = 0
                swept_levels[bar_idx] = 0.0

        signal_map = filtered_map

    return actions, swept_levels, signal_map


def build_range_sfp_features(
    df: pd.DataFrame,
    actions: np.ndarray,
    signal_map: dict,
    tf_hours: float,
    asset_id: float = 1.0,
):
    """Build 18 range-SFP features matching src/train_range_sfp.py.

    Returns:
        feat_values: float32 array (N-30, 18)
        actions_trimmed: actions with warmup dropped
        signal_map_shifted: signal_map with indices shifted by -30
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    feat = pd.DataFrame()

    # --- Range context features (10) ---
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_high = np.zeros(n, dtype=np.float32)
    range_touches_low = np.zeros(n, dtype=np.float32)
    range_time_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth = np.zeros(n, dtype=np.float32)
    reclaim_strength = np.zeros(n, dtype=np.float32)
    position_in_range = np.zeros(n, dtype=np.float32)
    vol_compression = np.zeros(n, dtype=np.float32)
    ms_alignment = np.zeros(n, dtype=np.float32)

    for i, sig in signal_map.items():
        r = sig.range_ref
        range_h = r.high - r.low
        mid_price = (r.high + r.low) / 2.0

        range_height_pct[i] = range_h / (mid_price + 1e-8)
        range_touches_high[i] = r.touches_high / 5.0
        range_touches_low[i] = r.touches_low / 5.0
        range_time_concentration[i] = getattr(r, '_time_concentration', 0.8)
        range_age[i] = (i - r.confirmed) / 200.0
        sweep_depth[i] = sig.sweep_depth
        reclaim_strength[i] = sig.reclaim_strength
        if range_h > 0:
            position_in_range[i] = (closes[i] - r.low) / range_h
        vol_compression[i] = getattr(r, '_vol_compression', 1.0)
        ms_alignment[i] = sig.ms_alignment

    feat["range_height_pct"] = range_height_pct
    feat["range_touches_high"] = range_touches_high
    feat["range_touches_low"] = range_touches_low
    feat["range_time_concentration"] = range_time_concentration
    feat["range_age"] = range_age
    feat["sweep_depth"] = sweep_depth
    feat["reclaim_strength"] = reclaim_strength
    feat["position_in_range"] = position_in_range
    feat["vol_compression"] = vol_compression
    feat["ms_alignment"] = ms_alignment

    # --- Price context features (4) ---
    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["rsi"] = df["rsi"] / 100.0
    feat["vol_rel_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)
    feat["trend_strength"] = (df["Close"] - df["ema_21"]) / df["Close"]

    # --- Context features (4) ---
    feat["ms_strength"] = ms_strength_arr

    direction_feat = np.zeros(n, dtype=np.float32)
    direction_feat[actions == 1] = 1.0
    direction_feat[actions == 2] = -1.0
    feat["direction"] = direction_feat
    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions_trimmed = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth"] = feat["sweep_depth"].clip(0, 2.0)
    feat["reclaim_strength"] = feat["reclaim_strength"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)

    return feat.values.astype(np.float32), actions_trimmed, signal_map_shifted
