"""Validate Liq+Range+SFP model on recent BTC data from recent_data.json.

Auto-detects timeframe from bar intervals.

Usage:
    python validate_liq_range_sfp.py
"""

import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from ta import volume, volatility, trend, momentum

from src.labels.liq_sfp_labels import generate_labels
from src.labels.range_sfp_labels import detect_market_structure
from src.labels.three_tap_labels import compute_atr
from src.models.liq_range_sfp_model import LiqRangeSFPClassifier

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

from server.config import WINDOW_BY_TF

N_FEATURES = 24
MODEL_FILE = "best_model_liq_range_sfp.pth"


def load_recent_data():
    """Load CSV historical data + append recent_data.json for full context.

    Auto-detects timeframe from recent_data.json bar intervals.
    Appends recent bars to CSV tail so ranges have proper context.
    Returns (df, tf_key, tf_hours, recent_start_idx).
    """
    print("Loading recent_data.json...")
    with open("recent_data.json") as f:
        raw = json.load(f)

    rows = []
    for k in raw:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    recent_df = pd.DataFrame(rows)

    # Auto-detect timeframe
    if len(recent_df) > 1:
        interval_hours = (recent_df["timestamp"].iloc[1] - recent_df["timestamp"].iloc[0]).total_seconds() / 3600
    else:
        interval_hours = 4.0

    if interval_hours <= 0.3:
        tf_key, tf_hours, csv_suffix = "15m", 0.25, "15min"
    elif interval_hours <= 1.1:
        tf_key, tf_hours, csv_suffix = "1h", 1.0, "1h"
    else:
        tf_key, tf_hours, csv_suffix = "4h", 4.0, "4h"

    # Load CSV for historical context
    csv_file = f"data/btc_{csv_suffix}.csv"
    print(f"  Loading CSV context from {csv_file}...")
    csv_df = pd.read_csv(csv_file)
    csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"], utc=True)

    # Only keep tail for context (last 3000 bars is plenty)
    context_bars = 3000
    if len(csv_df) > context_bars:
        csv_df = csv_df.iloc[-context_bars:].reset_index(drop=True)

    # Find where recent data starts (after CSV ends)
    csv_end = csv_df["timestamp"].iloc[-1]
    recent_new = recent_df[recent_df["timestamp"] > csv_end].reset_index(drop=True)
    print(f"  CSV context: {len(csv_df)} bars (ends {csv_end})")
    print(f"  Recent new bars: {len(recent_new)} (after CSV)")

    # Combine: CSV context + recent new bars
    # Keep only OHLCV + timestamp columns from CSV
    csv_slim = csv_df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    combined = pd.concat([csv_slim, recent_new], ignore_index=True)

    # Record where recent data starts (for evaluation)
    recent_start_idx = len(csv_slim)

    # Compute indicators on combined data
    combined["obv"] = volume.on_balance_volume(close=combined["Close"], volume=combined["Volume"], fillna=True)
    combined["bb"] = volatility.bollinger_wband(close=combined["Close"], window=20, window_dev=2, fillna=True)
    combined["ema_21"] = trend.ema_indicator(close=combined["Close"], window=21, fillna=True)
    combined["rsi"] = momentum.rsi(close=combined["Close"], fillna=True)
    combined = combined.dropna().reset_index(drop=True)

    print(f"  Combined: {len(combined)} bars | TF: {tf_key}")
    print(f"  Range: {combined['timestamp'].iloc[0]} to {combined['timestamp'].iloc[-1]}")
    print(f"  Evaluating signals from idx {recent_start_idx} onward (recent data only)")
    return combined, tf_key, tf_hours, recent_start_idx


def run_pipeline(df, tf_key="4h", tf_hours=4.0):
    """Run label generation + feature engineering."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values

    actions, quality, tp_labels, sl_labels, swept_levels, signal_map = generate_labels(
        highs, lows, closes, opens, volumes=volumes, tf_key=tf_key,
    )

    # Build features (same as train_liq_range_sfp.py build_features)
    n = len(df)
    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    feat = pd.DataFrame()

    # Range features (6)
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_norm = np.zeros(n, dtype=np.float32)
    range_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth_range = np.zeros(n, dtype=np.float32)
    reclaim_strength_range = np.zeros(n, dtype=np.float32)

    # Liq features (6)
    n_liq_swept_norm = np.zeros(n, dtype=np.float32)
    weighted_liq_swept = np.zeros(n, dtype=np.float32)
    max_leverage_norm = np.zeros(n, dtype=np.float32)
    liq_cascade_depth = np.zeros(n, dtype=np.float32)
    liq_cluster_density = np.zeros(n, dtype=np.float32)
    n_swings_with_liq_norm = np.zeros(n, dtype=np.float32)

    # SFP candle features (6)
    body_ratio = np.zeros(n, dtype=np.float32)
    wick_ratio = np.zeros(n, dtype=np.float32)
    vol_spike = np.zeros(n, dtype=np.float32)
    close_position = np.zeros(n, dtype=np.float32)
    zone_sl_dist = np.zeros(n, dtype=np.float32)
    zone_tp_dist = np.zeros(n, dtype=np.float32)

    vol_ma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i, sig in signal_map.items():
        r = sig.range_ref
        entry = sig.swept_level

        range_height_pct[i] = sig.range_height_pct
        range_touches_norm[i] = min(sig.range_touches, 5) / 5.0
        range_concentration[i] = sig.range_concentration
        range_age[i] = sig.range_age
        sweep_depth_range[i] = sig.sweep_depth_range
        reclaim_strength_range[i] = sig.reclaim_strength_range

        n_liq_swept_norm[i] = min(sig.n_liq_swept, 30) / 30.0
        weighted_liq_swept[i] = min(sig.weighted_liq_swept, 3.0) / 3.0
        max_leverage_norm[i] = sig.max_leverage_swept / 100.0
        local_atr = atr[i] if atr[i] > 0 else 1e-8
        liq_cascade_depth[i] = np.clip(sig.liq_cascade_depth / local_atr, 0, 5)
        liq_cluster_density[i] = sig.liq_cluster_density
        n_swings_with_liq_norm[i] = min(sig.n_swings_with_liq, 10) / 10.0

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

    feat["rsi"] = df["rsi"].values / 100.0
    feat["trend_strength"] = ((df["Close"] - df["ema_21"]) / df["Close"]).values
    feat["ms_alignment"] = np.zeros(n, dtype=np.float32)
    feat["ms_strength"] = ms_strength_arr
    for i, sig in signal_map.items():
        feat.at[i, "ms_alignment"] = sig.ms_alignment
    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = 1.0

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]
    swept_levels = swept_levels[drop_n:]
    timestamps = df["timestamp"].values[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_spike"] = feat["vol_spike"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth_range"] = feat["sweep_depth_range"].clip(0, 2.0)
    feat["reclaim_strength_range"] = feat["reclaim_strength_range"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_dist"] = feat["zone_sl_dist"].clip(0, 0.10)
    feat["zone_tp_dist"] = feat["zone_tp_dist"].clip(0, 0.15)

    print(f"  Features: {feat.shape[1]} columns")
    return feat.values.astype(np.float32), actions, quality, tp_labels, sl_labels, swept_levels, timestamps, signal_map_shifted


def predict_signals(feat_values, actions, model, recent_start_idx=0, tf_key="4h"):
    """Run model on signal bars in recent portion only."""
    window = WINDOW_BY_TF.get(tf_key, 30)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_values)

    # Only evaluate signals in the recent data portion
    min_idx = max(window - 1, recent_start_idx)
    signal_indices = [i for i in range(min_idx, len(actions)) if actions[i] != 0]
    print(f"  Signal bars in recent data: {len(signal_indices)} (window={window})")

    if not signal_indices:
        return [], np.array([])

    model.eval()
    all_probs = []
    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(signal_indices), batch_size):
            batch_idx = signal_indices[start:start + batch_size]
            batch_x = np.stack([scaled[i - window + 1:i + 1] for i in batch_idx])
            x_t = torch.FloatTensor(batch_x).to(device)
            logit = model(x_t)
            probs = torch.sigmoid(logit)
            all_probs.extend(probs.cpu().numpy().tolist())

    return signal_indices, np.array(all_probs)


def analyze(signal_indices, probs, actions, quality, tp_labels, sl_labels, swept_levels, timestamps, signal_map):
    """Analyze predictions on recent data."""
    n_signals = len(signal_indices)
    if n_signals == 0:
        print("\nNo signals detected in recent data.")
        return

    n_wins = sum(quality[i] == 1 for i in signal_indices)
    base_wr = n_wins / n_signals * 100

    print(f"\n{'='*80}")
    print(f"LIQ+RANGE+SFP VALIDATION — Recent BTC Data")
    print(f"{'='*80}")
    print(f"Total signals: {n_signals} | Base win rate: {base_wr:.0f}% ({n_wins}W / {n_signals - n_wins}L)")

    # Threshold analysis
    print(f"\n{'Thresh':>8} | {'Trades':>6} | {'Wins':>5} | {'WR%':>6} | {'AvgTP':>7} | {'AvgSL':>7} | {'R:R':>5} | {'EV/trade':>9}")
    print("-" * 75)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        take_indices = [signal_indices[j] for j in range(n_signals) if probs[j] > thresh]
        n_take = len(take_indices)
        if n_take == 0:
            print(f"  P>{thresh}: 0 trades")
            continue
        n_win = sum(quality[i] == 1 for i in take_indices)
        wr = n_win / n_take * 100
        avg_tp = np.mean([tp_labels[i] for i in take_indices]) * 100
        avg_sl = np.mean([sl_labels[i] for i in take_indices]) * 100
        rr = avg_tp / (avg_sl + 1e-8)
        ev = (wr / 100) * avg_tp - (1 - wr / 100) * avg_sl
        print(f"  P>{thresh:.1f}  | {n_take:>6} | {n_win:>5} | {wr:>5.0f}% | {avg_tp:>6.2f}% | {avg_sl:>6.2f}% | {rr:>5.2f} | {ev:>+8.3f}%")

    # Detailed trade log
    print(f"\n{'='*80}")
    print(f"Trade Log (all signals)")
    print(f"{'='*80}")
    print(f"{'Timestamp':<22} | {'Dir':>5} | {'Entry':>10} | {'P(win)':>6} | {'TP%':>6} | {'SL%':>6} | {'R:R':>5} | {'Result':>6} | {'Liq':>4}")
    print("-" * 90)

    n_win_total = 0
    n_lose_total = 0
    balance = 10_000.0
    risk_pct = 0.01

    for j in range(n_signals):
        i = signal_indices[j]
        ts = pd.Timestamp(timestamps[i])
        direction = "LONG" if actions[i] == 1 else "SHORT"
        entry = swept_levels[i]
        prob = probs[j]
        tp = tp_labels[i] * 100
        sl = sl_labels[i] * 100
        rr = tp / (sl + 1e-8)
        won = quality[i] == 1
        result = "WIN" if won else "LOSE"

        sig = signal_map.get(i)
        n_liq = sig.n_liq_swept if sig else 0

        if won:
            n_win_total += 1
            actual_rr = min(rr, 3.0)
            balance *= (1 + risk_pct * actual_rr)
        else:
            n_lose_total += 1
            balance *= (1 - risk_pct)

        marker = " <--" if prob > 0.5 and won else (" !!!" if prob > 0.5 and not won else "")
        print(f"{str(ts):<22} | {direction:>5} | {entry:>10.2f} | {prob:>5.2f}  | {tp:>5.2f}% | {sl:>5.2f}% | {rr:>5.2f} | {result:>6} | {n_liq:>4}{marker}")

    total = n_win_total + n_lose_total
    if total > 0:
        pnl_pct = (balance - 10_000) / 10_000 * 100
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"  Trades: {total} | Wins: {n_win_total} | Losses: {n_lose_total}")
        print(f"  Win Rate: {n_win_total/total*100:.0f}%")
        print(f"  Account: $10,000 -> ${balance:,.0f} ({pnl_pct:+.1f}%)")
        print(f"  (1R = 1% risk, R:R capped at 3.0)")


def main():
    df, tf_key, tf_hours, recent_start_idx = load_recent_data()

    print(f"\nRunning Liq+Range+SFP pipeline ({tf_key})...")
    feat_values, actions, quality, tp_labels, sl_labels, swept_levels, timestamps, signal_map = run_pipeline(df, tf_key=tf_key, tf_hours=tf_hours)

    # Adjust recent_start_idx for warmup drop
    drop_n = 30
    recent_start_idx_adj = max(0, recent_start_idx - drop_n)

    window = WINDOW_BY_TF.get(tf_key, 30)
    print(f"\nLoading model from {MODEL_FILE}... (window={window} for {tf_key})")
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=window, hidden=32).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    print("\nRunning predictions (recent data only)...")
    signal_indices, probs = predict_signals(feat_values, actions, model, recent_start_idx=recent_start_idx_adj, tf_key=tf_key)

    analyze(signal_indices, probs, actions, quality, tp_labels, sl_labels, swept_levels, timestamps, signal_map)


if __name__ == "__main__":
    main()
