"""Quick check: run liq+range+sfp model on recent_data.json — show ALL signals.

Shows model P(win) gate, predicted TP1/TP2 distances, and actual MFE outcome.
Deduplicates signals from the same range (same direction + SL targets).

Usage:
    python check_signal.py                  # default: btc, 4h
    python check_signal.py --asset gold
    python check_signal.py --tf 1h
    python check_signal.py --asset sol --tf 15m
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
from ta import volume, volatility, trend, momentum
from src.labels.liq_sfp_labels import generate_labels
from server.pipeline import build_liq_range_sfp_features
from server.inference import load_model, load_scaler
from server.config import MODEL_PATH, SCALER_PATH, WINDOW_BY_TF, MODEL_CONFIDENCE

ASSETS = {
    "btc": 1.0,
    "gold": 2.0,
    "silver": 3.0,
    "sol": 4.0,
    "eth": 5.0,
}

TF_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0}

args = sys.argv[1:]
asset_name = "btc"
tf_key = "4h"
if "--asset" in args:
    idx = args.index("--asset")
    asset_name = args[idx + 1]
if "--tf" in args:
    idx = args.index("--tf")
    tf_key = args[idx + 1]
asset_id = ASSETS.get(asset_name, 1.0)
tf_hours = TF_HOURS.get(tf_key, 4.0)

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
df = pd.DataFrame(rows)

df["obv"] = volume.on_balance_volume(close=df["Close"], volume=df["Volume"], fillna=True)
df["bb"] = volatility.bollinger_wband(close=df["Close"], window=20, window_dev=2, fillna=True)
df["ema_21"] = trend.ema_indicator(close=df["Close"], window=21, fillna=True)
df["rsi"] = momentum.rsi(close=df["Close"], fillna=True)
df = df.dropna().reset_index(drop=True)

print(f"Asset: {asset_name.upper()}  TF: {tf_key}  Candles: {len(df)}")
print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Detection + labels (7-tuple: actions, quality, mfe, sl_labels, ttp_labels, swept_levels, signal_map)
highs = df["High"].values
lows = df["Low"].values
closes = df["Close"].values
opens = df["Open"].values
volumes_arr = df["Volume"].values if "Volume" in df.columns else None

actions, quality, mfe, sl_labels, ttp_labels, swept_levels, signal_map = generate_labels(
    highs, lows, closes, opens, volumes=volumes_arr, tf_key=tf_key,
)

# Build features
feat_values, actions_trimmed, signal_map_shifted = build_liq_range_sfp_features(
    df, actions, signal_map, tf_hours, asset_id=asset_id,
)

drop_n = 30
swept_trimmed = swept_levels[drop_n:]
quality_trimmed = quality[drop_n:]
mfe_trimmed = mfe[drop_n:]
sl_trimmed = sl_labels[drop_n:]

total_signals = int((actions_trimmed != 0).sum())
print(f"Signals detected (raw): {total_signals}")

# Load model + predict
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
scaled = scaler.transform(feat_values)

# Collect all signals first, then dedup
all_signals = []
window = WINDOW_BY_TF.get(tf_key, 30)
for i in range(window - 1, len(feat_values)):
    action = int(actions_trimmed[i])
    if action == 0:
        continue

    x = scaled[i - window + 1 : i + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)
    ASSET_ID_MAP = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
    TF_ID_MAP = {"15m": 0, "1h": 1, "4h": 2}
    a_id = torch.LongTensor([ASSET_ID_MAP.get(asset_id, 0)])
    t_id = torch.LongTensor([TF_ID_MAP.get(tf_key, 0)])
    d_id = torch.LongTensor([action])
    with torch.no_grad():
        out = model(x_t, asset_ids=a_id, tf_ids=t_id, direction_ids=d_id).squeeze(0)  # (4,)
    p_win = torch.sigmoid(out[0]).item()
    tp1_dist = out[1].item()
    tp2_dist = out[2].item()

    ts = df["timestamp"].iloc[drop_n + i].tz_convert("Asia/Ho_Chi_Minh")
    entry = float(swept_trimmed[i])
    sl_pct = float(sl_trimmed[i])
    actual_mfe = float(mfe_trimmed[i])
    is_profitable = int(quality_trimmed[i]) == 1

    if action == 1:  # LONG
        tp1_price = entry * (1 + tp1_dist)
        tp2_price = entry * (1 + tp2_dist)
        sl_price = entry * (1 - sl_pct)
    else:  # SHORT
        tp1_price = entry * (1 - tp1_dist)
        tp2_price = entry * (1 - tp2_dist)
        sl_price = entry * (1 + sl_pct)

    # Outcome: did MFE reach predicted TP?
    hit_tp1 = actual_mfe >= tp1_dist
    hit_tp2 = actual_mfe >= tp2_dist

    if hit_tp2:
        result = "TP2"
    elif hit_tp1:
        result = "TP1"
    elif is_profitable:
        result = "MFE+"
    else:
        result = "SL"

    all_signals.append({
        "ts": ts, "action": action, "entry": entry,
        "p_win": p_win, "tp1_dist": tp1_dist, "tp2_dist": tp2_dist,
        "tp1_price": tp1_price, "tp2_price": tp2_price,
        "sl_price": sl_price, "sl_pct": sl_pct,
        "mfe": actual_mfe, "result": result,
        "passed": p_win >= MODEL_CONFIDENCE,
    })

# Dedup: group by (direction, sl_price_rounded, entry_rounded)
# Keep the signal with highest P(win) per unique trade setup
deduped = {}
for s in all_signals:
    key = (s["action"], round(s["entry"], 0), round(s["sl_price"], 0))
    if key not in deduped or s["p_win"] > deduped[key]["p_win"]:
        deduped[key] = s

signals = sorted(deduped.values(), key=lambda s: s["ts"])

print(f"Signals after dedup: {len(signals)} (from {len(all_signals)} raw)")
print()

print(f"{'Timestamp':<26} {'Dir':>5} {'Entry':>10} | {'P(win)':>7} {'TP1%':>6} {'TP2%':>6} | {'TP1':>10} {'TP2':>10} {'SL':>10} | {'MFE%':>6} {'Result':>6} {'Pass?':>5}")
print("-" * 130)

n_win, n_lose, n_pass_win, n_pass_lose = 0, 0, 0, 0
for s in signals:
    direction = "LONG" if s["action"] == 1 else "SHORT"
    marker = "YES" if s["passed"] else "no"
    won = s["result"] != "SL"

    if won:
        n_win += 1
    else:
        n_lose += 1
    if s["passed"]:
        if won:
            n_pass_win += 1
        else:
            n_pass_lose += 1

    print(
        f"  {s['ts']}  {direction:>5}  ${s['entry']:>9,.2f} |"
        f" {s['p_win']:>6.3f}  {s['tp1_dist']*100:>5.2f}  {s['tp2_dist']*100:>5.2f} |"
        f" ${s['tp1_price']:>9,.2f} ${s['tp2_price']:>9,.2f} ${s['sl_price']:>9,.2f} |"
        f" {s['mfe']*100:>5.2f}  {s['result']:>5} {marker:>5}"
    )

total = n_win + n_lose
n_passed = n_pass_win + n_pass_lose
print()
print(f"All signals: {n_win}W / {n_lose}L = {n_win/max(total,1)*100:.0f}% WR ({total} total)")
if n_passed > 0:
    print(f"Passed only: {n_pass_win}W / {n_pass_lose}L = {n_pass_win/max(n_passed,1)*100:.0f}% WR ({n_passed} total)")
