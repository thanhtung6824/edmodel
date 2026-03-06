"""Quick check: run liq+range+sfp model on recent_data.json — show ALL signals.

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
from server.pipeline import run_liq_range_sfp_detection, build_liq_range_sfp_features
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

# Detection + features
actions, swept_levels, signal_map = run_liq_range_sfp_detection(df, tf_key)
feat_values, actions_trimmed, signal_map_shifted = build_liq_range_sfp_features(
    df, actions, signal_map, tf_hours, asset_id=asset_id,
)

drop_n = 30
swept_trimmed = swept_levels[drop_n:]

total_signals = int((actions_trimmed != 0).sum())
print(f"Signals detected: {total_signals}")

# Load model + predict
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
scaled = scaler.transform(feat_values)

print(f"\n{'Timestamp':<26} {'Dir':>5} {'Entry':>10} {'P(win)':>8} {'Signal?':>8}")
print("-" * 65)

window = WINDOW_BY_TF.get(tf_key, 30)
found_any = False
for i in range(window - 1, len(feat_values)):
    action = int(actions_trimmed[i])
    if action == 0:
        continue

    x = scaled[i - window + 1 : i + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)
    with torch.no_grad():
        logit = model(x_t)
    prob = torch.sigmoid(logit).item()

    ts = df["timestamp"].iloc[drop_n + i].tz_convert("Asia/Ho_Chi_Minh")
    direction = "LONG" if action == 1 else "SHORT"
    entry = float(swept_trimmed[i])
    passed = prob >= MODEL_CONFIDENCE
    marker = "YES" if passed else "no"

    print(f"  {ts}  {direction:>5}  ${entry:>9,.2f}  {prob:>7.3f}  {marker:>7}")
    found_any = True

if not found_any:
    print("  No signals found in this data range")
