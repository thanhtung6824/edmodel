"""Quick check: run model on recent_data.json — show ALL signals."""

import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from ta import volume, volatility, trend, momentum
from server.pipeline import run_sfp_detection, build_features
from server.inference import load_model

WINDOW = 30
THRESHOLD = 1.4

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

print(f"Candles: {len(df)}")
print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

actions, swept_levels = run_sfp_detection(df)
feat_values, actions_trimmed = build_features(df, actions, tf_hours=4.0)

# Align swept_levels with trimmed actions
drop_n = 30
swept_trimmed = swept_levels[drop_n:]

total_sfp = int((actions_trimmed != 0).sum())
print(f"SFPs detected: {total_sfp}")

# Load model, scale features, run on ALL SFP bars
model = load_model("best_model_transformer.pth")
scaler = StandardScaler()
scaled = scaler.fit_transform(feat_values)

print(f"\n{'Timestamp':<26} {'Dir':>5} {'Entry':>10} {'Ratio':>6} {'Signal?':>8}")
print("-" * 65)

found_any = False
for i in range(WINDOW - 1, len(feat_values)):
    action = int(actions_trimmed[i])
    if action == 0:
        continue

    x = scaled[i - WINDOW + 1 : i + 1]
    x_t = torch.FloatTensor(x).unsqueeze(0)
    with torch.no_grad():
        tp_pred, sl_pred = model(x_t)
    tp = tp_pred.item()
    sl = sl_pred.item()
    ratio = tp / (sl + 1e-6)

    ts = df["timestamp"].iloc[drop_n + i]
    direction = "LONG" if action == 1 else "SHORT"
    entry = swept_trimmed[i]
    passed = ratio > THRESHOLD
    marker = "YES" if passed else "no"

    print(f"  {ts}  {direction:>5}  ${entry:>9,.2f}  {ratio:>5.2f}  {marker:>7}")
    found_any = True

if not found_any:
    print("  No SFP bars found in this data range")
