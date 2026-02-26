"""Benchmark: log current model quality across all timeframes and assets.

Run this before making changes to establish a baseline.
Output saved to benchmark_results.txt

Usage:
    python benchmark.py                         # all assets with CSVs
    python benchmark.py --assets btc gold silver # specific assets
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from src.labels.sfp_labels import (
    detect_swings, build_swing_level_series, detect_sfp, compute_tp_sl_labels,
)
from src.models.sfp_transformer import SFPTransformer
from server.pipeline import run_sfp_detection, build_features as build_feat_shared

device = "cpu"
MODEL_FILE = "best_model_transformer.pth"
CUTOFF = "2024-01-01"
HORIZON = 18
WINDOW = 30
TF_MAP = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
THRESHOLDS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0, "symbol": "BTCUSDT"},
    "gold": {"prefix": "gold", "asset_id": 2.0, "symbol": "XAUUSDT"},
    "silver": {"prefix": "silver", "asset_id": 3.0, "symbol": "XAGUSDT"},
    "sol": {"prefix": "sol", "asset_id": 4.0, "symbol": "SOLUSDT"},
    "eth": {"prefix": "eth", "asset_id": 5.0, "symbol": "ETHUSDT"},
}

# Parse --assets flag
args = sys.argv[1:]
if "--assets" in args:
    idx = args.index("--assets")
    SELECTED_ASSETS = args[idx + 1:]
else:
    SELECTED_ASSETS = [
        name for name, cfg in ASSETS.items()
        if any(os.path.exists(f"data/{cfg['prefix']}_{tf}.csv") for tf in TF_MAP)
    ]

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(msg)


def run_sfp_pipeline(df, tf_hours, asset_id=1.0):
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    actions, swept_levels = run_sfp_detection(df)
    quality, tp_labels, sl_labels = compute_tp_sl_labels(highs, lows, closes, actions, swept_levels, horizon=HORIZON)

    feat_values, actions_trimmed = build_feat_shared(df, actions, tf_hours, asset_id=asset_id)

    drop_n = 30
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]
    swept_levels = swept_levels[drop_n:]
    timestamps = df["timestamp"].values[drop_n:]

    return feat_values, actions_trimmed, quality, tp_labels, sl_labels, swept_levels, timestamps


def predict_all(feat_values, model):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_values)
    all_indices = list(range(WINDOW - 1, len(feat_values)))
    all_tp, all_sl = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_indices), 256):
            batch_idx = all_indices[start:start + 256]
            batch_x = np.stack([scaled[i - WINDOW + 1:i + 1] for i in batch_idx])
            x_t = torch.FloatTensor(batch_x)
            tp_pred, sl_pred = model(x_t)
            all_tp.extend(tp_pred.numpy().tolist())
            all_sl.extend(sl_pred.numpy().tolist())
    return all_indices, np.array(all_tp), np.array(all_sl)


def benchmark_tf(tf, tf_hours, model, asset_name="btc", asset_id=1.0):
    prefix = ASSETS[asset_name]["prefix"]
    data_file = f"data/{prefix}_{tf}.csv"
    if not os.path.exists(data_file):
        log(f"\n  SKIPPED {asset_name}/{tf}: {data_file} not found")
        return None
    log(f"\n{'='*80}")
    log(f"  BENCHMARK: {tf} {asset_name.upper()} (from {CUTOFF})")
    log(f"{'='*80}")

    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    before = len(df)
    df = df[df["timestamp"] >= CUTOFF].reset_index(drop=True)
    log(f"  Data: {len(df)} bars (filtered from {before})")
    log(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    feat_values, actions, quality, tp_labels, sl_labels, swept_levels, timestamps = run_sfp_pipeline(df, tf_hours, asset_id=asset_id)

    total_sfp = int(np.sum(actions != 0))
    n_prof = int(np.sum((actions != 0) & (quality == 1)))
    base_rate = n_prof / total_sfp * 100 if total_sfp else 0
    log(f"  SFP signals: {total_sfp} | Base win rate: {base_rate:.0f}%")

    all_indices, tp_preds, sl_preds = predict_all(feat_values, model)
    ratio = tp_preds / (sl_preds + 1e-6)

    # Threshold analysis
    log(f"\n  {'Thresh':>6} | {'Trades':>6} | {'Wins':>5} | {'WinRate':>7} | {'AvgTP':>6} | {'AvgSL':>6} | {'R:R':>5} | {'EV/trade':>8}")
    log(f"  {'-'*70}")

    results = {}
    for thresh in THRESHOLDS:
        take = ratio > thresh
        flagged_indices = [all_indices[j] for j in range(len(all_indices)) if take[j]]
        sfp_flagged = [i for i in flagged_indices if actions[i] != 0]
        n_sfp = len(sfp_flagged)
        if n_sfp == 0:
            continue
        n_prof = sum(quality[i] == 1 for i in sfp_flagged)
        prec = n_prof / n_sfp * 100
        avg_tp = np.mean([tp_labels[i] for i in sfp_flagged]) * 100
        avg_sl = np.mean([sl_labels[i] for i in sfp_flagged]) * 100
        rr = avg_tp / avg_sl if avg_sl > 0 else 0
        ev = (prec / 100 * avg_tp) - ((100 - prec) / 100 * avg_sl)
        log(f"  {thresh:>6.1f} | {n_sfp:>6} | {n_prof:>5} | {prec:>6.0f}% | {avg_tp:>5.2f}% | {avg_sl:>5.2f}% | {rr:>5.2f} | {ev:>+7.3f}%")
        results[thresh] = {"trades": int(n_sfp), "wins": int(n_prof), "win_rate": round(float(prec), 1), "avg_tp": round(float(avg_tp), 2), "avg_sl": round(float(avg_sl), 2), "rr": round(float(rr), 2), "ev": round(float(ev), 3)}

    # Account simulation at ratio > 1.4
    target = 1.4
    take = ratio > target
    risk_pct = 0.01
    account = 10000.0
    n_win, n_lose = 0, 0
    total_r = 0.0
    for j in range(len(all_indices)):
        if not take[j]:
            continue
        i = all_indices[j]
        if actions[i] == 0:
            continue
        actual_rr = min(tp_labels[i] / (sl_labels[i] + 1e-8), 3.0)
        if quality[i] == 1:
            n_win += 1
            total_r += actual_rr
            account *= (1 + risk_pct * actual_rr)
        else:
            n_lose += 1
            total_r -= 1.0
            account *= (1 - risk_pct)

    total = n_win + n_lose
    if total > 0:
        log(f"\n  Account simulation (ratio > {target}, 1R = 1% risk):")
        log(f"    Trades: {total} | Wins: {n_win} | Win rate: {n_win/total*100:.0f}%")
        log(f"    Total R: {total_r:+.1f}R")
        log(f"    $10,000 -> ${account:,.0f} ({(account/10000-1)*100:+.1f}%)")

    return results


def main():
    log(f"SFP Transformer Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model: {MODEL_FILE}")
    log(f"Assets: {SELECTED_ASSETS}")
    log(f"Cutoff: {CUTOFF} | Horizon: {HORIZON} | Window: {WINDOW}")

    model = SFPTransformer(n_features=22)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    log("Model loaded")

    all_results = {}
    for asset_name in SELECTED_ASSETS:
        asset_cfg = ASSETS[asset_name]
        asset_id = asset_cfg["asset_id"]
        all_results[asset_name] = {}
        for tf, tf_hours in TF_MAP.items():
            try:
                result = benchmark_tf(tf, tf_hours, model, asset_name=asset_name, asset_id=asset_id)
                if result is not None:
                    all_results[asset_name][tf] = result
            except Exception as e:
                log(f"\n  SKIPPED {asset_name}/{tf}: {e}")

    # Summary
    log(f"\n{'='*80}")
    log(f"  SUMMARY (ratio > 1.4)")
    log(f"{'='*80}")
    log(f"  {'Asset':>6} {'TF':>6} | {'Trades':>6} | {'WinRate':>7} | {'R:R':>5} | {'EV/trade':>8}")
    log(f"  {'-'*55}")
    for asset_name in SELECTED_ASSETS:
        for tf in TF_MAP:
            if tf in all_results.get(asset_name, {}) and 1.4 in all_results[asset_name][tf]:
                r = all_results[asset_name][tf][1.4]
                log(f"  {asset_name:>6} {tf:>6} | {r['trades']:>6} | {r['win_rate']:>6.0f}% | {r['rr']:>5.2f} | {r['ev']:>+7.3f}%")

    # Save to benchmark/ directory
    os.makedirs("benchmark", exist_ok=True)
    tag = "_".join(SELECTED_ASSETS)
    out_file = f"benchmark/{tag}_results.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(output_lines))
    log(f"\nSaved to {out_file}")

    # Also save machine-readable JSON
    json_file = f"benchmark/{tag}_results.json"
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_FILE,
            "assets": SELECTED_ASSETS,
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            "window": WINDOW,
            "results": all_results,
        }, f, indent=2)
    log(f"Saved to {json_file}")


main()
