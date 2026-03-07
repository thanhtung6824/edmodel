"""Validate LiqRangeSFP model on 4h BTC data (MFE regression)."""

import json
import numpy as np
import pandas as pd
import torch
import joblib

from src.labels.liq_sfp_labels import generate_labels
from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
from server.pipeline import build_liq_range_sfp_features

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

TIMEFRAME = "4h"
TF_KEY = "4h"
TF_HOURS = 4.0
CUTOFF = "2024-01-01"
HORIZON = 18
WINDOW_BY_TF = {"15m": 120, "1h": 48, "4h": 30}
MODEL_FILE = "best_model_liq_range_sfp.pth"
SCALER_FILE = "liq_range_sfp_scaler.joblib"
N_FEATURES = 33
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_data():
    print("Loading 4h data...")
    df = pd.read_csv("data/btc_4h.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  4h bars: {len(df)}, range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    before = len(df)
    df = df[df["timestamp"] >= CUTOFF].reset_index(drop=True)
    print(f"  After cutoff {CUTOFF}: {len(df)} (filtered from {before})")
    print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def run_pipeline(df):
    """Run LiqRangeSFP detection and feature engineering."""
    actions, quality, mfe, sl_labels, ttp_labels, swept_levels, signal_map = generate_labels(
        df["High"].values, df["Low"].values,
        df["Close"].values, df["Open"].values,
        volumes=df["Volume"].values if "Volume" in df.columns else None,
        tf_key=TF_KEY,
    )

    feat_arr, actions_trimmed, signal_map_shifted = build_liq_range_sfp_features(
        df, actions, signal_map, TF_HOURS, asset_id=1.0,
    )

    drop_n = 30
    quality = quality[drop_n:]
    mfe = mfe[drop_n:]
    sl_labels = sl_labels[drop_n:]
    swept_levels = swept_levels[drop_n:]
    timestamps = df["timestamp"].values[drop_n:]

    total = int(np.sum(actions_trimmed != 0))
    n_approach = sum(1 for s in signal_map_shifted.values() if s.signal_type == 1)
    mask = actions_trimmed != 0
    n_prof = int(np.sum(mask & (quality == 1)))
    avg_mfe = float(np.mean(mfe[mask])) * 100 if mask.any() else 0
    print(f"  Signals: {total} ({total - n_approach} SFP + {n_approach} approach) | "
          f"profitable: {n_prof} ({n_prof/max(total,1)*100:.0f}%) | avg MFE: {avg_mfe:.2f}%")

    return feat_arr, actions_trimmed, quality, mfe, sl_labels, swept_levels, timestamps, signal_map_shifted


def predict_signals(model, scaler, feat_values, actions, signal_map_shifted):
    """Run model on signal bars, return (p_win, tp1_dist, tp2_dist) per bar."""
    TF_ID_MAP = {"15m": 0, "1h": 1, "4h": 2}
    window = WINDOW_BY_TF.get(TF_KEY, 30)
    scaled = scaler.transform(feat_values)

    a_id = torch.LongTensor([0]).to(device)  # btc = 0
    t_id = torch.LongTensor([TF_ID_MAP.get(TF_KEY, 0)]).to(device)

    preds = {}
    n = len(actions)
    model.eval()
    with torch.no_grad():
        for bar_idx in range(window - 1, n):
            if actions[bar_idx] == 0:
                continue
            if bar_idx not in signal_map_shifted:
                continue
            x = scaled[bar_idx - window + 1: bar_idx + 1]
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
            d_id = torch.LongTensor([actions[bar_idx]]).to(device)
            out = model(x_t, asset_ids=a_id, tf_ids=t_id, direction_ids=d_id).squeeze(0).cpu()  # (4,)
            p_win = torch.sigmoid(out[0]).item()
            tp1_dist = out[1].item()
            tp2_dist = out[2].item()
            preds[bar_idx] = (p_win, tp1_dist, tp2_dist)

    return preds


def analyze(actions, quality, mfe, sl_labels, swept_levels, timestamps, preds, signal_map_shifted):
    """Threshold analysis + detailed signal log using MFE regression."""
    signals = []
    for bar_idx, (p_win, tp1_dist, tp2_dist) in preds.items():
        action = actions[bar_idx]
        if action == 0:
            continue
        sig = signal_map_shifted.get(bar_idx)
        signals.append({
            "bar_idx": bar_idx,
            "p_win": p_win, "tp1_dist": tp1_dist, "tp2_dist": tp2_dist,
            "quality": quality[bar_idx],
            "mfe": mfe[bar_idx],
            "sl": sl_labels[bar_idx],
            "action": action,
            "swept": swept_levels[bar_idx],
            "signal_type": sig.signal_type if sig else 0,
        })

    n_total = len(signals)
    if n_total == 0:
        print("\nNo signals to evaluate.")
        return

    n_sfp = sum(1 for s in signals if s["signal_type"] == 0)
    n_approach = sum(1 for s in signals if s["signal_type"] == 1)

    print(f"\n{'='*80}")
    print(f"Signal Analysis — {TIMEFRAME} BTC data (from {CUTOFF})")
    print(f"{'='*80}")
    n_prof = sum(1 for s in signals if s["quality"] == 1)
    avg_mfe_val = np.mean([s["mfe"] for s in signals]) * 100
    avg_sl_val = np.mean([s["sl"] for s in signals]) * 100
    print(f"Total: {n_total} ({n_sfp} SFP + {n_approach} approach) | "
          f"profitable: {n_prof} ({n_prof/n_total*100:.0f}%) | avg MFE: {avg_mfe_val:.2f}% | avg SL: {avg_sl_val:.2f}%\n")

    # Threshold analysis
    print(f"  {'Thresh':>8} | {'Trades':>6} | {'GateWR':>7} | {'TP1WR':>6} | {'TP2WR':>6} | {'TP1%':>6} | {'TP2%':>6} | {'SL%':>6} | {'EV1%':>8} | {'TotalR':>8} | {'$10K->':>10}")
    print(f"  {'-'*105}")

    for thresh in THRESHOLDS:
        filtered = [s for s in signals if s["p_win"] >= thresh]
        n_filt = len(filtered)
        if n_filt == 0:
            continue

        gate_wr = sum(1 for s in filtered if s["quality"] == 1) / n_filt * 100
        tp1_wr = sum(1 for s in filtered if s["mfe"] >= s["tp1_dist"]) / n_filt * 100
        tp2_wr = sum(1 for s in filtered if s["mfe"] >= s["tp2_dist"]) / n_filt * 100
        avg_tp1 = np.mean([s["tp1_dist"] for s in filtered]) * 100
        avg_tp2 = np.mean([s["tp2_dist"] for s in filtered]) * 100
        avg_sl = np.mean([s["sl"] for s in filtered]) * 100
        ev1 = (tp1_wr / 100) * avg_tp1 - (1 - tp1_wr / 100) * avg_sl

        total_r = 0.0
        balance = 10_000.0
        for s in filtered:
            r = s["tp1_dist"] / (s["sl"] + 1e-8)
            if s["mfe"] >= s["tp1_dist"]:
                total_r += r
                balance *= (1 + 0.01 * r)
            else:
                total_r -= 1.0
                balance *= (1 - 0.01)

        print(f"  {thresh:>8.1f} | {n_filt:>6} | {gate_wr:>6.1f}% | {tp1_wr:>5.1f}% | {tp2_wr:>5.1f}% | {avg_tp1:>5.2f}% | {avg_tp2:>5.2f}% | {avg_sl:>5.2f}% | {ev1:>+7.3f}% | {total_r:>+7.1f}R | ${balance:>9,.0f}")
    print()

    # Detailed log at P(win)>0.5
    target_thresh = 0.5
    filtered = [s for s in signals if s["p_win"] >= target_thresh]

    print(f"{'='*80}")
    print(f"Detailed signals at P(win) > {target_thresh}")
    print(f"{'='*80}")
    print(f"{'Timestamp':<22} | {'Dir':>5} | {'Type':>8} | {'Entry':>10} | {'P(win)':>6} | {'TP1%':>5} {'TP2%':>5} | {'MFE%':>5} | {'SL%':>5} | {'Result':>6}")
    print("-" * 100)

    n_win, n_lose = 0, 0
    total_pnl = 0.0
    for s in filtered:
        ts = pd.Timestamp(timestamps[s["bar_idx"]])
        direction = "LONG" if s["action"] == 1 else "SHORT"
        sig_type = "approach" if s["signal_type"] == 1 else "sfp"
        tp1_pct = s["tp1_dist"] * 100
        mfe_pct = s["mfe"] * 100
        act_sl = s["sl"] * 100
        won = s["mfe"] >= s["tp1_dist"]
        result = "WIN" if won else "LOSE"
        if won:
            n_win += 1
            total_pnl += tp1_pct
        else:
            n_lose += 1
            total_pnl -= act_sl

        print(f"{str(ts):<22} | {direction:>5} | {sig_type:>8} | {s['swept']:>10.2f} | {s['p_win']:>6.3f} | {tp1_pct:>4.2f}% {s['tp2_dist']*100:>4.2f}% | {mfe_pct:>4.2f}% | {act_sl:>4.2f}% | {result:>6}")

    total = n_win + n_lose
    if total > 0:
        print(f"\nSummary: {n_win} wins, {n_lose} losses | Win rate: {n_win/total*100:.0f}% | Cumulative P&L: {total_pnl:+.2f}%")


def save_signals(actions, quality, mfe, sl_labels, swept_levels, timestamps, preds, signal_map_shifted, threshold=0.5):
    """Save filtered signals to JSON (using P(win) >= threshold)."""
    signals = []
    for bar_idx, (p_win, tp1_dist, tp2_dist) in sorted(preds.items()):
        if p_win < threshold:
            continue
        action = actions[bar_idx]
        if action == 0:
            continue

        ts = pd.Timestamp(timestamps[bar_idx])
        unix_ms = int(ts.value // 10**6)
        entry = float(swept_levels[bar_idx])
        sig = signal_map_shifted.get(bar_idx)

        if action == 1:
            tp1_price = entry * (1 + tp1_dist)
            sl_price = entry * (1 - sl_labels[bar_idx])
        else:
            tp1_price = entry * (1 - tp1_dist)
            sl_price = entry * (1 + sl_labels[bar_idx])

        signals.append({
            "time_ms": unix_ms,
            "timestamp": str(ts),
            "dir": int(action),
            "entry": round(entry, 2),
            "tp_price": round(float(tp1_price), 2),
            "sl_price": round(float(sl_price), 2),
            "p_win": round(p_win, 4),
            "tp1_dist": round(tp1_dist, 6),
            "tp2_dist": round(tp2_dist, 6),
            "signal_type": sig.signal_type if sig else 0,
            "result": 1 if mfe[bar_idx] >= tp1_dist else 0,
        })

    output_file = f"signals_{TIMEFRAME}.json"
    with open(output_file, "w") as f:
        json.dump({"timeframe": TIMEFRAME, "threshold": threshold, "signals": signals}, f, indent=2)

    n_wins = sum(s["result"] == 1 for s in signals)
    print(f"\nSaved {len(signals)} signals to {output_file} (P(win) > {threshold})")
    print(f"  Wins: {n_wins} | Losses: {len(signals) - n_wins}")


def main():
    df = load_data()

    print(f"\nRunning LiqRangeSFP pipeline on {TIMEFRAME} data...")
    feat_arr, actions, quality, mfe, sl_labels, swept_levels, timestamps, signal_map_shifted = run_pipeline(df)

    print("\nLoading model + scaler...")
    window = max(WINDOW_BY_TF.values())
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=window, hidden=32).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    scaler = joblib.load(SCALER_FILE)
    print(f"  Model: {MODEL_FILE} | Scaler: {SCALER_FILE}")

    print("\nRunning predictions...")
    preds = predict_signals(model, scaler, feat_arr, actions, signal_map_shifted)

    analyze(actions, quality, mfe, sl_labels, swept_levels, timestamps, preds, signal_map_shifted)
    save_signals(actions, quality, mfe, sl_labels, swept_levels, timestamps, preds, signal_map_shifted)


main()
