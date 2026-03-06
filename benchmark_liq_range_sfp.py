"""Benchmark: Liq+Range+SFP strategy WITH model predictions.

Runs signal detection → feature engineering → model inference → account simulation.
Uses 2024 data as out-of-sample evaluation period.
Shows results at multiple confidence thresholds.

Usage:
    python benchmark_liq_range_sfp.py
    python benchmark_liq_range_sfp.py --assets btc gold
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from src.labels.liq_sfp_labels import generate_labels
from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
from server.inference import load_scaler
from server.pipeline import build_liq_range_sfp_features

CUTOFF = "2024-01-01"
TFS = ["15min", "1h", "4h"]
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
from server.config import WINDOW_BY_TF

HORIZON = 18
MODEL_FILE = "best_model_liq_range_sfp.pth"
SCALER_FILE = "liq_range_sfp_scaler.joblib"
N_FEATURES = 24
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0},
    "gold": {"prefix": "gold", "asset_id": 2.0},
    "sol": {"prefix": "sol", "asset_id": 4.0},
    "eth": {"prefix": "eth", "asset_id": 5.0},
}

args = sys.argv[1:]
if "--assets" in args:
    idx = args.index("--assets")
    SELECTED_ASSETS = args[idx + 1:]
else:
    SELECTED_ASSETS = list(ASSETS.keys())


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file {MODEL_FILE} not found. Train first with:")
        print(f"  python -m src.train_liq_range_sfp")
        sys.exit(1)

    window = max(WINDOW_BY_TF.values())
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=window, hidden=32)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    print(f"Loaded model from {MODEL_FILE}")
    return model


def predict_signals(model, feat_values, actions, signal_map_shifted, tf_key="4h"):
    """Run model on all signal bars, return P(win) per signal bar index."""
    window = WINDOW_BY_TF.get(tf_key, 30)
    scaler = load_scaler(SCALER_FILE)
    scaled = scaler.transform(feat_values)

    probs = {}
    n = len(actions)
    for bar_idx in range(window - 1, n):
        if actions[bar_idx] == 0:
            continue
        if bar_idx not in signal_map_shifted:
            continue

        x = scaled[bar_idx - window + 1: bar_idx + 1]
        x_t = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            logit = model(x_t)
        probs[bar_idx] = torch.sigmoid(logit).item()

    return probs


def eval_with_model(actions, quality, tp_labels, sl_labels, probs, cutoff_idx, tf_label, tf_key):
    """Evaluate at multiple confidence thresholds."""
    drop_n = 30
    post_cutoff = max(0, cutoff_idx - drop_n)

    # Collect signal data
    signals = []
    for bar_idx, prob in probs.items():
        if bar_idx < post_cutoff:
            continue
        action = actions[bar_idx]
        if action == 0:
            continue
        signals.append({
            "bar_idx": bar_idx,
            "prob": prob,
            "quality": quality[bar_idx + drop_n],
            "tp": tp_labels[bar_idx + drop_n],
            "sl": sl_labels[bar_idx + drop_n],
            "action": action,
        })

    n_total = len(signals)
    if n_total == 0:
        print(f"  No signals after cutoff")
        return None

    # Base stats (no model filter)
    base_wins = sum(1 for s in signals if s["quality"] == 1)
    base_wr = base_wins / n_total * 100
    base_tp = np.mean([s["tp"] for s in signals]) * 100
    base_sl = np.mean([s["sl"] for s in signals]) * 100
    base_ev = (base_wr / 100) * base_tp - (1 - base_wr / 100) * base_sl

    print(f"\n  {tf_label}: {n_total} signals after {CUTOFF}")
    print(f"  Base (no filter): WR={base_wr:.1f}%, TP={base_tp:.2f}%, SL={base_sl:.2f}%, EV={base_ev:+.3f}%")

    # P(win) spread
    win_probs = [s["prob"] for s in signals if s["quality"] == 1]
    loss_probs = [s["prob"] for s in signals if s["quality"] == 0]
    if win_probs and loss_probs:
        print(f"  P(win) spread — winners: {np.mean(win_probs):.3f} vs losers: {np.mean(loss_probs):.3f}")

    # Threshold analysis
    print(f"\n  {'Thresh':>8} {'Trades':>8} {'WR%':>8} {'TP%':>8} {'SL%':>8} {'EV%':>8} {'TotalR':>8} {'$10K->':>12}")
    print(f"  {'-'*72}")

    results = []
    for thresh in THRESHOLDS:
        filtered = [s for s in signals if s["prob"] >= thresh]
        n_filt = len(filtered)
        if n_filt == 0:
            print(f"  {thresh:>8.1f} {'0':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>12}")
            continue

        wins = sum(1 for s in filtered if s["quality"] == 1)
        wr = wins / n_filt * 100
        avg_tp = np.mean([s["tp"] for s in filtered]) * 100
        avg_sl = np.mean([s["sl"] for s in filtered]) * 100
        ratios = [s["tp"] / (s["sl"] + 1e-8) for s in filtered]
        ev = (wr / 100) * avg_tp - (1 - wr / 100) * avg_sl

        # Account sim
        total_r = 0.0
        balance = 10_000.0
        for s in filtered:
            r = s["tp"] / (s["sl"] + 1e-8)
            if s["quality"] == 1:
                total_r += r
                balance *= (1 + 0.01 * r)
            else:
                total_r -= 1.0
                balance *= (1 - 0.01)

        print(f"  {thresh:>8.1f} {n_filt:>8} {wr:>7.1f}% {avg_tp:>7.2f}% {avg_sl:>7.2f}% {ev:>+7.3f}% {total_r:>+7.1f}R ${balance:>10,.0f}")

        results.append({
            "threshold": thresh,
            "n_signals": n_filt,
            "win_rate": round(wr, 1),
            "avg_tp": round(avg_tp, 2),
            "avg_sl": round(avg_sl, 2),
            "ev_per_trade": round(ev, 3),
            "total_r": round(total_r, 1),
            "final_balance": round(balance, 0),
        })

    return {
        "label": tf_label,
        "tf_key": tf_key,
        "n_signals_raw": n_total,
        "base_wr": round(base_wr, 1),
        "base_ev": round(base_ev, 3),
        "thresholds": results,
    }


def run_benchmark():
    print("=" * 70)
    print(f"LIQ+RANGE+SFP BENCHMARK WITH MODEL (cutoff: {CUTOFF})")
    print("=" * 70)

    model = load_model()
    all_results = []

    for asset_name in SELECTED_ASSETS:
        cfg = ASSETS.get(asset_name)
        if not cfg:
            continue

        print(f"\n{'='*70}")
        print(f"  {asset_name.upper()}")
        print(f"{'='*70}")

        for tf in TFS:
            tf_key = TF_KEYS[tf]
            tf_hours = TF_HOURS[tf]
            data_file = f"data/{cfg['prefix']}_{tf}.csv"
            if not os.path.exists(data_file):
                continue

            df = pd.read_csv(data_file).reset_index(drop=True)
            if "timestamp" in df.columns:
                mask = df["timestamp"] >= CUTOFF
                cutoff_idx = mask.idxmax() if mask.any() else 0
            else:
                cutoff_idx = int(len(df) * 0.8)

            label = f"{asset_name.upper()}/{tf_key}"

            actions, quality, tp_labels, sl_labels, swept_levels, signal_map = generate_labels(
                df["High"].values, df["Low"].values,
                df["Close"].values, df["Open"].values,
                volumes=df["Volume"].values if "Volume" in df.columns else None,
                tf_key=tf_key,
            )

            feat, actions_trimmed, signal_map_shifted = build_liq_range_sfp_features(
                df, actions, signal_map, tf_hours, asset_id=cfg["asset_id"],
            )

            feat_arr = feat.values if hasattr(feat, 'values') else feat
            probs = predict_signals(model, feat_arr.astype(np.float32), actions_trimmed, signal_map_shifted, tf_key=tf_key)

            result = eval_with_model(
                actions_trimmed, quality, tp_labels, sl_labels,
                probs, cutoff_idx, label, tf_key,
            )
            if result:
                all_results.append(result)

    # Save
    if all_results:
        os.makedirs("benchmark", exist_ok=True)
        out_file = "benchmark/liq_range_sfp_results.txt"
        with open(out_file, "w") as f:
            f.write(f"LIQ+RANGE+SFP BENCHMARK RESULTS\n")
            f.write(f"Model: {MODEL_FILE}  |  Cutoff: {CUTOFF}  |  Horizon: {HORIZON} bars\n")
            f.write(f"{'='*80}\n\n")

            for r in all_results:
                f.write(f"{r['label']}  ({r['n_signals_raw']} signals)\n")
                f.write(f"  Base: WR={r['base_wr']:.1f}%  EV={r['base_ev']:+.3f}%\n")
                f.write(f"  {'Thresh':>8} {'Trades':>8} {'WR%':>8} {'TP%':>8} {'SL%':>8} {'EV%':>8} {'TotalR':>8} {'$10K->':>12}\n")
                f.write(f"  {'-'*72}\n")
                for t in r["thresholds"]:
                    f.write(f"  {t['threshold']:>8.1f} {t['n_signals']:>8} {t['win_rate']:>7.1f}% {t['avg_tp']:>7.2f}% {t['avg_sl']:>7.2f}% {t['ev_per_trade']:>+7.3f}% {t['total_r']:>+7.1f}R ${t['final_balance']:>10,.0f}\n")
                f.write(f"\n")

            # Summary
            f.write(f"{'='*80}\n")
            f.write(f"SUMMARY (best threshold per asset/tf)\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Label':<16} {'Thresh':>8} {'Trades':>8} {'WR%':>8} {'EV%':>8} {'TotalR':>8} {'$10K->':>12}\n")
            f.write(f"{'-'*72}\n")
            for r in all_results:
                best = max(r["thresholds"], key=lambda t: t["ev_per_trade"]) if r["thresholds"] else None
                if best:
                    f.write(f"{r['label']:<16} {best['threshold']:>8.1f} {best['n_signals']:>8} {best['win_rate']:>7.1f}% {best['ev_per_trade']:>+7.3f}% {best['total_r']:>+7.1f}R ${best['final_balance']:>10,.0f}\n")

        print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
