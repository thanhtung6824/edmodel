"""Benchmark: Liq+Range+SFP strategy WITH model predictions (MFE regression).

Runs signal detection -> feature engineering -> model inference -> account simulation.
Uses 2024 data as out-of-sample evaluation period.
Shows results at multiple confidence thresholds with model-predicted TP levels.

Usage:
    python benchmark_liq_range_sfp.py
    python benchmark_liq_range_sfp.py --assets btc gold
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

import joblib

from src.labels.liq_sfp_labels import generate_labels
from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
from server.pipeline import build_liq_range_sfp_features

CUTOFF = "2024-01-01"
TFS = ["15min", "1h", "4h"]
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
WINDOW_BY_TF = {"15m": 120, "1h": 48, "4h": 30}

HORIZON_BY_TF = {"15m": 36, "1h": 18, "4h": 18}
MODEL_FILE = "best_model_liq_range_sfp.pth"
SCALER_FILE = "liq_range_sfp_scaler.joblib"
N_FEATURES = 27
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
    model = LiqRangeSFPClassifier(n_features=N_FEATURES, window=window, hidden=48)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    print(f"Loaded model from {MODEL_FILE}")
    return model


def predict_signals(model, feat_values, actions, signal_map_shifted, tf_key="4h", asset_id=1.0):
    """Run model on all signal bars, return per-bar predictions.

    Returns dict of bar_idx -> (p_win, tp1_dist, tp2_dist).
    """
    ASSET_ID_MAP = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
    TF_ID_MAP = {"15m": 0, "1h": 1, "4h": 2}
    window = WINDOW_BY_TF.get(tf_key, 30)
    scaler = joblib.load(SCALER_FILE)
    scaled = scaler.transform(feat_values)

    a_id = torch.LongTensor([ASSET_ID_MAP.get(asset_id, 0)])
    t_id = torch.LongTensor([TF_ID_MAP.get(tf_key, 0)])

    preds = {}
    n = len(actions)
    for bar_idx in range(window - 1, n):
        if actions[bar_idx] == 0:
            continue
        if bar_idx not in signal_map_shifted:
            continue

        x = scaled[bar_idx - window + 1: bar_idx + 1]
        x_t = torch.FloatTensor(x).unsqueeze(0)
        d_id = torch.LongTensor([actions[bar_idx]])
        with torch.no_grad():
            out = model(x_t, asset_ids=a_id, tf_ids=t_id, direction_ids=d_id).squeeze(0)  # (4,)
        p_win = torch.sigmoid(out[0]).item()
        tp1_dist = out[1].item()
        tp2_dist = out[2].item()
        preds[bar_idx] = (p_win, tp1_dist, tp2_dist)

    return preds


def eval_with_model(actions, quality, mfe, sl_labels, preds, cutoff_idx, tf_label, tf_key):
    """Evaluate at multiple confidence thresholds using MFE-based outcomes."""
    drop_n = 30
    post_cutoff = max(0, cutoff_idx - drop_n)

    # Collect signal data
    signals = []
    for bar_idx, (p_win, tp1_dist, tp2_dist) in preds.items():
        if bar_idx < post_cutoff:
            continue
        action = actions[bar_idx]
        if action == 0:
            continue
        raw = bar_idx + drop_n
        signals.append({
            "bar_idx": bar_idx,
            "p_win": p_win, "tp1_dist": tp1_dist, "tp2_dist": tp2_dist,
            "quality": quality[raw],
            "mfe": mfe[raw],
            "sl": sl_labels[raw],
            "action": action,
        })

    n_total = len(signals)
    if n_total == 0:
        print(f"  No signals after cutoff")
        return None

    n_prof = sum(1 for s in signals if s["quality"] == 1)
    avg_mfe = np.mean([s["mfe"] for s in signals]) * 100
    avg_sl = np.mean([s["sl"] for s in signals]) * 100
    print(f"\n  {tf_label}: {n_total} signals after {CUTOFF}")
    print(f"  Base: {n_prof}/{n_total} profitable ({n_prof/n_total*100:.0f}%), avg MFE={avg_mfe:.2f}%, avg SL={avg_sl:.2f}%")

    # Threshold analysis
    print(f"\n  {'Thresh':>8} {'Trades':>8} {'GateWR':>8} {'TP1WR%':>8} {'TP2WR%':>8} {'TP1%':>8} {'TP2%':>8} {'SL%':>8} {'EV1%':>8} {'EV2%':>8} {'TotalR':>8} {'$10K->':>12}")
    print(f"  {'-'*112}")

    results = []
    for thresh in THRESHOLDS:
        filtered = [s for s in signals if s["p_win"] >= thresh]
        n_filt = len(filtered)
        if n_filt == 0:
            continue

        # Gate WR: MFE > SL
        gate_wr = sum(1 for s in filtered if s["quality"] == 1) / n_filt * 100
        # TP1 WR: MFE >= predicted TP1
        tp1_wr = sum(1 for s in filtered if s["mfe"] >= s["tp1_dist"]) / n_filt * 100
        # TP2 WR: MFE >= predicted TP2
        tp2_wr = sum(1 for s in filtered if s["mfe"] >= s["tp2_dist"]) / n_filt * 100

        avg_tp1 = np.mean([s["tp1_dist"] for s in filtered]) * 100
        avg_tp2 = np.mean([s["tp2_dist"] for s in filtered]) * 100
        avg_sl_f = np.mean([s["sl"] for s in filtered]) * 100
        ev1 = (tp1_wr / 100) * avg_tp1 - (1 - tp1_wr / 100) * avg_sl_f
        ev2 = (tp2_wr / 100) * avg_tp2 - (1 - tp2_wr / 100) * avg_sl_f

        # Account sim using TP1
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

        print(f"  {thresh:>8.1f} {n_filt:>8} {gate_wr:>7.1f}% {tp1_wr:>7.1f}% {tp2_wr:>7.1f}% {avg_tp1:>7.2f}% {avg_tp2:>7.2f}% {avg_sl_f:>7.2f}% {ev1:>+7.3f}% {ev2:>+7.3f}% {total_r:>+7.1f}R ${balance:>10,.0f}")

        results.append({
            "threshold": thresh,
            "n_signals": n_filt,
            "gate_wr": round(gate_wr, 1),
            "win_rate_tp1": round(tp1_wr, 1),
            "win_rate_tp2": round(tp2_wr, 1),
            "avg_tp1": round(avg_tp1, 2),
            "avg_tp2": round(avg_tp2, 2),
            "avg_sl": round(avg_sl_f, 2),
            "ev1": round(ev1, 3),
            "ev2": round(ev2, 3),
            "total_r": round(total_r, 1),
            "final_balance": round(balance, 0),
        })

    return {
        "label": tf_label,
        "tf_key": tf_key,
        "n_signals_raw": n_total,
        "base_profitable_pct": round(n_prof / n_total * 100, 1),
        "avg_mfe": round(avg_mfe, 2),
        "avg_sl": round(avg_sl, 2),
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

            actions, quality, mfe, sl_labels, ttp_labels, swept_levels, signal_map, _mae = generate_labels(
                df["High"].values, df["Low"].values,
                df["Close"].values, df["Open"].values,
                volumes=df["Volume"].values if "Volume" in df.columns else None,
                tf_key=tf_key,
            )

            feat, actions_trimmed, signal_map_shifted = build_liq_range_sfp_features(
                df, actions, signal_map, tf_hours, asset_id=cfg["asset_id"],
            )

            feat_arr = feat.values if hasattr(feat, 'values') else feat
            preds = predict_signals(model, feat_arr.astype(np.float32), actions_trimmed, signal_map_shifted, tf_key=tf_key, asset_id=cfg["asset_id"])

            result = eval_with_model(
                actions_trimmed, quality, mfe, sl_labels,
                preds, cutoff_idx, label, tf_key,
            )
            if result:
                all_results.append(result)

    # Save
    if all_results:
        os.makedirs("benchmark", exist_ok=True)
        out_file = "benchmark/liq_range_sfp_results.txt"
        with open(out_file, "w") as f:
            f.write(f"LIQ+RANGE+SFP BENCHMARK RESULTS (MFE Regression)\n")
            f.write(f"Model: {MODEL_FILE}  |  Cutoff: {CUTOFF}  |  Horizon: {HORIZON_BY_TF}\n")
            f.write(f"{'='*120}\n\n")

            for r in all_results:
                f.write(f"{r['label']}  ({r['n_signals_raw']} signals, {r['base_profitable_pct']:.0f}% profitable, avg MFE={r['avg_mfe']:.2f}%, avg SL={r['avg_sl']:.2f}%)\n")
                f.write(f"  {'Thresh':>8} {'Trades':>8} {'GateWR':>8} {'TP1WR%':>8} {'TP2WR%':>8} {'TP1%':>8} {'TP2%':>8} {'SL%':>8} {'EV1%':>8} {'EV2%':>8} {'TotalR':>8} {'$10K->':>12}\n")
                f.write(f"  {'-'*112}\n")
                for t in r["thresholds"]:
                    f.write(f"  {t['threshold']:>8.1f} {t['n_signals']:>8} {t['gate_wr']:>7.1f}% {t['win_rate_tp1']:>7.1f}% {t['win_rate_tp2']:>7.1f}% {t['avg_tp1']:>7.2f}% {t['avg_tp2']:>7.2f}% {t['avg_sl']:>7.2f}% {t['ev1']:>+7.3f}% {t['ev2']:>+7.3f}% {t['total_r']:>+7.1f}R ${t['final_balance']:>10,.0f}\n")
                f.write(f"\n")

            # Summary
            f.write(f"{'='*120}\n")
            f.write(f"SUMMARY (best TP1 EV threshold per asset/tf)\n")
            f.write(f"{'='*120}\n")
            f.write(f"{'Label':<16} {'Thresh':>8} {'Trades':>8} {'GateWR':>8} {'TP1WR%':>8} {'TP2WR%':>8} {'EV1%':>8} {'TotalR':>8} {'$10K->':>12}\n")
            f.write(f"{'-'*100}\n")
            for r in all_results:
                best = max(r["thresholds"], key=lambda t: t["ev1"]) if r["thresholds"] else None
                if best:
                    f.write(f"{r['label']:<16} {best['threshold']:>8.1f} {best['n_signals']:>8} {best['gate_wr']:>7.1f}% {best['win_rate_tp1']:>7.1f}% {best['win_rate_tp2']:>7.1f}% {best['ev1']:>+7.3f}% {best['total_r']:>+7.1f}R ${best['final_balance']:>10,.0f}\n")

        print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
