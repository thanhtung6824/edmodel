"""Benchmark: Three-Tap strategy — per-TF range detection.

Each timeframe detects its own ranges, then runs deviation/MSS/FVG/retest.

Usage:
    python benchmark_three_tap.py
    python benchmark_three_tap.py --assets btc gold
"""

import os
import sys
import numpy as np
import pandas as pd

from src.labels.three_tap_labels import generate_labels

CUTOFF = "2023-01-01"
TFS = ["15min", "1h", "4h"]
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}

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


def eval_signals(actions, quality, tp_labels, sl_labels, cutoff_idx, tf_label):
    """Evaluate signals and print stats."""
    signal_mask = (actions != 0)
    post_cutoff_mask = np.zeros(len(actions), dtype=bool)
    post_cutoff_mask[cutoff_idx:] = True
    eval_mask = signal_mask & post_cutoff_mask

    n_total = int(signal_mask.sum())
    n_eval = int(eval_mask.sum())
    n_long = int(((actions == 1) & post_cutoff_mask).sum())
    n_short = int(((actions == 2) & post_cutoff_mask).sum())

    print(f"\n  Signals (total / after {CUTOFF}): {n_total} / {n_eval}")
    print(f"  Long: {n_long}  |  Short: {n_short}")

    if n_eval == 0:
        print("  No signals after cutoff")
        return None

    wins = quality[eval_mask] == 1
    n_wins = int(wins.sum())
    n_losses = n_eval - n_wins
    win_rate = n_wins / n_eval * 100

    tp_vals = tp_labels[eval_mask]
    sl_vals = sl_labels[eval_mask]
    ratios = tp_vals / (sl_vals + 1e-8)

    avg_tp = float(np.mean(tp_vals)) * 100
    avg_sl = float(np.mean(sl_vals)) * 100
    avg_ratio = float(np.mean(ratios))
    med_ratio = float(np.median(ratios))
    ev = (win_rate / 100) * avg_tp - (1 - win_rate / 100) * avg_sl

    print(f"\n  Win Rate:   {win_rate:.1f}% ({n_wins}W / {n_losses}L)")
    print(f"  Avg TP:     {avg_tp:.2f}%")
    print(f"  Avg SL:     {avg_sl:.2f}%")
    print(f"  Avg R:R:    {avg_ratio:.2f}")
    print(f"  Median R:R: {med_ratio:.2f}")
    print(f"  EV/trade:   {ev:+.3f}%")

    for dir_name, dir_code in [("LONG", 1), ("SHORT", 2)]:
        dir_mask = eval_mask & (actions == dir_code)
        n_dir = int(dir_mask.sum())
        if n_dir == 0:
            continue
        dir_wins = int((quality[dir_mask] == 1).sum())
        dir_wr = dir_wins / n_dir * 100
        dir_tp = float(np.mean(tp_labels[dir_mask])) * 100
        dir_sl = float(np.mean(sl_labels[dir_mask])) * 100
        dir_ev = (dir_wr / 100) * dir_tp - (1 - dir_wr / 100) * dir_sl
        print(f"    {dir_name}: {n_dir} signals, WR={dir_wr:.1f}%, TP={dir_tp:.2f}%, SL={dir_sl:.2f}%, EV={dir_ev:+.3f}%")

    return {
        "label": tf_label,
        "n_signals": n_eval,
        "win_rate": round(win_rate, 1),
        "avg_tp": round(avg_tp, 2),
        "avg_sl": round(avg_sl, 2),
        "avg_rr": round(avg_ratio, 2),
        "ev_per_trade": round(ev, 3),
    }


def run_benchmark():
    print("=" * 70)
    print("THREE-TAP STRATEGY BENCHMARK (per-TF range detection)")
    print("=" * 70)

    all_results = []

    for asset_name in SELECTED_ASSETS:
        cfg = ASSETS.get(asset_name)
        if not cfg:
            print(f"Unknown asset: {asset_name}")
            continue

        print(f"\n{'='*70}")
        print(f"  {asset_name.upper()}")
        print(f"{'='*70}")

        for tf in TFS:
            tf_key = TF_KEYS[tf]
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
            print(f"\n  --- {label} ---")

            actions, quality, tp_labels, sl_labels, entry_levels, _zones = generate_labels(
                df["High"].values, df["Low"].values,
                df["Close"].values, df["Open"].values,
                precomputed_ranges=None,  # detect ranges on this TF's data
                tf_key=tf_key,
                require_mss=True,
                allow_multi_dev=False,
                mss_mode="soft",
            )

            result = eval_signals(
                actions, quality, tp_labels, sl_labels,
                cutoff_idx, label,
            )
            if result:
                all_results.append(result)

    # Summary table
    if all_results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Label':<12} {'Signals':>8} {'WR%':>8} {'AvgTP%':>8} {'AvgSL%':>8} {'R:R':>8} {'EV%':>8}")
        print("-" * 66)
        for r in all_results:
            print(f"{r['label']:<12} {r['n_signals']:>8} {r['win_rate']:>7.1f}% {r['avg_tp']:>7.2f}% {r['avg_sl']:>7.2f}% {r['avg_rr']:>7.2f} {r['ev_per_trade']:>+7.3f}%")


if __name__ == "__main__":
    run_benchmark()
