"""Validate SFP Transformer on resampled BTC data from 1-min."""

import json
import numpy as np
import pandas as pd
import torch
from ta import volume, volatility, trend, momentum
from sklearn.preprocessing import StandardScaler

from src.labels.sfp_labels import (
    detect_swings,
    build_swing_level_series,
    compute_swing_level_info,
    detect_sfp,
    compute_tp_sl_labels,
)
from src.models.sfp_transformer import SFPTransformer

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def resample_1m(timeframe="1h", cutoff="2024-01-01"):
    """Resample 1-min BTC data to target timeframe with indicators."""
    print("Loading 1-min data...")
    df = pd.read_csv("data/btc_data.csv")
    df["timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("timestamp")
    print(f"  1-min bars: {len(df)}, range: {df.index[0]} to {df.index[-1]}")

    print(f"Resampling to {timeframe}...")
    df_rs = (
        df.resample(timeframe)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )

    # Compute indicators
    df_rs["obv"] = volume.on_balance_volume(close=df_rs["Close"], volume=df_rs["Volume"], fillna=True)
    df_rs["bb"] = volatility.bollinger_wband(close=df_rs["Close"], window=20, window_dev=2, fillna=True)
    df_rs["ema_21"] = trend.ema_indicator(close=df_rs["Close"], window=21, fillna=True)
    df_rs["rsi"] = momentum.rsi(close=df_rs["Close"], fillna=True)
    df_rs = df_rs.dropna().reset_index()

    # Filter to recent data only (mature BTC market)
    before = len(df_rs)
    df_rs = df_rs[df_rs["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"  {timeframe} bars: {len(df_rs)} (filtered from {before}, cutoff {cutoff})")
    print(f"  Range: {df_rs['timestamp'].iloc[0]} to {df_rs['timestamp'].iloc[-1]}")
    return df_rs


def run_sfp_pipeline(df):
    """Run SFP detection and feature engineering on 1h data."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    # --- SFP detection (same as training) ---
    reclaim_windows = {5: 1, 10: 3}
    results = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl, _, _ = build_swing_level_series(highs, lows, sh, sl, n, max_age=150)
        actions, swept = detect_sfp(
            highs, lows, closes, opens, active_sh, active_sl,
            reclaim_window=reclaim_windows[n],
        )
        results[n] = (actions, swept)

    # Merge
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

    # TP/SL labels for validation
    quality, tp_labels, sl_labels = compute_tp_sl_labels(highs, lows, closes, actions, swept_levels)

    total = int(np.sum(actions != 0))
    prof = int(np.sum((actions != 0) & (quality == 1)))
    print(f"  SFP signals: {total} | Profitable: {prof} ({prof/total*100:.0f}%) | Losing: {total - prof}")

    # --- Feature engineering (same 20 features as training) ---
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
    feat["tf_hours"] = 1.0 / 4.0  # 1h
    feat["asset_id"] = 1.0  # BTC

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]
    swept_levels = swept_levels[drop_n:]
    timestamps = df["timestamp"].values[drop_n:]

    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    for n in [5, 10]:
        feat[f"sweep_below_{n}"] = feat[f"sweep_below_{n}"].clip(0, 0.05)
        feat[f"sweep_above_{n}"] = feat[f"sweep_above_{n}"].clip(0, 0.05)
    feat["obv_slope"] = feat["obv_slope"].clip(-5.0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["reclaim_distance"] = feat["reclaim_distance"].clip(0, 0.05)

    print(f"  Features ({feat.shape[1]}): {list(feat.columns)}")
    return feat.values.astype(np.float32), actions, quality, tp_labels, sl_labels, swept_levels, timestamps


def predict_all_candles(feat_values, model, window=30):
    """Run model on ALL candles and return predictions for every bar."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_values)

    all_indices = list(range(window - 1, len(feat_values)))
    print(f"  Total candles to evaluate: {len(all_indices)}")

    all_tp, all_sl = [], []
    model.eval()
    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(all_indices), batch_size):
            batch_idx = all_indices[start : start + batch_size]
            batch_x = np.stack([scaled[i - window + 1 : i + 1] for i in batch_idx])
            x_t = torch.FloatTensor(batch_x).to(device)
            tp_pred, sl_pred = model(x_t)
            all_tp.extend(tp_pred.cpu().numpy().tolist())
            all_sl.extend(sl_pred.cpu().numpy().tolist())

    return all_indices, np.array(all_tp), np.array(all_sl)


def analyze(all_indices, tp_preds, sl_preds, actions, quality, tp_labels, sl_labels, swept_levels, timestamps):
    """Full analysis: pass all candles, check which high-ratio ones are real SFP signals."""
    ratio = tp_preds / (sl_preds + 1e-6)

    total_sfp = sum(actions[i] != 0 for i in all_indices)
    sfp_prof = sum((actions[i] != 0 and quality[i] == 1) for i in all_indices)

    print(f"\n{'='*80}")
    print(f"Signal Analysis — 1h BTC data (from cutoff)")
    print(f"{'='*80}")
    print(f"Total candles evaluated: {len(all_indices)}")
    print(f"SFP signals in data: {total_sfp} ({sfp_prof} profitable, {sfp_prof/total_sfp*100:.0f}% base rate)\n")

    print(f"{'Thresh':>6} | {'Flagged':>7} | {'IsSFP':>5} | {'SFP%':>5} | {'Prof':>5} | {'Prec':>5} | {'R:R':>5} | {'AvgTP':>6} | {'AvgSL':>6} | {'EV/trade':>8}")
    print("-" * 90)

    for thresh in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        take = ratio > thresh
        n_flagged = take.sum()
        if n_flagged == 0:
            continue
        flagged_indices = [all_indices[j] for j in range(len(all_indices)) if take[j]]
        sfp_flagged = [i for i in flagged_indices if actions[i] != 0]
        n_sfp = len(sfp_flagged)
        sfp_pct = n_sfp / n_flagged * 100
        n_prof = sum(quality[i] == 1 for i in sfp_flagged)
        prec = n_prof / n_sfp * 100 if n_sfp > 0 else 0
        avg_tp = np.mean([tp_labels[i] for i in sfp_flagged]) * 100 if n_sfp > 0 else 0
        avg_sl = np.mean([sl_labels[i] for i in sfp_flagged]) * 100 if n_sfp > 0 else 0
        rr = avg_tp / avg_sl if avg_sl > 0 else 0
        ev = (prec / 100 * avg_tp) - ((100 - prec) / 100 * avg_sl) if n_sfp > 0 else 0
        print(f"{thresh:>6.1f} | {n_flagged:>7} | {n_sfp:>5} | {sfp_pct:>4.0f}% | {n_prof:>5} | {prec:>4.0f}% | {rr:>5.2f} | {avg_tp:>5.2f}% | {avg_sl:>5.2f}% | {ev:>+7.3f}%")

    # --- Detailed log at ratio > 1.4 (only SFP candles) ---
    target_thresh = 1.4
    take = ratio > target_thresh

    print(f"\n{'='*80}")
    print(f"Detailed SFP signals at ratio > {target_thresh}")
    print(f"{'='*80}")

    n_false_alarm = sum(1 for j in range(len(all_indices)) if take[j] and actions[all_indices[j]] == 0)
    n_sfp_flagged = sum(1 for j in range(len(all_indices)) if take[j] and actions[all_indices[j]] != 0)
    print(f"Model flagged {int(take.sum())} candles: {n_sfp_flagged} SFP signals + {n_false_alarm} non-SFP (ignored)\n")

    print(f"{'Timestamp':<22} | {'Dir':>5} | {'Entry':>10} | {'Ratio':>5} | {'ActTP':>7} | {'ActSL':>7} | {'Result':>6}")
    print("-" * 80)

    n_win, n_lose = 0, 0
    total_pnl = 0.0
    for j in range(len(all_indices)):
        if not take[j]:
            continue
        i = all_indices[j]
        if actions[i] == 0:
            continue
        ts = pd.Timestamp(timestamps[i])
        direction = "LONG" if actions[i] == 1 else "SHORT"
        entry = swept_levels[i]
        pred_ratio = ratio[j]
        act_tp = tp_labels[i] * 100
        act_sl = sl_labels[i] * 100
        won = quality[i] == 1
        result = "WIN" if won else "LOSE"
        if won:
            n_win += 1
            total_pnl += act_tp
        else:
            n_lose += 1
            total_pnl -= act_sl

        print(
            f"{str(ts):<22} | {direction:>5} | {entry:>10.2f} | {pred_ratio:>5.2f} | {act_tp:>6.2f}% | {act_sl:>6.2f}% | {result:>6}"
        )

    total = n_win + n_lose
    if total == 0:
        print("\nNo SFP trades at this threshold.")
        return
    print(f"\nSummary: {n_win} wins, {n_lose} losses | Win rate: {n_win/total*100:.0f}% | Cumulative P&L: {total_pnl:+.2f}%")

    # --- Yearly breakdown ---
    print(f"\n{'='*80}")
    print(f"Yearly breakdown at ratio > {target_thresh}")
    print(f"{'='*80}")
    yearly = {}
    risk_pct = 0.01
    account = 10000.0
    yearly_accounts = {}
    for j in range(len(all_indices)):
        if not take[j]:
            continue
        i = all_indices[j]
        if actions[i] == 0:
            continue
        year = pd.Timestamp(timestamps[i]).year
        if year not in yearly:
            yearly[year] = {"wins": 0, "losses": 0, "pnl": 0.0}
            yearly_accounts[year] = account
        actual_rr = min(tp_labels[i] / (sl_labels[i] + 1e-8), 3.0)
        if quality[i] == 1:
            yearly[year]["wins"] += 1
            yearly[year]["pnl"] += actual_rr
            account *= (1 + risk_pct * actual_rr)
        else:
            yearly[year]["losses"] += 1
            yearly[year]["pnl"] -= 1.0
            account *= (1 - risk_pct)

    print(f"{'Year':>6} | {'Trades':>6} | {'Wins':>5} | {'WinRate':>7} | {'P&L (R)':>8} | {'Account':>14} | {'YearRet':>8}")
    print("-" * 80)
    for year in sorted(yearly):
        y = yearly[year]
        total = y["wins"] + y["losses"]
        wr = y["wins"] / total * 100
        start_acc = yearly_accounts[year]
        sorted_years = sorted(yearly)
        idx = sorted_years.index(year)
        end_acc = yearly_accounts[sorted_years[idx + 1]] if idx + 1 < len(sorted_years) else account
        year_ret = (end_acc / start_acc - 1) * 100
        print(f"{year:>6} | {total:>6} | {y['wins']:>5} | {wr:>6.0f}% | {y['pnl']:>+7.1f}R | ${end_acc:>12,.0f} | {year_ret:>+7.1f}%")

    print(f"\n  Starting account: $10,000")
    print(f"  Final account:    ${account:,.0f}")
    print(f"  Total return:     {(account / 10000 - 1) * 100:+.1f}%")
    print(f"  (1R = 1% risk per trade, compounding)")


def save_signals(all_indices, tp_preds, sl_preds, actions, quality, tp_labels, sl_labels, swept_levels, timestamps, threshold=1.4):
    """Save filtered signals to JSON for TradingView Pine Script generation."""
    ratio = tp_preds / (sl_preds + 1e-6)
    signals = []

    for j in range(len(all_indices)):
        if ratio[j] <= threshold:
            continue
        i = all_indices[j]
        if actions[i] == 0:
            continue

        ts = pd.Timestamp(timestamps[i])
        unix_ms = int(ts.value // 10**6)
        direction = int(actions[i])
        entry = float(swept_levels[i])

        if direction == 1:
            tp_price = entry * (1 + tp_labels[i])
            sl_price = entry * (1 - sl_labels[i])
        else:
            tp_price = entry * (1 - tp_labels[i])
            sl_price = entry * (1 + sl_labels[i])

        signals.append({
            "time_ms": unix_ms,
            "timestamp": str(ts),
            "dir": direction,
            "entry": round(entry, 2),
            "tp_price": round(float(tp_price), 2),
            "sl_price": round(float(sl_price), 2),
            "ratio": round(float(ratio[j]), 4),
            "result": 1 if quality[i] == 1 else 0,
        })

    output_file = "signals_1h.json"
    with open(output_file, "w") as f:
        json.dump({"timeframe": "1h", "threshold": threshold, "signals": signals}, f, indent=2)

    n_wins = sum(s["result"] == 1 for s in signals)
    print(f"\nSaved {len(signals)} signals to {output_file} (ratio > {threshold})")
    print(f"  Wins: {n_wins} | Losses: {len(signals) - n_wins}")


def main():
    # 1. Resample
    df_1h = resample_1m()

    # 2. Run SFP pipeline
    print("\nRunning SFP pipeline on 1h data...")
    feat_values, actions, quality, tp_labels, sl_labels, swept_levels, timestamps = run_sfp_pipeline(df_1h)

    # 3. Load trained model
    print("\nLoading trained model...")
    model = SFPTransformer().to(device)
    model.load_state_dict(torch.load("best_model_transformer.pth", weights_only=True))
    print(f"  Model loaded from best_model_transformer.pth")

    # 4. Predict all candles
    print("\nRunning predictions on ALL candles...")
    all_indices, tp_preds, sl_preds = predict_all_candles(feat_values, model)

    # 5. Analyze
    analyze(all_indices, tp_preds, sl_preds, actions, quality, tp_labels, sl_labels, swept_levels, timestamps)
    save_signals(all_indices, tp_preds, sl_preds, actions, quality, tp_labels, sl_labels, swept_levels, timestamps)


main()
