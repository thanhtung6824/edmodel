"""Range-SFP classifier training: predict P(win) for boundary SFPs.

Usage:
    python -m src.train_range_sfp              # all TFs, all assets with CSVs
    python -m src.train_range_sfp 1h           # 1h only
    python -m src.train_range_sfp 4h 1h 15min  # all timeframes combined
    python -m src.train_range_sfp 4h 1h 15min --assets btc gold
    python -m src.train_range_sfp 4h 1h 15min --resume
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.range_sfp_labels import (
    generate_labels,
    detect_market_structure,
)
from src.labels.three_tap_labels import compute_atr
from src.models.dataset import SFPDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0},
    "gold": {"prefix": "gold", "asset_id": 2.0},
    "sol": {"prefix": "sol", "asset_id": 4.0},
    "eth": {"prefix": "eth", "asset_id": 5.0},
}

# Parse CLI
args = sys.argv[1:]
RESUME = "--resume" in args
if RESUME:
    args.remove("--resume")
if "--assets" in args:
    idx = args.index("--assets")
    TIMEFRAMES = args[:idx] if idx > 0 else ["4h", "1h", "15min"]
    SELECTED_ASSETS = args[idx + 1:]
else:
    TIMEFRAMES = args if args else ["4h", "1h", "15min"]
    SELECTED_ASSETS = []

if not SELECTED_ASSETS:
    SELECTED_ASSETS = [
        name for name, cfg in ASSETS.items()
        if any(os.path.exists(f"data/{cfg['prefix']}_{tf}.csv") for tf in TIMEFRAMES)
    ]

MODEL_FILE = "best_model_range_sfp.pth"

TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Model: {MODEL_FILE}")

N_FEATURES = 20


def build_features(df, actions, signal_map, tf_hours, asset_id=1.0):
    """Build 20 Range-SFP features.

    12 range context + 4 price context + 4 context/direction.
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    feat = pd.DataFrame()

    # --- Range context features (12) ---
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_high = np.zeros(n, dtype=np.float32)
    range_touches_low = np.zeros(n, dtype=np.float32)
    range_time_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth = np.zeros(n, dtype=np.float32)
    reclaim_strength = np.zeros(n, dtype=np.float32)
    position_in_range = np.zeros(n, dtype=np.float32)
    vol_compression = np.zeros(n, dtype=np.float32)
    ms_alignment = np.zeros(n, dtype=np.float32)
    zone_sl_distance = np.zeros(n, dtype=np.float32)  # structural SL distance
    zone_tp_distance = np.zeros(n, dtype=np.float32)  # distance to opposite zone

    for i, sig in signal_map.items():
        r = sig.range_ref
        range_h = r.high - r.low
        mid_price = (r.high + r.low) / 2.0
        entry = sig.swept_level

        range_height_pct[i] = range_h / (mid_price + 1e-8)
        range_touches_high[i] = r.touches_high / 5.0
        range_touches_low[i] = r.touches_low / 5.0
        range_time_concentration[i] = getattr(r, '_time_concentration', 0.8)
        range_age[i] = (i - r.confirmed) / 200.0
        sweep_depth[i] = sig.sweep_depth
        reclaim_strength[i] = sig.reclaim_strength
        if range_h > 0:
            position_in_range[i] = (closes[i] - r.low) / range_h
        vol_compression[i] = getattr(r, '_vol_compression', 1.0)
        ms_alignment[i] = sig.ms_alignment

        # Structural SL/TP distances from zone boundaries
        if entry > 0:
            if sig.direction == 1:  # long
                zone_sl_distance[i] = (entry - r.support.bottom) / entry
                zone_tp_distance[i] = (r.resistance.top - entry) / entry
            elif sig.direction == 2:  # short
                zone_sl_distance[i] = (r.resistance.top - entry) / entry
                zone_tp_distance[i] = (entry - r.support.bottom) / entry

    feat["range_height_pct"] = range_height_pct
    feat["range_touches_high"] = range_touches_high
    feat["range_touches_low"] = range_touches_low
    feat["range_time_concentration"] = range_time_concentration
    feat["range_age"] = range_age
    feat["sweep_depth"] = sweep_depth
    feat["reclaim_strength"] = reclaim_strength
    feat["position_in_range"] = position_in_range
    feat["vol_compression"] = vol_compression
    feat["ms_alignment"] = ms_alignment
    feat["zone_sl_distance"] = zone_sl_distance
    feat["zone_tp_distance"] = zone_tp_distance

    # --- Price context features (4) ---
    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["rsi"] = df["rsi"] / 100.0
    feat["vol_rel_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)
    feat["trend_strength"] = (df["Close"] - df["ema_21"]) / df["Close"]

    # --- Context features (4) ---
    feat["ms_strength"] = ms_strength_arr

    direction_feat = np.zeros(n, dtype=np.float32)
    direction_feat[actions == 1] = 1.0
    direction_feat[actions == 2] = -1.0
    feat["direction"] = direction_feat

    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth"] = feat["sweep_depth"].clip(0, 2.0)
    feat["reclaim_strength"] = feat["reclaim_strength"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_distance"] = feat["zone_sl_distance"].clip(0, 0.10)
    feat["zone_tp_distance"] = feat["zone_tp_distance"].clip(0, 0.15)

    return feat, actions, signal_map_shifted


def _process_one_tf(args):
    """Process a single asset/TF combo — runs in a worker process."""
    asset_name, prefix, asset_id, tf, tf_key, tf_hours = args
    data_file = f"data/{prefix}_{tf}.csv"
    if not os.path.exists(data_file):
        return None

    df = pd.read_csv(data_file).reset_index(drop=True)

    actions, quality, tp_labels, sl_labels, swept_levels, signal_map, _all_ranges, _active_per_bar = generate_labels(
        df["High"].values, df["Low"].values,
        df["Close"].values, df["Open"].values,
        tf_key=tf_key,
    )

    feat, actions, signal_map_shifted = build_features(
        df, actions, signal_map, tf_hours, asset_id=asset_id,
    )

    drop_n = 30
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]

    feat_values = feat.values.astype(np.float32)
    signal_mask = actions != 0
    total_signals = int(np.sum(signal_mask))
    n_profitable = int(np.sum(quality[signal_mask] == 1))

    if total_signals == 0:
        return None

    split_idx = int(len(feat_values) * 0.8)
    return {
        "label": f"{asset_name}/{tf_key}",
        "n_bars": len(feat_values),
        "n_signals": total_signals,
        "n_profitable": n_profitable,
        "train_feat": feat_values[:split_idx],
        "train_actions": actions[:split_idx],
        "train_quality": quality[:split_idx],
        "train_tp": tp_labels[:split_idx],
        "train_sl": sl_labels[:split_idx],
        "test_feat": feat_values[split_idx:],
        "test_actions": actions[split_idx:],
        "test_quality": quality[split_idx:],
        "test_tp": tp_labels[split_idx:],
        "test_sl": sl_labels[split_idx:],
    }


def load_data_set():
    from multiprocessing import Pool

    train_feats, train_actions, train_quality, train_tp, train_sl = [], [], [], [], []
    test_feats, test_actions, test_quality, test_tp, test_sl = [], [], [], [], []

    work_items = []
    for asset_name in SELECTED_ASSETS:
        asset_cfg = ASSETS[asset_name]
        prefix = asset_cfg["prefix"]
        asset_id = asset_cfg["asset_id"]
        for tf in TIMEFRAMES:
            tf_key = TF_KEYS[tf]
            tf_hours = TF_HOURS[tf]
            work_items.append((asset_name, prefix, asset_id, tf, tf_key, tf_hours))

    print(f"\nProcessing {len(work_items)} asset/TF combos in parallel...")
    with Pool() as pool:
        results = pool.map(_process_one_tf, work_items)

    for result in results:
        if result is None:
            continue

        label = result["label"]
        print(f"  {label}: {result['n_bars']} bars, {result['n_signals']} signals, "
              f"{result['n_profitable']} profitable ({result['n_profitable']/result['n_signals']*100:.0f}%)")

        train_feats.append(result["train_feat"])
        train_actions.append(result["train_actions"])
        train_quality.append(result["train_quality"])
        train_tp.append(result["train_tp"])
        train_sl.append(result["train_sl"])
        test_feats.append(result["test_feat"])
        test_actions.append(result["test_actions"])
        test_quality.append(result["test_quality"])
        test_tp.append(result["test_tp"])
        test_sl.append(result["test_sl"])

    if not train_feats:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    train_feat = np.concatenate(train_feats)
    test_feat = np.concatenate(test_feats)

    n_features = train_feat.shape[1]
    total_train_signals = int(np.sum(np.concatenate(train_actions) != 0))
    total_test_signals = int(np.sum(np.concatenate(test_actions) != 0))
    print(f"\nCombined — Train: {len(train_feat)} bars ({total_train_signals} signals) | Test: {len(test_feat)} bars ({total_test_signals} signals)")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)

    window = 30
    train_set = SFPDataset(
        train_scaled,
        np.concatenate(train_actions),
        np.concatenate(train_quality),
        np.concatenate(train_tp),
        np.concatenate(train_sl),
        window=window,
    )
    test_set = SFPDataset(
        test_scaled,
        np.concatenate(test_actions),
        np.concatenate(test_quality),
        np.concatenate(test_tp),
        np.concatenate(test_sl),
        window=window,
    )

    print(f"Train signals: {len(train_set)}, Test signals: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return train_loader, test_loader, n_features


def evaluate(model, test_loader, criterion, tp_labels_raw, sl_labels_raw):
    """Evaluate classifier at multiple confidence thresholds."""
    model.eval()
    all_logits = []
    all_quality = []
    total_loss = 0

    with torch.no_grad():
        for x, direction, q, tp, sl in test_loader:
            x = x.to(device)
            q_t = q.to(device).float()

            logit = model(x)
            loss = criterion(logit, q_t)
            total_loss += loss.item()

            all_logits.append(logit.cpu())
            all_quality.append(q.cpu())

    logits = torch.cat(all_logits)
    quality = torch.cat(all_quality)
    probs = torch.sigmoid(logits)

    base_wr = (quality == 1).float().mean().item() * 100
    n_total = len(quality)

    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        take = probs > thresh
        n_take = take.sum().item()
        if n_take > 0:
            n_win = (quality[take] == 1).sum().item()
            precision = n_win / n_take * 100
            recall = n_win / max((quality == 1).sum().item(), 1) * 100
            take_idx = take.numpy()
            avg_tp = float(np.mean(tp_labels_raw[take_idx])) * 100
            avg_sl = float(np.mean(sl_labels_raw[take_idx])) * 100
            ev = (precision / 100) * avg_tp - (1 - precision / 100) * avg_sl
        else:
            precision = recall = avg_tp = avg_sl = ev = 0
        results[thresh] = (n_take, precision, recall, avg_tp, avg_sl, ev)

    prof_probs = probs[quality == 1]
    lose_probs = probs[quality == 0]
    if len(prof_probs) > 0 and len(lose_probs) > 0:
        print(
            f"    P(win) spread — "
            f"winners: {prof_probs.mean().item():.3f} vs "
            f"losers: {lose_probs.mean().item():.3f} | "
            f"base WR: {base_wr:.0f}%"
        )

    return total_loss / len(test_loader), results


def train():
    train_loader, test_loader, n_features = load_data_set()

    from src.models.range_sfp_model import RangeSFPClassifier
    model = RangeSFPClassifier(n_features=n_features, window=30, hidden=22).to(device)

    if RESUME and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        print(f"Resumed from {MODEL_FILE}")
    elif RESUME:
        print(f"WARNING: --resume but {MODEL_FILE} not found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RangeSFPClassifier: {n_params:,} parameters")

    # Compute class weight
    all_q = []
    for x, direction, q, tp, sl in train_loader:
        all_q.append(q)
    all_q = torch.cat(all_q)
    n_pos = (all_q == 1).sum().item()
    n_neg = (all_q == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    print(f"Class balance — wins: {n_pos}, losses: {n_neg}, pos_weight: {pos_weight.item():.2f}")

    # Collect raw TP/SL for test set EV computation
    test_tp_raw, test_sl_raw = [], []
    for x, direction, q, tp, sl in test_loader:
        test_tp_raw.append(tp.numpy())
        test_sl_raw.append(sl.numpy())
    test_tp_raw = np.concatenate(test_tp_raw)
    test_sl_raw = np.concatenate(test_sl_raw)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    resume_lr = 1e-4 if RESUME else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=resume_lr, epochs=150,
        steps_per_epoch=len(train_loader),
    )

    best_score = -float("inf")
    best_loss = float("inf")
    counter = 0
    epochs = 150

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, direction, q, tp, sl in train_loader:
            x = x.to(device)
            q_t = q.to(device).float()

            logit = model(x)
            loss = criterion(logit, q_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        test_loss, results = evaluate(model, test_loader, criterion, test_tp_raw, test_sl_raw)
        avg_train_loss = train_loss / len(train_loader)

        r50 = results.get(0.5, (0, 0, 0, 0, 0, 0))
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} / {test_loss:.4f} | "
            f"P>0.5: {r50[0]} trades, {r50[1]:.0f}% WR, EV={r50[5]:+.3f}%"
        )

        if (epoch + 1) % 10 == 0:
            print("  --- Confidence threshold analysis ---")
            for thresh, (n, prec, rec, avg_tp, avg_sl, ev) in sorted(results.items()):
                print(
                    f"    P(win) > {thresh}: {n} trades | "
                    f"WR: {prec:.0f}% | Recall: {rec:.0f}% | "
                    f"TP: {avg_tp:.2f}% | SL: {avg_sl:.2f}% | EV: {ev:+.3f}%"
                )

        # Save best model by weighted EV score
        score = 0.0
        for thresh, weight in [(0.4, 1.0), (0.5, 2.0), (0.6, 3.0)]:
            n_t, p, _, atp, asl, ev = results.get(thresh, (0, 0, 0, 0, 0, 0))
            if n_t >= 10 and ev > 0:
                score += weight * ev * min(n_t, 200)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"  -> Saved best model (score: {score:.1f})")

        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= 40:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation with best model
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, results = evaluate(model, test_loader, criterion, test_tp_raw, test_sl_raw)
    print(f"\n{'=' * 60}")
    print(f"Best model — Classification results")
    print(f"{'=' * 60}")
    for thresh, (n, prec, rec, avg_tp, avg_sl, ev) in sorted(results.items()):
        print(
            f"  P(win) > {thresh}: {n} trades | "
            f"WR: {prec:.0f}% | Recall: {rec:.0f}% | "
            f"TP: {avg_tp:.2f}% | SL: {avg_sl:.2f}% | EV: {ev:+.3f}%"
        )


train()
