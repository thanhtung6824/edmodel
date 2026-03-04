"""Three-Tap Transformer training: predict TP/SL, use ratio as quality filter.

Ranges detected on 4h, signals on all TFs. Uses soft MSS mode.

Usage:
    python -m src.train_three_tap              # all TFs, all assets with CSVs
    python -m src.train_three_tap 1h           # 1h only
    python -m src.train_three_tap 4h 1h 15min  # all timeframes combined
    python -m src.train_three_tap 4h 1h 15min --assets btc gold
    python -m src.train_three_tap 4h 1h 15min --resume
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.three_tap_labels import generate_labels
from src.models.dataset import SFPDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0},
    "gold": {"prefix": "gold", "asset_id": 2.0},
    "sol": {"prefix": "sol", "asset_id": 4.0},
    "eth": {"prefix": "eth", "asset_id": 5.0},
}

# Parse CLI: positional args are timeframes, --assets and --resume flags
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

# Default: all assets that have at least one CSV file in data/
if not SELECTED_ASSETS:
    SELECTED_ASSETS = [
        name for name, cfg in ASSETS.items()
        if any(os.path.exists(f"data/{cfg['prefix']}_{tf}.csv") for tf in TIMEFRAMES)
    ]

MODEL_FILE = "best_model_three_tap.pth"

# Map timeframe string to hours for the context feature
TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Model: {MODEL_FILE}")


N_FEATURES = 18  # for model construction


def build_features(df, actions, signal_zones, tf_hours, asset_id=1.0):
    """Build 18 three-tap specific features.

    10 setup quality features (per-signal, from zone/range metadata)
    + 5 price context features + direction + 2 context identifiers.
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values

    # Precompute ATR and volume average
    from src.labels.three_tap_labels import compute_atr
    atr = compute_atr(highs, lows, closes, period=14)
    vol_avg_20 = df["Volume"].rolling(20).mean().values

    prev_close = df["Close"].shift(1).values

    feat = pd.DataFrame()

    # --- Setup quality features (only non-zero at signal bars) ---

    # 1. Range height / ATR — how significant is this range
    range_height_atr = np.zeros(n, dtype=np.float32)
    # 2. Range total touches — how well-tested
    range_touches = np.zeros(n, dtype=np.float32)
    # 3. Range age — bars since confirmed to signal bar
    range_age = np.zeros(n, dtype=np.float32)
    # 4. Deviation depth — sweep depth as fraction of range height
    deviation_depth = np.zeros(n, dtype=np.float32)
    # 5. Zone width / close — how tight is the demand zone
    zone_width_pct = np.zeros(n, dtype=np.float32)
    # 6. Structural R:R — TP distance / SL distance
    structural_rr = np.zeros(n, dtype=np.float32)
    # 7. Has FVG — was there a real imbalance or fallback
    has_fvg = np.zeros(n, dtype=np.float32)
    # 8. Volume at deviation / average — stop hunt confirmation
    dev_volume_ratio = np.zeros(n, dtype=np.float32)
    # 9. Bars from deviation to retest — how fresh is the setup
    bars_since_dev = np.zeros(n, dtype=np.float32)
    # 10. MSS candle body ratio — strength of reclaim
    mss_body_ratio = np.zeros(n, dtype=np.float32)

    for i, zone in signal_zones.items():
        range_h = zone._range_high - zone._range_low
        local_atr = atr[i] if atr[i] > 0 else 1e-8

        range_height_atr[i] = range_h / local_atr
        range_touches[i] = zone._range_touches / 10.0  # normalize
        range_age[i] = (i - zone._range_confirmed) / 100.0  # normalize

        if range_h > 0:
            if zone.direction == 1:  # bullish: sweep below range low
                deviation_depth[i] = (zone._range_low - zone.deviation_wick) / range_h
            else:  # bearish: sweep above range high
                deviation_depth[i] = (zone.deviation_wick - zone._range_high) / range_h

        zone_w = zone.top - zone.bottom
        zone_width_pct[i] = zone_w / (closes[i] + 1e-8)

        # Structural R:R from TP/SL distances
        if zone.direction == 1:
            tp_dist = zone.tp_target - zone.top
            sl_dist = zone.top - zone.bottom
        else:
            tp_dist = zone.bottom - zone.tp_target
            sl_dist = zone.top - zone.bottom
        structural_rr[i] = (tp_dist / (sl_dist + 1e-8))

        has_fvg[i] = 1.0 if zone._has_fvg else 0.0

        dev_bar = zone._deviation_bar
        if 0 <= dev_bar < n and vol_avg_20[dev_bar] > 0:
            dev_volume_ratio[i] = volumes[dev_bar] / (vol_avg_20[dev_bar] + 1e-8)

        bars_since_dev[i] = (i - zone._deviation_bar) / 30.0  # normalize

        mss_bar = zone._mss_bar
        if 0 < mss_bar < n:
            candle_range = highs[mss_bar] - lows[mss_bar]
            if candle_range > 0:
                mss_body_ratio[i] = abs(closes[mss_bar] - opens[mss_bar]) / candle_range

    feat["range_height_atr"] = range_height_atr
    feat["range_touches"] = range_touches
    feat["range_age"] = range_age
    feat["deviation_depth"] = deviation_depth
    feat["zone_width_pct"] = zone_width_pct
    feat["structural_rr"] = structural_rr
    feat["has_fvg"] = has_fvg
    feat["dev_volume_ratio"] = dev_volume_ratio
    feat["bars_since_dev"] = bars_since_dev
    feat["mss_body_ratio"] = mss_body_ratio

    # --- Price context features (every bar) ---

    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["rsi"] = df["rsi"] / 100.0
    feat["trend_strength"] = (df["Close"] - df["ema_21"]) / df["Close"]
    feat["vol_rel_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)
    feat["bb_width"] = df["bb"] / 100.0

    # --- Direction + context ---

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
    # Remap signal_zones keys
    signal_zones_shifted = {k - drop_n: v for k, v in signal_zones.items() if k >= drop_n}

    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["structural_rr"] = feat["structural_rr"].clip(0, 10.0)
    feat["dev_volume_ratio"] = feat["dev_volume_ratio"].clip(0, 5.0)
    feat["range_height_atr"] = feat["range_height_atr"].clip(0, 15.0)

    return feat, actions


def load_data_set():
    train_feats, train_actions, train_quality, train_tp, train_sl = [], [], [], [], []
    test_feats, test_actions, test_quality, test_tp, test_sl = [], [], [], [], []

    for asset_name in SELECTED_ASSETS:
        asset_cfg = ASSETS[asset_name]
        prefix = asset_cfg["prefix"]
        asset_id = asset_cfg["asset_id"]

        print(f"\n{'='*60}")
        print(f"  {asset_name.upper()}")
        print(f"{'='*60}")

        # Per-TF range detection — each TF detects its own ranges
        for tf in TIMEFRAMES:
            tf_key = TF_KEYS[tf]
            tf_hours = TF_HOURS[tf]
            data_file = f"data/{prefix}_{tf}.csv"
            if not os.path.exists(data_file):
                print(f"\n  WARNING: {data_file} not found, skipping {asset_name}/{tf}")
                continue

            print(f"\nLoading {asset_name}/{tf_key} from {data_file}...")
            df = pd.read_csv(data_file).reset_index(drop=True)

            actions, quality, tp_labels, sl_labels, entry_levels, signal_zones = generate_labels(
                df["High"].values, df["Low"].values,
                df["Close"].values, df["Open"].values,
                precomputed_ranges=None,  # detect ranges on this TF
                tf_key=tf_key,
                require_mss=True,
                allow_multi_dev=False,
                mss_mode="soft",
            )

            feat, actions = build_features(df, actions, signal_zones, tf_hours, asset_id=asset_id)

            # Align labels with dropped warmup
            drop_n = 30
            quality = quality[drop_n:]
            tp_labels = tp_labels[drop_n:]
            sl_labels = sl_labels[drop_n:]

            feat_values = feat.values.astype(np.float32)
            signal_mask = actions != 0
            total_signals = int(np.sum(signal_mask))
            n_profitable = int(np.sum(quality[signal_mask] == 1))
            if total_signals > 0:
                print(f"  {asset_name}/{tf_key}: {len(feat_values)} bars, {total_signals} signals, {n_profitable} profitable ({n_profitable/total_signals*100:.0f}%)")
            else:
                print(f"  {asset_name}/{tf_key}: {len(feat_values)} bars, 0 signals — skipping")
                continue

            # No TP/SL normalization needed — classification uses quality labels directly
            # Raw TP/SL kept for EV computation
            split_idx = int(len(feat_values) * 0.8)

            # Split 80/20 (time-based)
            train_feats.append(feat_values[:split_idx])
            train_actions.append(actions[:split_idx])
            train_quality.append(quality[:split_idx])
            train_tp.append(tp_labels[:split_idx])
            train_sl.append(sl_labels[:split_idx])
            test_feats.append(feat_values[split_idx:])
            test_actions.append(actions[split_idx:])
            test_quality.append(quality[split_idx:])
            test_tp.append(tp_labels[split_idx:])
            test_sl.append(sl_labels[split_idx:])

    if not train_feats:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    # Concatenate
    train_feat = np.concatenate(train_feats)
    test_feat = np.concatenate(test_feats)

    n_features = train_feat.shape[1]
    total_train_signals = int(np.sum(np.concatenate(train_actions) != 0))
    total_test_signals = int(np.sum(np.concatenate(test_actions) != 0))
    print(f"\nCombined — Train: {len(train_feat)} bars ({total_train_signals} signals) | Test: {len(test_feat)} bars ({total_test_signals} signals)")
    print(f"Features ({n_features}): {list(feat.columns)}")

    all_train_tp = np.concatenate(train_tp)
    all_train_sl = np.concatenate(train_sl)
    all_train_actions = np.concatenate(train_actions)
    sig_m = all_train_actions != 0
    print(f"Normalized labels — TP median: {np.median(all_train_tp[sig_m]):.3f}, SL median: {np.median(all_train_sl[sig_m]):.3f}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)

    window = 30
    train_set = SFPDataset(
        train_scaled,
        all_train_actions,
        np.concatenate(train_quality),
        all_train_tp,
        all_train_sl,
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

    # Base win rate
    base_wr = (quality == 1).float().mean().item() * 100
    n_total = len(quality)

    # Evaluate at confidence thresholds
    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        take = probs > thresh
        n_take = take.sum().item()
        if n_take > 0:
            n_win = (quality[take] == 1).sum().item()
            precision = n_win / n_take * 100
            recall = n_win / max((quality == 1).sum().item(), 1) * 100
            # Compute EV using raw TP/SL
            take_idx = take.numpy()
            avg_tp = float(np.mean(tp_labels_raw[take_idx])) * 100
            avg_sl = float(np.mean(sl_labels_raw[take_idx])) * 100
            ev = (precision / 100) * avg_tp - (1 - precision / 100) * avg_sl
        else:
            precision = recall = avg_tp = avg_sl = ev = 0
        results[thresh] = (n_take, precision, recall, avg_tp, avg_sl, ev)

    # Pred spread
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

    from src.models.three_tap_model import ThreeTapClassifier
    model = ThreeTapClassifier(n_features=n_features, window=30, hidden=24).to(device)

    if RESUME and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        print(f"Resumed from {MODEL_FILE}")
    elif RESUME:
        print(f"WARNING: --resume but {MODEL_FILE} not found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ThreeTapClassifier: {n_params:,} parameters")

    # Compute class weight for imbalanced data (more losers than winners)
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

        # Save best model by EV at P>0.5 (or P>0.4 if more trades)
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
