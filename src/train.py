from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.sfp_labels import detect_swings, build_swing_level_series, generate_labels
from src.models.dataset import SFPDataset
from src.models.losses import SFPLoss
from src.models.sfp_model import SFPModel

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def load_data_set():
    df = pd.read_csv("data/btc_4h.csv")

    # Use all available data
    df = df.reset_index(drop=True)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    # --- Generate SFP labels (all detected SFPs + quality) ---
    actions, quality, tp_labels, sl_labels = generate_labels(highs, lows, closes, opens)

    # --- Swing levels for feature engineering ---
    swing_levels = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl = build_swing_level_series(highs, lows, sh, sl, n, max_age=150)
        nearest_sh = np.array([levels[0] if levels else np.nan for levels in active_sh])
        nearest_sl = np.array([levels[0] if levels else np.nan for levels in active_sl])
        swing_levels[n] = (nearest_sh, nearest_sl)

    # --- Feature engineering (SFP-focused, 14 features) ---
    prev_close = df["Close"].shift(1)
    feat = pd.DataFrame()

    # Price action (returns-based)
    feat["Open"] = df["Open"] / prev_close - 1
    feat["High"] = df["High"] / prev_close - 1
    feat["Low"] = df["Low"] / prev_close - 1
    feat["Close"] = df["Close"] / prev_close - 1

    # Context
    feat["rsi"] = df["rsi"] / 100.0
    feat["volatility"] = df["Close"].pct_change().rolling(30).std()

    # Volume spike detection
    vol_avg_20 = df["Volume"].rolling(20).mean()
    feat["vol_rel_20"] = df["Volume"] / (vol_avg_20 + 1e-8)

    # Candle structure (wick rejection = SFP signature)
    candle_range = df["High"] - df["Low"]
    candle_range_safe = candle_range.replace(0, 1e-8)
    feat["body_ratio"] = (df["Close"] - df["Open"]) / candle_range_safe
    feat["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range_safe
    feat["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range_safe

    # Sweep depth past swing levels
    for n in [5, 10]:
        recent_sh, recent_sl = swing_levels[n]
        sweep_below = np.maximum(0, recent_sl - lows) / (closes + 1e-8)
        sweep_above = np.maximum(0, highs - recent_sh) / (closes + 1e-8)
        feat[f"sweep_below_{n}"] = sweep_below
        feat[f"sweep_above_{n}"] = sweep_above

    # --- Drop first 30 rows (NaN from rolling windows) and align ---
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]

    # Clean up
    feat = feat.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)

    # Clip outliers
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    for n in [5, 10]:
        feat[f"sweep_below_{n}"] = feat[f"sweep_below_{n}"].clip(0, 0.05)
        feat[f"sweep_above_{n}"] = feat[f"sweep_above_{n}"].clip(0, 0.05)

    feat_values = feat.values.astype(np.float32)
    n_features = feat_values.shape[1]
    print(f"Features ({n_features}): {list(feat.columns)}")

    # --- Print label stats ---
    sfp_mask = actions != 0
    total_sfp = int(np.sum(sfp_mask))
    n_long = int(np.sum(actions == 1))
    n_short = int(np.sum(actions == 2))
    n_profitable = int(np.sum(quality[sfp_mask] == 1))
    n_losing = total_sfp - n_profitable
    print(f"\nSFP Label Stats:")
    print(f"  Total bars: {len(actions)}")
    print(f"  SFP signals: {total_sfp} ({total_sfp / len(actions) * 100:.1f}%)")
    print(f"  Long: {n_long}, Short: {n_short}")
    print(f"  Profitable: {n_profitable} ({n_profitable / max(total_sfp, 1) * 100:.0f}%), "
          f"Losing: {n_losing} ({n_losing / max(total_sfp, 1) * 100:.0f}%)")
    if n_profitable > 0:
        prof_mask = sfp_mask & (quality == 1)
        print(f"  Profitable — Avg TP: {tp_labels[prof_mask].mean() * 100:.2f}%, "
              f"Avg SL: {sl_labels[prof_mask].mean() * 100:.2f}%")
    if n_losing > 0:
        lose_mask = sfp_mask & (quality == 0)
        print(f"  Losing     — Avg TP: {tp_labels[lose_mask].mean() * 100:.2f}%, "
              f"Avg SL: {sl_labels[lose_mask].mean() * 100:.2f}%")

    # --- Train/test split (time-based) ---
    split_idx = int(len(feat_values) * 0.8)
    train_feat = feat_values[:split_idx]
    test_feat = feat_values[split_idx:]
    train_actions = actions[:split_idx]
    test_actions = actions[split_idx:]
    train_quality = quality[:split_idx]
    test_quality = quality[split_idx:]
    train_tp = tp_labels[:split_idx]
    test_tp = tp_labels[split_idx:]
    train_sl = sl_labels[:split_idx]
    test_sl = sl_labels[split_idx:]

    # --- Scale features ---
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)

    # --- Create datasets (SFP-only) ---
    window = 30
    train_set = SFPDataset(train_scaled, train_actions, train_quality, train_tp, train_sl, window=window)
    test_set = SFPDataset(test_scaled, test_actions, test_quality, test_tp, test_sl, window=window)

    print(f"\nTrain SFPs: {len(train_set)}, Test SFPs: {len(test_set)}")
    train_q = train_quality[train_actions != 0]
    print(f"Train quality: {int(np.sum(train_q == 1))} profitable, {int(np.sum(train_q == 0))} losing")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # pos_weight to balance profitable vs losing
    n_pos = int(np.sum(train_q == 1))
    n_neg = int(np.sum(train_q == 0))
    pos_weight = n_neg / max(n_pos, 1)
    print(f"BCE pos_weight: {pos_weight:.2f}")

    return train_loader, test_loader, n_features, pos_weight


def compute_metrics(quality_preds, quality_targets, tp_preds, sl_preds, tp_targets, sl_targets):
    """Compute per-epoch metrics for quality classification + TP/SL regression."""
    total = len(quality_targets)

    # Quality accuracy, precision, recall
    correct = (quality_preds == quality_targets).sum().item()
    accuracy = correct / total * 100 if total > 0 else 0

    pred_pos = (quality_preds == 1)
    true_pos = (quality_targets == 1)
    tp_cls = (pred_pos & true_pos).sum().item()
    precision = tp_cls / pred_pos.sum().item() * 100 if pred_pos.sum() > 0 else 0
    recall = tp_cls / true_pos.sum().item() * 100 if true_pos.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # TP/SL MAE
    tp_mae = (tp_preds - tp_targets).abs().mean().item() * 100
    sl_mae = (sl_preds - sl_targets).abs().mean().item() * 100

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_mae": tp_mae,
        "sl_mae": sl_mae,
        "pred_rate": pred_pos.sum().item() / total * 100,
    }


def train():
    train_loader, test_loader, n_features, pos_weight = load_data_set()
    model = SFPModel(n_features=n_features).to(device)

    criterion = SFPLoss(pos_weight=pos_weight, lambda_tp=1.0, lambda_sl=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_f1 = 0
    counter = 0
    epochs = 100

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0

        for x, direction, q, tp, sl in train_loader:
            x = x.to(device)
            q = q.to(device)
            tp = tp.to(device)
            sl = sl.to(device)

            quality_logit, tp_pred, sl_pred = model(x)
            loss, _, _, _ = criterion(quality_logit, tp_pred, sl_pred, q, tp, sl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # --- Evaluate on test set ---
        model.eval()
        all_quality_preds = []
        all_quality_targets = []
        all_tp_preds = []
        all_sl_preds = []
        all_tp_targets = []
        all_sl_targets = []

        with torch.no_grad():
            for x, direction, q, tp, sl in test_loader:
                x = x.to(device)
                q = q.to(device)
                tp = tp.to(device)
                sl = sl.to(device)

                quality_logit, tp_pred, sl_pred = model(x)

                quality_pred = (torch.sigmoid(quality_logit) > 0.5).long()
                all_quality_preds.append(quality_pred.cpu())
                all_quality_targets.append(q.cpu())
                all_tp_preds.append(tp_pred.cpu())
                all_sl_preds.append(sl_pred.cpu())
                all_tp_targets.append(tp.cpu())
                all_sl_targets.append(sl.cpu())

        quality_preds = torch.cat(all_quality_preds)
        quality_targets = torch.cat(all_quality_targets)
        tp_preds = torch.cat(all_tp_preds)
        sl_preds = torch.cat(all_sl_preds)
        tp_targets = torch.cat(all_tp_targets)
        sl_targets = torch.cat(all_sl_targets)

        metrics = compute_metrics(quality_preds, quality_targets, tp_preds, sl_preds, tp_targets, sl_targets)
        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.1f}% | "
            f"P/R/F1: {metrics['precision']:.0f}%/{metrics['recall']:.0f}%/{metrics['f1']:.1f}% | "
            f"Pred+: {metrics['pred_rate']:.1f}% | "
            f"TP MAE: {metrics['tp_mae']:.2f}% | "
            f"SL MAE: {metrics['sl_mae']:.2f}%"
        )

        # --- Early stopping on quality F1 ---
        f1 = metrics["f1"]
        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (F1: {best_f1:.1f}%)")
        else:
            counter += 1
            if counter >= 15:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nBest Quality F1: {best_f1:.1f}%")


train()
