"""SFP Transformer training: predict TP/SL, use ratio as quality filter.

Usage:
    python -m src.train_transformer              # default: 4h only, all assets with CSVs
    python -m src.train_transformer 1h           # 1h only
    python -m src.train_transformer 4h 1h 15min  # all timeframes combined
    python -m src.train_transformer 4h 1h 15min --assets btc gold silver
    python -m src.train_transformer 4h 1h 15min --assets btc gold --resume  # continue from existing weights
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.sfp_labels import detect_swings, build_swing_level_series, compute_swing_level_info, generate_labels
from src.models.dataset import SFPDataset
from src.models.sfp_transformer import SFPTransformer

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0, "symbol": "BTCUSDT"},
    "gold": {"prefix": "gold", "asset_id": 2.0, "symbol": "XAUUSDT"},
    "silver": {"prefix": "silver", "asset_id": 3.0, "symbol": "XAGUSDT"},
    "sol": {"prefix": "sol", "asset_id": 4.0, "symbol": "SOLUSDT"},
    "eth": {"prefix": "eth", "asset_id": 5.0, "symbol": "ETHUSDT"},
}

# Parse CLI: positional args are timeframes, --assets and --resume flags
args = sys.argv[1:]
RESUME = "--resume" in args
if RESUME:
    args.remove("--resume")
if "--assets" in args:
    idx = args.index("--assets")
    TIMEFRAMES = args[:idx] if idx > 0 else ["4h"]
    SELECTED_ASSETS = args[idx + 1:]
else:
    TIMEFRAMES = args if args else ["4h"]
    SELECTED_ASSETS = []

# Default: all assets that have at least one CSV file in data/
if not SELECTED_ASSETS:
    SELECTED_ASSETS = [
        name for name, cfg in ASSETS.items()
        if any(os.path.exists(f"data/{cfg['prefix']}_{tf}.csv") for tf in TIMEFRAMES)
    ]

MODEL_FILE = "best_model_transformer.pth"

# Map timeframe string to hours for the context feature
TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Model: {MODEL_FILE}")

class RankingRegLoss(nn.Module):
    """Regression loss + margin ranking loss on predicted TP/SL ratio.

    The ranking term forces profitable SFPs to have higher predicted
    TP/SL ratios than losing SFPs, which pushes predictions apart
    instead of collapsing to the population mean.
    """

    def __init__(self, margin=0.3, lambda_rank=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank

    def forward(self, tp_pred, sl_pred, tp_target, sl_target, quality=None):
        tp_loss = F.smooth_l1_loss(tp_pred, tp_target)
        sl_loss = F.smooth_l1_loss(sl_pred, sl_target)
        reg_loss = tp_loss + sl_loss

        # Without quality labels, return regression only (used in eval)
        if quality is None:
            return reg_loss, tp_loss, sl_loss

        ratio_pred = tp_pred / (sl_pred + 1e-6)

        prof_mask = quality == 1
        lose_mask = quality == 0
        n_prof = prof_mask.sum()
        n_lose = lose_mask.sum()

        if n_prof == 0 or n_lose == 0:
            return reg_loss, tp_loss, sl_loss

        prof_ratios = ratio_pred[prof_mask]  # (n_prof,)
        lose_ratios = ratio_pred[lose_mask]  # (n_lose,)

        # All pairs: each profitable should rank higher than each losing
        prof_exp = prof_ratios.unsqueeze(1).expand(n_prof, n_lose).reshape(-1)
        lose_exp = lose_ratios.unsqueeze(0).expand(n_prof, n_lose).reshape(-1)
        target = torch.ones_like(prof_exp)  # +1 = first input should rank higher

        rank_loss = F.margin_ranking_loss(
            prof_exp, lose_exp, target, margin=self.margin
        )

        total = reg_loss + self.lambda_rank * rank_loss
        return total, tp_loss, sl_loss

def build_features(df, actions, tf_hours, asset_id=1.0):
    """Build 22 features for one timeframe's data."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

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

    # Context features
    feat["tf_hours"] = tf_hours / 4.0  # timeframe in hours (normalized by 4h)
    feat["asset_id"] = asset_id

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]

    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    for n in [5, 10]:
        feat[f"sweep_below_{n}"] = feat[f"sweep_below_{n}"].clip(0, 0.05)
        feat[f"sweep_above_{n}"] = feat[f"sweep_above_{n}"].clip(0, 0.05)
    feat["obv_slope"] = feat["obv_slope"].clip(-5.0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["reclaim_distance"] = feat["reclaim_distance"].clip(0, 0.05)

    return feat, actions


def load_data_set():
    train_feats, train_actions, train_quality, train_tp, train_sl = [], [], [], [], []
    test_feats, test_actions, test_quality, test_tp, test_sl = [], [], [], [], []

    for asset_name in SELECTED_ASSETS:
        asset_cfg = ASSETS[asset_name]
        prefix = asset_cfg["prefix"]
        asset_id = asset_cfg["asset_id"]

        for tf in TIMEFRAMES:
            data_file = f"data/{prefix}_{tf}.csv"
            if not os.path.exists(data_file):
                print(f"\n  WARNING: {data_file} not found, skipping {asset_name}/{tf}")
                continue

            tf_hours = TF_HOURS[tf]
            print(f"\nLoading {asset_name}/{tf} from {data_file}...")
            df = pd.read_csv(data_file).reset_index(drop=True)

            highs = df["High"].values
            lows = df["Low"].values
            closes = df["Close"].values
            opens = df["Open"].values

            actions, quality, tp_labels, sl_labels = generate_labels(highs, lows, closes, opens)
            feat, actions = build_features(df, actions, tf_hours, asset_id=asset_id)

            # Align labels with dropped warmup
            drop_n = 30
            quality = quality[drop_n:]
            tp_labels = tp_labels[drop_n:]
            sl_labels = sl_labels[drop_n:]

            feat_values = feat.values.astype(np.float32)
            sfp_mask = actions != 0
            total_sfp = int(np.sum(sfp_mask))
            n_profitable = int(np.sum(quality[sfp_mask] == 1))
            print(f"  {asset_name}/{tf}: {len(feat_values)} bars, {total_sfp} SFPs, {n_profitable} profitable ({n_profitable/total_sfp*100:.0f}%)")

            # Normalize TP/SL per asset per timeframe
            split_idx = int(len(feat_values) * 0.8)
            train_sfp_mask = (actions[:split_idx] != 0)
            median_tp = np.median(tp_labels[:split_idx][train_sfp_mask])
            median_sl = np.median(sl_labels[:split_idx][train_sfp_mask])
            print(f"  {asset_name}/{tf} normalization — median TP: {median_tp*100:.2f}%, median SL: {median_sl*100:.2f}%")

            # Normalize: all TFs/assets now have TP/SL centered around ~1.0
            tp_labels = tp_labels / (median_tp + 1e-8)
            sl_labels = sl_labels / (median_sl + 1e-8)

            # Split each asset/timeframe 80/20 individually (time-based)
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

    # Concatenate
    train_feat = np.concatenate(train_feats)
    test_feat = np.concatenate(test_feats)

    n_features = train_feat.shape[1]
    total_train_sfp = int(np.sum(np.concatenate(train_actions) != 0))
    total_test_sfp = int(np.sum(np.concatenate(test_actions) != 0))
    print(f"\nCombined — Train: {len(train_feat)} bars ({total_train_sfp} SFPs) | Test: {len(test_feat)} bars ({total_test_sfp} SFPs)")
    print(f"Features ({n_features}): {list(feat.columns)}")

    # Verify normalized labels have similar scale
    all_train_tp = np.concatenate(train_tp)
    all_train_sl = np.concatenate(train_sl)
    all_train_actions = np.concatenate(train_actions)
    sfp_m = all_train_actions != 0
    print(f"Normalized labels — TP median: {np.median(all_train_tp[sfp_m]):.3f}, SL median: {np.median(all_train_sl[sfp_m]):.3f}")

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

    print(f"Train SFPs: {len(train_set)}, Test SFPs: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return train_loader, test_loader, n_features


def evaluate(model, test_loader, criterion):
    """Evaluate and compute metrics including TP/SL ratio quality filter."""
    model.eval()
    all_tp_preds, all_sl_preds = [], []
    all_tp_targets, all_sl_targets = [], []
    all_quality = []
    total_loss = 0

    with torch.no_grad():
        for x, direction, q, tp, sl in test_loader:
            x = x.to(device)
            tp_t = tp.to(device)
            sl_t = sl.to(device)

            tp_pred, sl_pred = model(x)
            loss, _, _ = criterion(tp_pred, sl_pred, tp_t, sl_t)
            total_loss += loss.item()

            all_tp_preds.append(tp_pred.cpu())
            all_sl_preds.append(sl_pred.cpu())
            all_tp_targets.append(tp.cpu())
            all_sl_targets.append(sl.cpu())
            all_quality.append(q.cpu())

    tp_preds = torch.cat(all_tp_preds)
    sl_preds = torch.cat(all_sl_preds)
    tp_targets = torch.cat(all_tp_targets)
    sl_targets = torch.cat(all_sl_targets)
    quality = torch.cat(all_quality)

    tp_mae = (tp_preds - tp_targets).abs().mean().item()
    sl_mae = (sl_preds - sl_targets).abs().mean().item()

    # Test TP/SL ratio as quality filter at different thresholds
    ratio = tp_preds / (sl_preds + 1e-6)
    results = {}
    for thresh in [1.0, 1.5, 2.0, 3.0]:
        take = ratio > thresh
        n_take = take.sum().item()
        if n_take > 0:
            n_profitable = (quality[take] == 1).sum().item()
            precision = n_profitable / n_take * 100
            recall = n_profitable / max((quality == 1).sum().item(), 1) * 100
            avg_tp = tp_targets[take].mean().item()
            avg_sl = sl_targets[take].mean().item()
        else:
            precision = recall = avg_tp = avg_sl = 0
        results[thresh] = (n_take, precision, recall, avg_tp, avg_sl)

    # Log prediction spread
    ratio = tp_preds / (sl_preds + 1e-6)
    prof_ratio = ratio[quality == 1]
    lose_ratio = ratio[quality == 0]
    print(
        f"    Pred spread — "
        f"TP std: {tp_preds.std().item():.3f} | "
        f"SL std: {sl_preds.std().item():.3f} | "
        f"Ratio: prof={prof_ratio.mean().item():.2f} vs lose={lose_ratio.mean().item():.2f}"
    )

    return total_loss / len(test_loader), tp_mae, sl_mae, results


def train():
    train_loader, test_loader, n_features = load_data_set()
    model = SFPTransformer(n_features=n_features).to(device)

    if RESUME and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        print(f"Resumed from {MODEL_FILE}")
    elif RESUME:
        print(f"WARNING: --resume but {MODEL_FILE} not found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SFPTransformer: {n_params:,} parameters")
    resume_lr = 1e-4 if RESUME else 5e-4
    criterion = RankingRegLoss(margin=0.3, lambda_rank=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=resume_lr, epochs=100,
        steps_per_epoch=len(train_loader),
    )

    best_score = -1.0  # track best precision-based score
    best_loss = float("inf")
    counter = 0
    min_trades = 50  # minimum trades for a valid precision reading
    epochs = 100

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for x, direction, q, tp, sl in train_loader:
            x = x.to(device)
            tp_t = tp.to(device)
            sl_t = sl.to(device)
            q_t = q.to(device)

            tp_pred, sl_pred = model(x)
            loss, _, _ = criterion(tp_pred, sl_pred, tp_t, sl_t, quality=q_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # --- Evaluate ---
        test_loss, tp_mae, sl_mae, ratio_results = evaluate(model, test_loader, criterion)

        avg_train_loss = train_loss / len(train_loader)

        # Main log line
        r = ratio_results.get(1.5, (0, 0, 0, 0, 0))
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"TP MAE: {tp_mae:.3f} | "
            f"SL MAE: {sl_mae:.3f} | "
            f"Ratio>1.5: {r[0]} trades, {r[1]:.0f}% prec, {r[2]:.0f}% recall"
        )

        # Every 10 epochs, print full ratio analysis
        if (epoch + 1) % 10 == 0:
            print("  --- Ratio threshold analysis ---")
            for thresh, (n, prec, rec, avg_tp, avg_sl) in sorted(ratio_results.items()):
                print(
                    f"    TP/SL > {thresh}: {n} trades | "
                    f"Prec: {prec:.0f}% | Recall: {rec:.0f}% | "
                    f"Avg TP: {avg_tp:.2f} | Avg SL: {avg_sl:.2f}"
                )

        # --- Save best model by precision * R:R at ratio > 1.5 ---
        n_trades, prec, rec, avg_tp, avg_sl = ratio_results.get(1.5, (0, 0, 0, 0, 0))
        if n_trades >= min_trades and prec > 0 and avg_sl > 0:
            rr = avg_tp / avg_sl
            score = prec * rr  # precision weighted by R:R
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), MODEL_FILE)
                print(
                    f"  -> Saved best model (score: {score:.1f}, "
                    f"prec: {prec:.0f}%, R:R: {rr:.2f}, trades: {n_trades})"
                )

        # --- Early stopping on test loss (still useful to stop divergence) ---
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= 20:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # --- Final evaluation with best model ---
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, tp_mae, sl_mae, ratio_results = evaluate(model, test_loader, criterion)
    print(f"\n{'=' * 60}")
    print(f"Best model — TP MAE: {tp_mae:.3f}, SL MAE: {sl_mae:.3f} (normalized units)")
    print(f"{'=' * 60}")
    for thresh, (n, prec, rec, avg_tp, avg_sl) in sorted(ratio_results.items()):
        rr = avg_tp / avg_sl if avg_sl > 0 else 0
        print(
            f"  TP/SL > {thresh}: {n} trades | "
            f"Prec: {prec:.0f}% | R:R: {rr:.2f} | Recall: {rec:.0f}% | "
            f"Avg TP: {avg_tp:.2f} | Avg SL: {avg_sl:.2f}"
        )


train()
