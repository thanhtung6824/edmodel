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
from src.labels.three_tap_labels import detect_ranges, compute_atr, RANGE_PARAMS
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
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Model: {MODEL_FILE}")

class TPFocusedLoss(nn.Module):
    """TP-focused loss: SL is structural (known), so focus model capacity on TP.

    - TP regression: main loss + asymmetric penalty for overestimating losers' TP
    - SL regression: low weight (structural, easy to learn)
    - R:R ratio: uses predicted_TP / SL_TARGET (not predicted SL)
    - Ranking: pushes winners' predicted TP higher than losers'
    """

    def __init__(self, margin=1.0, lambda_rank=3.0, lambda_ratio=1.0, sl_weight=0.5,
                 lambda_overest=2.5, lambda_underest=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.lambda_ratio = lambda_ratio
        self.sl_weight = sl_weight
        self.lambda_overest = lambda_overest
        self.lambda_underest = lambda_underest

    def forward(self, tp_pred, sl_pred, tp_target, sl_target, quality=None):
        tp_loss = F.smooth_l1_loss(tp_pred, tp_target)
        sl_loss = F.smooth_l1_loss(sl_pred, sl_target)
        reg_loss = tp_loss + self.sl_weight * sl_loss

        # R:R ratio uses SL TARGET (structural, known) — not predicted SL
        ratio_pred = tp_pred / (sl_target + 1e-6)
        ratio_target = tp_target / (sl_target + 1e-6)
        ratio_loss = F.smooth_l1_loss(ratio_pred, ratio_target)

        if quality is None:
            total = reg_loss + self.lambda_ratio * ratio_loss
            return total, tp_loss, sl_loss

        prof_mask = quality == 1
        lose_mask = quality == 0
        n_prof = prof_mask.sum()
        n_lose = lose_mask.sum()

        # Asymmetric TP penalties — sharpen separation from BOTH sides:
        # 1. Penalize overestimating losers' TP (prevent false positives)
        # 2. Penalize underestimating winners' TP (preserve true positives)
        asym_loss = torch.tensor(0.0, device=tp_pred.device)
        if n_lose > 0:
            tp_error_lose = tp_pred[lose_mask] - tp_target[lose_mask]
            overest = torch.clamp(tp_error_lose, min=0)
            asym_loss = asym_loss + self.lambda_overest * (overest ** 2).mean()
        if n_prof > 0:
            tp_error_prof = tp_target[prof_mask] - tp_pred[prof_mask]
            underest = torch.clamp(tp_error_prof, min=0)
            asym_loss = asym_loss + self.lambda_underest * (underest ** 2).mean()

        if n_prof == 0 or n_lose == 0:
            total = reg_loss + self.lambda_ratio * ratio_loss + asym_loss
            return total, tp_loss, sl_loss

        # Rank by predicted_TP / known_SL — focus ranking on TP discrimination
        prof_ratios = ratio_pred[prof_mask]
        lose_ratios = ratio_pred[lose_mask]

        prof_exp = prof_ratios.unsqueeze(1).expand(n_prof, n_lose).reshape(-1)
        lose_exp = lose_ratios.unsqueeze(0).expand(n_prof, n_lose).reshape(-1)
        target = torch.ones_like(prof_exp)

        rank_loss = F.margin_ranking_loss(
            prof_exp, lose_exp, target, margin=self.margin
        )

        total = (reg_loss + self.lambda_rank * rank_loss
                 + self.lambda_ratio * ratio_loss + asym_loss)
        return total, tp_loss, sl_loss

def build_features(df, actions, tf_hours, asset_id=1.0, tf_key="4h"):
    """Build 23 features for one timeframe's data."""
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

    # Range boundary feature: is the swept level near a range high/low?
    at_range_boundary = np.zeros(len(df), dtype=np.float32)
    atr_vals = compute_atr(highs, lows, closes, period=14)
    params = RANGE_PARAMS.get(tf_key, RANGE_PARAMS["4h"])
    wp = params["wide"]
    _, wide_all = detect_ranges(
        highs, lows, closes, atr_vals,
        n_swing=wp["n_swing"], min_bars=wp["min_bars"], max_bars=wp["max_bars"],
        min_atr_mult=wp["min_atr_mult"], max_atr_mult=wp["max_atr_mult"],
        tolerance=0.01, touch_tolerance=0.005, min_touches=2,
    )
    # Build per-bar active ranges
    n_bars = len(highs)
    active_ranges = [[] for _ in range(n_bars)]
    for r in wide_all:
        for j in range(r.confirmed, min(r.confirmed + 500, n_bars)):
            if highs[j] > r.high * 1.03 or lows[j] < r.low * 0.97:
                break
            active_ranges[j].append(r)
    # For each SFP signal, check if swept level is near a range boundary
    nearest_sh_5, nearest_sl_5 = swing_levels[5]
    for i in range(n_bars):
        if actions[i] == 0:
            continue
        swept = nearest_sl_5[i] if actions[i] == 1 else nearest_sh_5[i]
        if np.isnan(swept) or not active_ranges[i]:
            continue
        for r in active_ranges[i]:
            rh, rl = r.high, r.low
            range_h = rh - rl
            tol = range_h * 0.003
            if swept >= rh - tol or swept <= rl + tol:
                at_range_boundary[i] = 1.0
                break
    feat["at_range_boundary"] = at_range_boundary

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
            tf_key = TF_KEYS[tf]
            feat, actions = build_features(df, actions, tf_hours, asset_id=asset_id, tf_key=tf_key)

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


def evaluate(model, test_loader, criterion, q_criterion):
    """Evaluate with TP/SL ratio + P(win) quality gate."""
    model.eval()
    all_tp_preds, all_sl_preds, all_q_logits = [], [], []
    all_tp_targets, all_sl_targets = [], []
    all_quality = []
    total_loss = 0

    with torch.no_grad():
        for x, direction, q, tp, sl in test_loader:
            x = x.to(device)
            tp_t = tp.to(device)
            sl_t = sl.to(device)
            q_t = q.to(device)

            tp_pred, sl_pred, q_logit = model(x)
            loss, _, _ = criterion(tp_pred, sl_pred, tp_t, sl_t)
            q_loss = q_criterion(q_logit, q_t)
            total_loss += (loss + q_loss).item()

            all_tp_preds.append(tp_pred.cpu())
            all_sl_preds.append(sl_pred.cpu())
            all_q_logits.append(q_logit.cpu())
            all_tp_targets.append(tp.cpu())
            all_sl_targets.append(sl.cpu())
            all_quality.append(q.cpu())

    tp_preds = torch.cat(all_tp_preds)
    sl_preds = torch.cat(all_sl_preds)
    q_logits = torch.cat(all_q_logits)
    tp_targets = torch.cat(all_tp_targets)
    sl_targets = torch.cat(all_sl_targets)
    quality = torch.cat(all_quality)

    tp_mae = (tp_preds - tp_targets).abs().mean().item()
    sl_mae = (sl_preds - sl_targets).abs().mean().item()

    # P(win) accuracy
    p_win = torch.sigmoid(q_logits)
    q_pred_binary = (p_win > 0.5).float()
    q_acc = (q_pred_binary == quality).float().mean().item() * 100

    # R:R ratio uses predicted_TP / SL_TARGET (SL is structural, known)
    ratio = tp_preds / (sl_targets + 1e-6)

    # Results: ratio-only thresholds
    results = {}
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        take = ratio > thresh
        n_take = take.sum().item()
        if n_take > 0:
            n_profitable = (quality[take] == 1).sum().item()
            precision = n_profitable / n_take * 100
            recall = n_profitable / max((quality == 1).sum().item(), 1) * 100
            avg_tp = tp_targets[take].mean().item()
            avg_sl = sl_targets[take].mean().item()
            rr = avg_tp / (avg_sl + 1e-6)
        else:
            precision = recall = avg_tp = avg_sl = rr = 0
        results[thresh] = (n_take, precision, recall, avg_tp, avg_sl, rr)

    # Results: ratio + P(win) combined gate
    combined_results = {}
    for ratio_thresh in [1.5, 2.0, 2.5, 3.0]:
        for q_thresh in [0.4, 0.5, 0.6]:
            take = (ratio > ratio_thresh) & (p_win > q_thresh)
            n_take = take.sum().item()
            if n_take > 0:
                n_profitable = (quality[take] == 1).sum().item()
                precision = n_profitable / n_take * 100
                avg_tp = tp_targets[take].mean().item()
                avg_sl = sl_targets[take].mean().item()
                rr = avg_tp / (avg_sl + 1e-6)
            else:
                precision = avg_tp = avg_sl = rr = 0
            combined_results[(ratio_thresh, q_thresh)] = (n_take, precision, avg_tp, avg_sl, rr)

    # Log prediction spread
    prof_ratio = ratio[quality == 1]
    lose_ratio = ratio[quality == 0]
    print(
        f"    Pred spread — "
        f"TP std: {tp_preds.std().item():.3f} | "
        f"SL std: {sl_preds.std().item():.3f} | "
        f"Ratio: prof={prof_ratio.mean().item():.2f} vs lose={lose_ratio.mean().item():.2f} | "
        f"P(win): wins={p_win[quality==1].mean().item():.3f} vs loses={p_win[quality==0].mean().item():.3f} | "
        f"Q acc: {q_acc:.0f}%"
    )

    return total_loss / len(test_loader), tp_mae, sl_mae, results, combined_results


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
    resume_lr = 1e-4 if RESUME else 3e-4
    criterion = TPFocusedLoss(margin=1.5, lambda_rank=3.0, lambda_ratio=1.0,
                              sl_weight=0.5, lambda_overest=2.5, lambda_underest=0.5)

    # Quality head loss: P(win) classification
    # Compute class weight from training data
    all_q_train = []
    for x, direction, q, tp, sl in train_loader:
        all_q_train.append(q)
    all_q_train = torch.cat(all_q_train)
    n_wins = (all_q_train == 1).sum().item()
    n_losses = (all_q_train == 0).sum().item()
    pos_weight = torch.tensor([n_losses / max(n_wins, 1)]).to(device)
    print(f"Quality head — wins: {n_wins}, losses: {n_losses}, pos_weight: {pos_weight.item():.2f}")
    q_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=resume_lr, epochs=150,
        steps_per_epoch=len(train_loader),
    )

    best_score = -1.0  # track best precision-based score
    best_loss = float("inf")
    counter = 0
    min_trades = 50  # minimum trades for a valid precision reading
    epochs = 150

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for x, direction, q, tp, sl in train_loader:
            x = x.to(device)
            tp_t = tp.to(device)
            sl_t = sl.to(device)
            q_t = q.to(device)

            tp_pred, sl_pred, q_logit = model(x)
            tp_sl_loss, _, _ = criterion(tp_pred, sl_pred, tp_t, sl_t, quality=q_t)
            q_loss = q_criterion(q_logit, q_t)
            loss = tp_sl_loss + q_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # --- Evaluate ---
        test_loss, tp_mae, sl_mae, ratio_results, combined_results = evaluate(
            model, test_loader, criterion, q_criterion
        )

        avg_train_loss = train_loss / len(train_loader)

        # Main log line
        r = ratio_results.get(1.5, (0, 0, 0, 0, 0, 0))
        # Combined: ratio>2.0 + P(win)>0.5
        c = combined_results.get((2.0, 0.5), (0, 0, 0, 0, 0))
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"TP MAE: {tp_mae:.3f} | "
            f"SL MAE: {sl_mae:.3f} | "
            f"Ratio>1.5: {r[0]}t {r[1]:.0f}% | "
            f"R>2+Q>0.5: {c[0]}t {c[1]:.0f}%"
        )

        # Every 10 epochs, print full analysis
        if (epoch + 1) % 10 == 0:
            print("  --- Ratio-only thresholds ---")
            for thresh, (n, prec, rec, avg_tp, avg_sl, rr) in sorted(ratio_results.items()):
                print(
                    f"    TP/SL > {thresh}: {n} trades | "
                    f"Prec: {prec:.0f}% | R:R: {rr:.2f} | Recall: {rec:.0f}% | "
                    f"Avg TP: {avg_tp:.2f} | Avg SL: {avg_sl:.2f}"
                )
            print("  --- Ratio + P(win) combined gate ---")
            for (rt, qt), (n, prec, avg_tp, avg_sl, rr) in sorted(combined_results.items()):
                if n > 0:
                    print(
                        f"    R>{rt} + Q>{qt}: {n} trades | "
                        f"Prec: {prec:.0f}% | R:R: {rr:.2f}"
                    )

        # --- Save best model by multi-threshold score ---
        # Score: ratio-only + combined gate bonus
        score = 0.0
        score_parts = []
        for sel_thresh, weight in [(1.5, 1.0), (2.0, 2.0), (2.5, 3.0)]:
            n_t, p, _, atp, asl, rr = ratio_results.get(sel_thresh, (0, 0, 0, 0, 0, 0))
            if n_t >= 5 and p > 0 and rr > 0:
                score += weight * p * rr
                score_parts.append(f">{sel_thresh}: {n_t}t/{p:.0f}%/R:{rr:.1f}")
        # Bonus for combined gate lifting precision
        for (rt, qt), (n, prec, atp, asl, rr) in combined_results.items():
            if n >= 5 and prec > 0 and qt == 0.5:
                score += 0.5 * prec * rr  # lighter weight, bonus only
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_FILE)
            print(
                f"  -> Saved best model (score: {score:.1f}, "
                f"{', '.join(score_parts) if score_parts else 'n/a'})"
            )

        # --- Early stopping on test loss (still useful to stop divergence) ---
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= 30:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # --- Final evaluation with best model ---
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, tp_mae, sl_mae, ratio_results, combined_results = evaluate(
        model, test_loader, criterion, q_criterion
    )
    print(f"\n{'=' * 60}")
    print(f"Best model — TP MAE: {tp_mae:.3f}, SL MAE: {sl_mae:.3f} (normalized units)")
    print(f"{'=' * 60}")
    print("  --- Ratio-only ---")
    for thresh, (n, prec, rec, avg_tp, avg_sl, rr) in sorted(ratio_results.items()):
        print(
            f"  TP/SL > {thresh}: {n} trades | "
            f"Prec: {prec:.0f}% | R:R: {rr:.2f} | Recall: {rec:.0f}% | "
            f"Avg TP: {avg_tp:.2f} | Avg SL: {avg_sl:.2f}"
        )
    print("  --- Ratio + P(win) gate ---")
    for (rt, qt), (n, prec, avg_tp, avg_sl, rr) in sorted(combined_results.items()):
        if n > 0:
            print(
                f"  R>{rt} + Q>{qt}: {n} trades | "
                f"Prec: {prec:.0f}% | R:R: {rr:.2f}"
            )


train()
