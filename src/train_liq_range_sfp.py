"""Liq+Range+SFP model training: P(profitable) gate + MFE quantile regression.

Usage:
    python -m src.train_liq_range_sfp              # all TFs, all assets
    python -m src.train_liq_range_sfp 1h           # 1h only
    python -m src.train_liq_range_sfp 4h 1h 15min --assets btc gold
    python -m src.train_liq_range_sfp 4h 1h 15min --resume
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.liq_sfp_labels import generate_labels
from src.labels.range_sfp_labels import detect_market_structure
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

MODEL_FILE = "best_model_liq_range_sfp.pth"

TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
WINDOW_BY_TF = {"15m": 120, "1h": 48, "4h": 30}
TRAIN_START = "2018-01-01"
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Start: {TRAIN_START} | Model: {MODEL_FILE}")

N_FEATURES = 30


def build_features(df, actions, signal_map, tf_hours, asset_id=1.0):
    """Build 30 Liq+Range+SFP features.

    6 range + 6 liquidation + 6 SFP candle + 6 context + 6 range fingerprint.
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.ones(n)

    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    feat = pd.DataFrame()

    # --- Range features (6) ---
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_norm = np.zeros(n, dtype=np.float32)
    range_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth_range = np.zeros(n, dtype=np.float32)
    reclaim_strength_range = np.zeros(n, dtype=np.float32)

    # --- Liq features (6) ---
    n_liq_swept_norm = np.zeros(n, dtype=np.float32)
    weighted_liq_swept = np.zeros(n, dtype=np.float32)
    max_leverage_norm = np.zeros(n, dtype=np.float32)
    liq_cascade_depth = np.zeros(n, dtype=np.float32)
    liq_cluster_density = np.zeros(n, dtype=np.float32)
    n_swings_with_liq_norm = np.zeros(n, dtype=np.float32)

    # --- SFP candle features (6) ---
    body_ratio = np.zeros(n, dtype=np.float32)
    wick_ratio = np.zeros(n, dtype=np.float32)
    vol_spike = np.zeros(n, dtype=np.float32)
    close_position = np.zeros(n, dtype=np.float32)
    zone_sl_dist = np.zeros(n, dtype=np.float32)
    zone_tp_dist = np.zeros(n, dtype=np.float32)

    # Precompute 20-bar volume average
    vol_ma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i, sig in signal_map.items():
        r = sig.range_ref
        range_h = r.high - r.low
        entry = sig.swept_level

        # Range features
        range_height_pct[i] = sig.range_height_pct
        range_touches_norm[i] = min(sig.range_touches, 5) / 5.0
        range_concentration[i] = sig.range_concentration
        range_age[i] = sig.range_age
        sweep_depth_range[i] = sig.sweep_depth_range
        reclaim_strength_range[i] = sig.reclaim_strength_range

        # Liq features
        n_liq_swept_norm[i] = min(sig.n_liq_swept, 30) / 30.0
        weighted_liq_swept[i] = min(sig.weighted_liq_swept, 3.0) / 3.0
        max_leverage_norm[i] = sig.max_leverage_swept / 100.0
        local_atr = atr[i] if atr[i] > 0 else 1e-8
        liq_cascade_depth[i] = np.clip(sig.liq_cascade_depth / local_atr, 0, 5)
        liq_cluster_density[i] = sig.liq_cluster_density
        n_swings_with_liq_norm[i] = min(sig.n_swings_with_liq, 10) / 10.0

        # SFP candle features
        candle_range = highs[i] - lows[i]
        if candle_range > 0:
            body_ratio[i] = (closes[i] - opens[i]) / candle_range
            if sig.direction == 1:
                wick_ratio[i] = (r.support.top - lows[i]) / candle_range
                close_position[i] = (closes[i] - lows[i]) / candle_range
            else:
                wick_ratio[i] = (highs[i] - r.resistance.bottom) / candle_range
                close_position[i] = (highs[i] - closes[i]) / candle_range

        vol_spike[i] = volumes[i] / (vol_ma20[i] + 1e-8)

        # Zone SL/TP distances
        if entry > 0:
            if sig.direction == 1:
                zone_sl_dist[i] = (entry - r.support.bottom) / entry
                zone_tp_dist[i] = (r.resistance.top - entry) / entry
            else:
                zone_sl_dist[i] = (r.resistance.top - entry) / entry
                zone_tp_dist[i] = (entry - r.support.bottom) / entry

    feat["range_height_pct"] = range_height_pct
    feat["range_touches_norm"] = range_touches_norm
    feat["range_concentration"] = range_concentration
    feat["range_age"] = range_age
    feat["sweep_depth_range"] = sweep_depth_range
    feat["reclaim_strength_range"] = reclaim_strength_range

    feat["n_liq_swept_norm"] = n_liq_swept_norm
    feat["weighted_liq_swept"] = weighted_liq_swept
    feat["max_leverage_norm"] = max_leverage_norm
    feat["liq_cascade_depth"] = liq_cascade_depth
    feat["liq_cluster_density"] = liq_cluster_density
    feat["n_swings_with_liq"] = n_swings_with_liq_norm

    feat["body_ratio"] = body_ratio
    feat["wick_ratio"] = wick_ratio
    feat["vol_spike"] = vol_spike
    feat["close_position"] = close_position
    feat["zone_sl_dist"] = zone_sl_dist
    feat["zone_tp_dist"] = zone_tp_dist

    # --- Context features (6) ---
    feat["rsi"] = df["rsi"].values / 100.0 if "rsi" in df.columns else 0.5
    feat["trend_strength"] = ((df["Close"] - df["ema_21"]) / df["Close"]).values if "ema_21" in df.columns else 0.0
    feat["ms_alignment"] = np.zeros(n, dtype=np.float32)
    feat["ms_strength"] = ms_strength_arr

    for i, sig in signal_map.items():
        feat.at[i, "ms_alignment"] = sig.ms_alignment

    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id

    # --- Range fingerprint features (6) ---
    signal_type_arr = np.zeros(n, dtype=np.float32)
    is_recaptured_arr = np.zeros(n, dtype=np.float32)
    is_nested_arr = np.zeros(n, dtype=np.float32)
    touch_symmetry_arr = np.zeros(n, dtype=np.float32)
    boundary_rejection_avg_arr = np.zeros(n, dtype=np.float32)
    range_position_arr = np.zeros(n, dtype=np.float32)

    for i, sig in signal_map.items():
        signal_type_arr[i] = float(sig.signal_type)
        is_recaptured_arr[i] = sig.is_recaptured
        is_nested_arr[i] = sig.is_nested
        touch_symmetry_arr[i] = sig.touch_symmetry
        r = sig.range_ref
        range_h = r.high - r.low
        boundary_rejection_avg_arr[i] = np.clip(sig.boundary_rejection_avg, 0, 2.0)
        range_position_arr[i] = sig.range_position

    feat["signal_type"] = signal_type_arr
    feat["is_recaptured"] = is_recaptured_arr
    feat["is_nested"] = is_nested_arr
    feat["touch_symmetry"] = touch_symmetry_arr
    feat["boundary_rejection_avg"] = boundary_rejection_avg_arr
    feat["range_position"] = range_position_arr

    # Drop warmup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}

    # Clean up
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_spike"] = feat["vol_spike"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth_range"] = feat["sweep_depth_range"].clip(0, 2.0)
    feat["reclaim_strength_range"] = feat["reclaim_strength_range"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_dist"] = feat["zone_sl_dist"].clip(0, 0.10)
    feat["zone_tp_dist"] = feat["zone_tp_dist"].clip(0, 0.15)

    return feat, actions, signal_map_shifted


def _process_one_tf(args):
    """Process a single asset/TF combo — runs in a worker process."""
    asset_name, prefix, asset_id, tf, tf_key, tf_hours = args
    data_file = f"data/{prefix}_{tf}.csv"
    if not os.path.exists(data_file):
        return None

    df = pd.read_csv(data_file).reset_index(drop=True)
    if "timestamp" in df.columns:
        df = df[df["timestamp"] >= TRAIN_START].reset_index(drop=True)

    actions, quality, mfe, sl_labels, swept_levels, signal_map = generate_labels(
        df["High"].values, df["Low"].values,
        df["Close"].values, df["Open"].values,
        volumes=df["Volume"].values if "Volume" in df.columns else None,
        tf_key=tf_key,
    )

    feat, actions, signal_map_shifted = build_features(
        df, actions, signal_map, tf_hours, asset_id=asset_id,
    )

    drop_n = 30
    quality = quality[drop_n:]
    mfe = mfe[drop_n:]
    sl_labels = sl_labels[drop_n:]

    feat_values = feat.values.astype(np.float32)
    signal_mask = actions != 0
    total_signals = int(np.sum(signal_mask))
    n_profitable = int(np.sum(quality[signal_mask] == 1))

    if total_signals == 0:
        return None

    split_idx = int(len(feat_values) * 0.8)
    return {
        "tf_key": tf_key,
        "label": f"{asset_name}/{tf_key}",
        "n_bars": len(feat_values),
        "n_signals": total_signals,
        "n_profitable": n_profitable,
        "train_feat": feat_values[:split_idx],
        "train_actions": actions[:split_idx],
        "train_quality": quality[:split_idx],
        "train_mfe": mfe[:split_idx],
        "train_sl": sl_labels[:split_idx],
        "test_feat": feat_values[split_idx:],
        "test_actions": actions[split_idx:],
        "test_quality": quality[split_idx:],
        "test_mfe": mfe[split_idx:],
        "test_sl": sl_labels[split_idx:],
    }


def load_data_set():
    from multiprocessing import Pool


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

    # Group results by tf_key
    tf_groups = {}
    for result in results:
        if result is None:
            continue

        label = result["label"]
        print(f"  {label}: {result['n_bars']} bars, {result['n_signals']} signals, "
              f"{result['n_profitable']} profitable ({result['n_profitable']/result['n_signals']*100:.0f}%)")

        tk = result["tf_key"]
        if tk not in tf_groups:
            tf_groups[tk] = {k: [] for k in [
                "train_feat", "train_actions",
                "train_quality", "train_mfe", "train_sl",
                "test_feat", "test_actions",
                "test_quality", "test_mfe", "test_sl",
            ]}
        g = tf_groups[tk]
        for split in ["train", "test"]:
            g[f"{split}_feat"].append(result[f"{split}_feat"])
            g[f"{split}_actions"].append(result[f"{split}_actions"])
            for k in ["quality", "mfe", "sl"]:
                g[f"{split}_{k}"].append(result[f"{split}_{k}"])

    if not tf_groups:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    # Fit scaler on ALL training data combined
    all_train = np.concatenate([np.concatenate(g["train_feat"]) for g in tf_groups.values()])
    n_features = all_train.shape[1]
    total_train = len(all_train)
    total_test = sum(sum(len(f) for f in g["test_feat"]) for g in tf_groups.values())

    scaler = StandardScaler()
    scaler.fit(all_train)
    joblib.dump(scaler, "liq_range_sfp_scaler.joblib")
    print(f"\nScaler fit on {total_train} train bars, saved to liq_range_sfp_scaler.joblib")

    # Create per-TF datasets with TF-specific windows
    train_loaders = {}
    test_loaders = {}

    for tk, g in tf_groups.items():
        window = WINDOW_BY_TF.get(tk, 30)
        train_feat = scaler.transform(np.concatenate(g["train_feat"]))
        test_feat = scaler.transform(np.concatenate(g["test_feat"]))

        train_set = SFPDataset(
            train_feat,
            np.concatenate(g["train_actions"]),
            np.concatenate(g["train_quality"]),
            np.concatenate(g["train_mfe"]),
            np.concatenate(g["train_sl"]),
            window=window,
        )
        test_set = SFPDataset(
            test_feat,
            np.concatenate(g["test_actions"]),
            np.concatenate(g["test_quality"]),
            np.concatenate(g["test_mfe"]),
            np.concatenate(g["test_sl"]),
            window=window,
        )

        print(f"  {tk} (window={window}): {len(train_set)} train, {len(test_set)} test signals")
        train_loaders[tk] = DataLoader(train_set, batch_size=128, shuffle=True)
        test_loaders[tk] = DataLoader(test_set, batch_size=128, shuffle=False)

    total_train_signals = sum(len(loader.dataset) for loader in train_loaders.values())
    total_test_signals = sum(len(loader.dataset) for loader in test_loaders.values())
    print(f"\nCombined — {total_train} train bars ({total_train_signals} signals) | {total_test} test bars ({total_test_signals} signals)")

    return train_loaders, test_loaders, n_features


def quantile_loss(pred, target, tau):
    """Pinball / quantile loss."""
    err = target - pred
    return torch.max(tau * err, (tau - 1) * err).mean()


def evaluate(model, test_loaders):
    """Evaluate: classification gate + MFE-based WR at predicted TP levels.

    Returns (test_loss, results_dict).
    """
    model.eval()
    all_cls_prob = []
    all_tp1_pred = []
    all_tp2_pred = []
    all_quality = []
    all_mfe = []
    all_sl = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for loader in test_loaders.values():
            for x, direction, q, mfe, sl in loader:
                x = x.to(device)
                q_t = q.to(device).float()
                mfe_t = mfe.to(device).float()

                out = model(x)  # (B, 3)
                cls_logit = out[:, 0]
                tp1_pred = out[:, 1]
                tp2_pred = out[:, 2]

                cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_logit, q_t)
                total_loss += cls_loss.item()
                n_batches += 1

                all_cls_prob.append(torch.sigmoid(cls_logit).cpu())
                all_tp1_pred.append(tp1_pred.cpu())
                all_tp2_pred.append(tp2_pred.cpu())
                all_quality.append(q.cpu())
                all_mfe.append(mfe.cpu())
                all_sl.append(sl.cpu())

    cls_prob = torch.cat(all_cls_prob)    # (N,)
    tp1_pred = torch.cat(all_tp1_pred)   # (N,)
    tp2_pred = torch.cat(all_tp2_pred)   # (N,)
    quality = torch.cat(all_quality)     # (N,)
    mfe = torch.cat(all_mfe)            # (N,)
    sl = torch.cat(all_sl)              # (N,)

    base_wr = (quality == 1).float().mean().item() * 100
    avg_tp1_all = tp1_pred.mean().item() * 100
    avg_tp2_all = tp2_pred.mean().item() * 100
    print(f"    Base WR: {base_wr:.0f}% | Pred TP1: {avg_tp1_all:.2f}% | Pred TP2: {avg_tp2_all:.2f}%")

    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        take = cls_prob > thresh
        n_take = take.sum().item()
        if n_take == 0:
            results[thresh] = {"n": 0, "wr": 0, "tp1_wr": 0, "tp2_wr": 0, "ev1": 0, "ev2": 0}
            continue

        # Gate WR: fraction of taken trades that are profitable (MFE > SL)
        q_taken = quality[take]
        wr = (q_taken == 1).float().mean().item() * 100

        # TP1 WR: MFE >= predicted TP1
        mfe_taken = mfe[take]
        sl_taken = sl[take]
        tp1_taken = tp1_pred[take]
        tp2_taken = tp2_pred[take]

        tp1_wr = (mfe_taken >= tp1_taken).float().mean().item() * 100
        tp2_wr = (mfe_taken >= tp2_taken).float().mean().item() * 100

        # EV: WR * avg_TP - (1-WR) * avg_SL
        avg_tp1 = tp1_taken.mean().item() * 100
        avg_tp2 = tp2_taken.mean().item() * 100
        avg_sl_taken = sl_taken.mean().item() * 100
        ev1 = (tp1_wr / 100) * avg_tp1 - (1 - tp1_wr / 100) * avg_sl_taken
        ev2 = (tp2_wr / 100) * avg_tp2 - (1 - tp2_wr / 100) * avg_sl_taken

        results[thresh] = {
            "n": n_take, "wr": wr,
            "tp1_wr": tp1_wr, "tp2_wr": tp2_wr,
            "avg_tp1": avg_tp1, "avg_tp2": avg_tp2,
            "avg_sl": avg_sl_taken,
            "ev1": ev1, "ev2": ev2,
        }

    return total_loss / max(n_batches, 1), results


def train():

    WINDOW = max(WINDOW_BY_TF.values())

    train_loaders, test_loaders, n_features = load_data_set()

    from src.models.liq_range_sfp_model import LiqRangeSFPClassifier
    model = LiqRangeSFPClassifier(n_features=n_features, window=WINDOW, hidden=32).to(device)

    if RESUME and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        print(f"Resumed from {MODEL_FILE}")
    elif RESUME:
        print(f"WARNING: --resume but {MODEL_FILE} not found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LiqRangeSFPClassifier: {n_params:,} parameters")

    # Compute class weight for gate head
    all_q = []
    for loader in train_loaders.values():
        for x, direction, q, mfe, sl in loader:
            all_q.append(q)
    all_q = torch.cat(all_q)  # (N,)
    n_pos = (all_q == 1).sum().item()
    n_neg = (all_q == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  Gate: wins={n_pos}, losses={n_neg}, pos_weight={pos_weight.item():.2f}")

    resume_lr = 1e-4 if RESUME else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=1e-3)
    total_train_batches = sum(len(loader) for loader in train_loaders.values())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=resume_lr, epochs=200,
        steps_per_epoch=total_train_batches,
    )

    best_score = -float("inf")
    best_loss = float("inf")
    counter = 0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for loader in train_loaders.values():
            for x, direction, q, mfe, sl in loader:
                x = x.to(device)
                q_t = q.to(device).float()      # (B,)
                mfe_t = mfe.to(device).float()   # (B,)
                sl_t = sl.to(device).float()     # (B,)

                out = model(x)                   # (B, 3)
                cls_logit = out[:, 0]             # (B,)
                tp1_pred = out[:, 1]              # (B,)
                tp2_pred = out[:, 2]              # (B,)

                # Classification loss: BCE on quality gate
                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    cls_logit, q_t, pos_weight=pos_weight,
                )

                # Regression loss: quantile on MFE, only for profitable signals
                profitable_mask = q_t > 0.5
                if profitable_mask.any():
                    mfe_prof = mfe_t[profitable_mask]
                    tp1_prof = tp1_pred[profitable_mask]
                    tp2_prof = tp2_pred[profitable_mask]
                    reg_loss = (
                        quantile_loss(tp1_prof, mfe_prof, tau=0.3) +
                        quantile_loss(tp2_prof, mfe_prof, tau=0.7)
                    )
                else:
                    reg_loss = torch.tensor(0.0, device=device)

                loss = cls_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                n_batches += 1

        test_loss, results = evaluate(model, test_loaders)
        avg_train_loss = train_loss / max(n_batches, 1)

        # Epoch summary at P>0.5
        r50 = results.get(0.5, {"n": 0, "tp1_wr": 0, "tp2_wr": 0, "ev1": 0, "ev2": 0})
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} / {test_loss:.4f} | "
            f"n={r50['n']} | "
            f"TP1: {r50['tp1_wr']:.0f}%WR EV={r50['ev1']:+.2f}% | "
            f"TP2: {r50['tp2_wr']:.0f}%WR EV={r50['ev2']:+.2f}%"
        )

        if (epoch + 1) % 10 == 0:
            print(f"  --- Threshold analysis ---")
            for thresh in sorted(results.keys()):
                r = results[thresh]
                if r["n"] == 0:
                    continue
                print(
                    f"    P > {thresh}: {r['n']} trades | "
                    f"Gate WR: {r['wr']:.0f}% | "
                    f"TP1: {r['tp1_wr']:.0f}%WR TP={r.get('avg_tp1',0):.2f}% EV={r['ev1']:+.3f}% | "
                    f"TP2: {r['tp2_wr']:.0f}%WR TP={r.get('avg_tp2',0):.2f}% EV={r['ev2']:+.3f}% | "
                    f"SL={r.get('avg_sl',0):.2f}%"
                )

        # Save best model by TP1 EV score
        score = 0.0
        for thresh, weight in [(0.4, 1.0), (0.5, 2.0), (0.6, 3.0)]:
            r = results.get(thresh, {"n": 0, "ev1": 0, "tp1_wr": 0})
            n_t = r["n"]
            ev = r["ev1"]
            if n_t >= 10 and ev > 0:
                score += weight * ev * min(n_t, 200)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"  -> Saved best model (TP1 score: {score:.1f})")

        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation with best model
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, results = evaluate(model, test_loaders)
    print(f"\n{'=' * 60}")
    print(f"Best model — MFE regression results")
    print(f"{'=' * 60}")
    for thresh in sorted(results.keys()):
        r = results[thresh]
        if r["n"] == 0:
            continue
        print(
            f"  P > {thresh}: {r['n']} trades | "
            f"Gate WR: {r['wr']:.0f}% | "
            f"TP1: {r['tp1_wr']:.0f}%WR TP={r.get('avg_tp1',0):.2f}% EV={r['ev1']:+.3f}% | "
            f"TP2: {r['tp2_wr']:.0f}%WR TP={r.get('avg_tp2',0):.2f}% EV={r['ev2']:+.3f}% | "
            f"SL={r.get('avg_sl',0):.2f}%"
        )


if __name__ == "__main__":
    train()
