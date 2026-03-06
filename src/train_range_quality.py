"""Per-bar Range Detector training: predict P(in_tradeable_range) per bar.

100K+ samples from per-bar labeling. Window-based model with BCEWithLogitsLoss.

Usage:
    python -m src.train_range_quality              # all TFs, all assets
    python -m src.train_range_quality 4h 1h 15min --assets btc gold
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.range_quality_labels import (
    build_per_bar_features,
    generate_per_bar_labels,
)
from src.models.range_quality import RangeDetector, RangeDetectorDataset

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

MODEL_FILE = "best_model_range_detector.pth"
N_FEATURES = 14
WINDOW = 60

TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0}
TF_KEYS = {"15min": "15m", "1h": "1h", "4h": "4h"}
print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Model: {MODEL_FILE}")


def _process_one_tf(args):
    """Process a single asset/TF combo — runs in a worker process."""
    asset_name, prefix, asset_id, tf, tf_key, tf_hours = args
    data_file = f"data/{prefix}_{tf}.csv"
    if not os.path.exists(data_file):
        return None

    df = pd.read_csv(data_file).reset_index(drop=True)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    volumes = df["Volume"].values
    rsi = df["rsi"].values if "rsi" in df.columns else np.full(len(highs), 50.0)
    bb = df["bb"].values if "bb" in df.columns else np.zeros(len(highs))
    ema_21 = df["ema_21"].values if "ema_21" in df.columns else closes.copy()

    # Generate per-bar labels
    labels, _all_ranges, _active_per_bar = generate_per_bar_labels(
        highs, lows, closes, opens, tf_key=tf_key,
    )

    # Build per-bar features
    features = build_per_bar_features(
        highs, lows, closes, opens, volumes, rsi, bb, ema_21, tf_hours, asset_id,
    )

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_total < 100:
        return None

    # Time-based split
    split_idx = int(n_total * 0.8)

    return {
        "label": f"{asset_name}/{tf_key}",
        "n_bars": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "train_feat": features[:split_idx],
        "train_labels": labels[:split_idx],
        "test_feat": features[split_idx:],
        "test_labels": labels[split_idx:],
    }


def load_data_set():
    from multiprocessing import Pool

    train_feats, train_labels = [], []
    test_feats, test_labels = [], []

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
        pos_pct = result["n_pos"] / max(result["n_bars"], 1) * 100
        print(f"  {label}: {result['n_bars']} bars "
              f"({result['n_pos']} pos / {result['n_neg']} neg, {pos_pct:.1f}% in range)")

        train_feats.append(result["train_feat"])
        train_labels.append(result["train_labels"])
        test_feats.append(result["test_feat"])
        test_labels.append(result["test_labels"])

    if not train_feats:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    train_feat = np.concatenate(train_feats)
    train_label = np.concatenate(train_labels).astype(np.float32)
    test_feat = np.concatenate(test_feats)
    test_label = np.concatenate(test_labels).astype(np.float32)

    n_train_pos = int(np.sum(train_label == 1))
    n_test_pos = int(np.sum(test_label == 1))
    print(f"\nCombined — Train: {len(train_feat)} bars ({n_train_pos} pos) | "
          f"Test: {len(test_feat)} bars ({n_test_pos} pos)")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat).astype(np.float32)
    test_scaled = scaler.transform(test_feat).astype(np.float32)

    train_set = RangeDetectorDataset(train_scaled, train_label, window=WINDOW)
    test_set = RangeDetectorDataset(test_scaled, test_label, window=WINDOW)

    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)

    # Compute pos_weight from training labels
    n_pos = n_train_pos
    n_neg = len(train_label) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    return train_loader, test_loader, pos_weight


def evaluate(model, test_loader, criterion):
    """Evaluate range detector at multiple thresholds."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            label_t = label.to(device)

            logit = model(x)
            loss = criterion(logit, label_t)
            total_loss += loss.item()

            all_logits.append(logit.cpu())
            all_labels.append(label.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)

    base_rate = labels.mean().item() * 100
    n_total = len(labels)

    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_pos = probs > thresh
        n_pred_pos = pred_pos.sum().item()
        if n_pred_pos > 0:
            true_pos = (labels[pred_pos] == 1).sum().item()
            precision = true_pos / n_pred_pos * 100
            recall = true_pos / max((labels == 1).sum().item(), 1) * 100
        else:
            precision = recall = 0
        pred_neg = ~pred_pos
        n_pred_neg = pred_neg.sum().item()
        true_neg = (labels[pred_neg] == 0).sum().item() if n_pred_neg > 0 else 0
        results[thresh] = (n_pred_pos, precision, recall, n_pred_neg, true_neg)

    # Probability spread
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        print(
            f"    P(range) spread — "
            f"in-range: {pos_probs.mean().item():.3f} vs "
            f"out-range: {neg_probs.mean().item():.3f} | "
            f"base rate: {base_rate:.1f}%"
        )

    return total_loss / max(len(test_loader), 1), results


def train():
    train_loader, test_loader, pos_weight_val = load_data_set()

    model = RangeDetector(n_features=N_FEATURES, hidden=22).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RangeDetector: {n_params:,} parameters")

    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"pos_weight: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=100,
        steps_per_epoch=len(train_loader),
    )

    best_score = -float("inf")
    best_loss = float("inf")
    counter = 0
    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, label in train_loader:
            x = x.to(device)
            label_t = label.to(device)

            logit = model(x)
            loss = criterion(logit, label_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        test_loss, results = evaluate(model, test_loader, criterion)
        avg_train_loss = train_loss / max(len(train_loader), 1)

        r50 = results.get(0.5, (0, 0, 0, 0, 0))
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} / {test_loss:.4f} | "
            f"P>0.5: {r50[0]} bars, {r50[1]:.0f}% prec, {r50[2]:.0f}% recall"
        )

        if (epoch + 1) % 10 == 0:
            print("  --- Threshold analysis ---")
            for thresh, (n_pass, prec, rec, n_rej, n_rej_correct) in sorted(results.items()):
                print(
                    f"    P(range) > {thresh}: {n_pass} pass | "
                    f"Prec: {prec:.0f}% | Recall: {rec:.0f}% | "
                    f"Rejected: {n_rej} ({n_rej_correct} truly out)"
                )

        # Save best by combined precision * recall score
        score = 0.0
        for thresh, weight in [(0.4, 1.0), (0.5, 2.0), (0.6, 1.5)]:
            n_t, p, r, _, _ = results.get(thresh, (0, 0, 0, 0, 0))
            if n_t >= 20:
                score += weight * (p * r / 100)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"  -> Saved best model (score: {score:.1f})")

        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= 20:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation with best model
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, results = evaluate(model, test_loader, criterion)
    print(f"\n{'=' * 60}")
    print(f"Best model — Per-bar Range Detection")
    print(f"{'=' * 60}")
    for thresh, (n_pass, prec, rec, n_rej, n_rej_correct) in sorted(results.items()):
        print(
            f"  P(range) > {thresh}: {n_pass} pass | "
            f"Prec: {prec:.0f}% | Recall: {rec:.0f}% | "
            f"Rejected: {n_rej} ({n_rej_correct} truly out)"
        )


if __name__ == "__main__":
    train()
