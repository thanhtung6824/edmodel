"""Regression-only SFP training: predict TP/SL, use ratio as quality filter."""
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.labels.sfp_labels import detect_swings, build_swing_level_series, generate_labels
from src.models.dataset import SFPDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# --- Regression-only model ---
class SFPRegModel(nn.Module):
    def __init__(self, n_features=14, hidden_size=64, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attn_weight = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.3)

        # TP head: positive percentage
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # SL head: positive percentage
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        out, _ = self.lstm(x)
        out = self.layer_norm(out)

        attn_scores = self.attn_weight(out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (out * attn_weights).sum(dim=1)
        context = self.dropout(context)

        tp = self.tp_head(context).squeeze(-1)
        sl = self.sl_head(context).squeeze(-1)
        return tp, sl


# --- Loss: regression only ---
class RegLoss(nn.Module):
    def forward(self, tp_pred, sl_pred, tp_target, sl_target):
        tp_loss = F.smooth_l1_loss(tp_pred, tp_target)
        sl_loss = F.smooth_l1_loss(sl_pred, sl_target)
        return tp_loss + sl_loss, tp_loss, sl_loss


def load_data_set():
    df = pd.read_csv("data/btc_4h.csv")
    df = df.reset_index(drop=True)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values

    actions, quality, tp_labels, sl_labels = generate_labels(highs, lows, closes, opens)

    swing_levels = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl = build_swing_level_series(highs, lows, sh, sl, n, max_age=150)
        nearest_sh = np.array([levels[0] if levels else np.nan for levels in active_sh])
        nearest_sl = np.array([levels[0] if levels else np.nan for levels in active_sl])
        swing_levels[n] = (nearest_sh, nearest_sl)

    # --- Feature engineering (14 features) ---
    prev_close = df["Close"].shift(1)
    feat = pd.DataFrame()

    feat["Open"] = df["Open"] / prev_close - 1
    feat["High"] = df["High"] / prev_close - 1
    feat["Low"] = df["Low"] / prev_close - 1
    feat["Close"] = df["Close"] / prev_close - 1
    feat["rsi"] = df["rsi"] / 100.0
    feat["volatility"] = df["Close"].pct_change().rolling(30).std()

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

    # Drop warmup rows
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    quality = quality[drop_n:]
    tp_labels = tp_labels[drop_n:]
    sl_labels = sl_labels[drop_n:]

    feat = feat.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    feat["vol_rel_20"] = feat["vol_rel_20"].clip(0, 5.0)
    for n in [5, 10]:
        feat[f"sweep_below_{n}"] = feat[f"sweep_below_{n}"].clip(0, 0.05)
        feat[f"sweep_above_{n}"] = feat[f"sweep_above_{n}"].clip(0, 0.05)

    feat_values = feat.values.astype(np.float32)
    n_features = feat_values.shape[1]
    print(f"Features ({n_features}): {list(feat.columns)}")

    # --- Stats ---
    sfp_mask = actions != 0
    total_sfp = int(np.sum(sfp_mask))
    n_profitable = int(np.sum(quality[sfp_mask] == 1))
    n_losing = total_sfp - n_profitable
    print(f"\nSFP signals: {total_sfp} | Profitable: {n_profitable} ({n_profitable/total_sfp*100:.0f}%) | Losing: {n_losing}")

    # --- Train/test split ---
    split_idx = int(len(feat_values) * 0.8)
    train_feat = feat_values[:split_idx]
    test_feat = feat_values[split_idx:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)

    window = 30
    train_set = SFPDataset(train_scaled, actions[:split_idx], quality[:split_idx],
                           tp_labels[:split_idx], sl_labels[:split_idx], window=window)
    test_set = SFPDataset(test_scaled, actions[split_idx:], quality[split_idx:],
                          tp_labels[split_idx:], sl_labels[split_idx:], window=window)

    print(f"Train SFPs: {len(train_set)}, Test SFPs: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

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

    tp_mae = (tp_preds - tp_targets).abs().mean().item() * 100
    sl_mae = (sl_preds - sl_targets).abs().mean().item() * 100

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
            avg_tp = tp_targets[take].mean().item() * 100
            avg_sl = sl_targets[take].mean().item() * 100
        else:
            precision = recall = avg_tp = avg_sl = 0
        results[thresh] = (n_take, precision, recall, avg_tp, avg_sl)

    return total_loss / len(test_loader), tp_mae, sl_mae, results


def train():
    train_loader, test_loader, n_features = load_data_set()
    model = SFPRegModel(n_features=n_features).to(device)
    criterion = RegLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_loss = float('inf')
    counter = 0
    epochs = 100

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for x, direction, q, tp, sl in train_loader:
            x = x.to(device)
            tp_t = tp.to(device)
            sl_t = sl.to(device)

            tp_pred, sl_pred = model(x)
            loss, _, _ = criterion(tp_pred, sl_pred, tp_t, sl_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Evaluate ---
        test_loss, tp_mae, sl_mae, ratio_results = evaluate(model, test_loader, criterion)
        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        # Main log line
        r = ratio_results.get(1.5, (0, 0, 0, 0, 0))
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"TP MAE: {tp_mae:.2f}% | "
            f"SL MAE: {sl_mae:.2f}% | "
            f"Ratio>1.5: {r[0]} trades, {r[1]:.0f}% prec, {r[2]:.0f}% recall"
        )

        # Every 10 epochs, print full ratio analysis
        if (epoch + 1) % 10 == 0:
            print("  --- Ratio threshold analysis ---")
            for thresh, (n, prec, rec, avg_tp, avg_sl) in sorted(ratio_results.items()):
                print(f"    TP/SL > {thresh}: {n} trades | "
                      f"Prec: {prec:.0f}% | Recall: {rec:.0f}% | "
                      f"Avg TP: {avg_tp:.2f}% | Avg SL: {avg_sl:.2f}%")

        # --- Early stopping on test loss ---
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), "best_model_reg.pth")
            print(f"  -> Saved best model (loss: {best_loss:.4f})")
        else:
            counter += 1
            if counter >= 15:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # --- Final evaluation with best model ---
    model.load_state_dict(torch.load("best_model_reg.pth", weights_only=True))
    _, tp_mae, sl_mae, ratio_results = evaluate(model, test_loader, criterion)
    print(f"\n{'='*60}")
    print(f"Best model — TP MAE: {tp_mae:.2f}%, SL MAE: {sl_mae:.2f}%")
    print(f"{'='*60}")
    for thresh, (n, prec, rec, avg_tp, avg_sl) in sorted(ratio_results.items()):
        print(f"  TP/SL > {thresh}: {n} trades | "
              f"Prec: {prec:.0f}% | Recall: {rec:.0f}% | "
              f"Avg TP: {avg_tp:.2f}% | Avg SL: {avg_sl:.2f}%")


train()
