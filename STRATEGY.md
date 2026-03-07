# Liq+Range+SFP Strategy

## Overview

The strategy detects **Swing Failure Patterns (SFPs)** that occur at **range boundaries** with optional **liquidation confluence**, then uses a trained model with two functions:
1. **Classification gate** — `P(profitable)`: should we take this trade?
2. **TP regression** — predicted TP1/TP2 distances via MFE quantile regression

**Signal pipeline:**
```
Swing Detection → SFP Detection → Range Boundary Filter → Liq Feature Enrichment → Model Scoring → TP Prediction
```

Only signals that pass all filters AND exceed the model's confidence threshold (`P(win) > 0.3`) are emitted, with model-predicted TP levels.

---

## 1. Swing Detection

**File:** `src/labels/sfp_labels.py:detect_swings()`

Detects swing highs and swing lows using an n-bar window on each side.

- **Swing High at bar `i`:** `High[i]` is the strict maximum of `High[i-n : i+n+1]` (must be unique)
- **Swing Low at bar `i`:** `Low[i]` is the strict minimum of `Low[i-n : i+n+1]` (must be unique)

Two swing sizes are used: `n=5` (fast) and `n=10` (slow), merged later.

### Active Swing Levels

**File:** `src/labels/sfp_labels.py:build_swing_level_series()`

For each bar, maintains a list of all confirmed swing levels within `max_age` bars. A swing at index `j` is only confirmed at `j+n` (no look-ahead bias). Levels older than `max_age=150` bars are dropped.

---

## 2. SFP Detection

**File:** `src/labels/sfp_labels.py:detect_sfp()`

An SFP occurs when price sweeps beyond an active swing level then **reclaims** back inside.

### Bullish SFP (Long signal)
1. Candle's Low sweeps below an active swing low: `Low[i] < swing_low`
2. Sweep distance is within 5%: `(swing_low - Low[i]) / swing_low <= 0.05`
3. Body position: `Open[i] > swing_low` (body is above the level — wick-only sweep)
4. Reclaim: `Close[k] > swing_low` within `reclaim_window` candles
5. Entry price = the swept swing low level

### Bearish SFP (Short signal)
1. Candle's High sweeps above an active swing high: `High[i] > swing_high`
2. Sweep distance is within 5%
3. Body position: `Open[i] < swing_high`
4. Reclaim: `Close[k] < swing_high` within `reclaim_window` candles
5. Entry price = the swept swing high level

### Merge Rule
- SFPs from `n=5` (reclaim_window=1) and `n=10` (reclaim_window=3) are merged
- Agreement: keep the signal
- One fires, other doesn't: keep the signal
- Conflict (long vs short on same bar): no trade

---

## 3. Range Detection (Zone-Based)

**File:** `src/labels/range_sfp_labels.py:detect_ranges_v2()`

Identifies consolidation ranges where price trades between support and resistance zones.

### Step 1: Collect Price Levels
For each temporal window, collect candidate levels:
- **Swing highs** → resistance candidates
- **Swing lows** → support candidates
- **Wick rejections**: candles where the upper wick > 30% of candle range → resistance; lower wick > 30% → support

### Step 2: Cluster Levels into Zones
**File:** `cluster_levels(prices_with_bars, pct=0.015)`

Groups nearby price levels into `PriceZone` objects:
- Sorted by price, iteratively merged if within `pct` (1.5%) of the running cluster mean
- Total cluster width capped at `2 * pct`
- Each zone has: `bottom`, `top`, `level` (mean), `count` (touches), `bars` (bar indices)

Only zones with enough touches are kept (`min_zone_count`: 4 for 15m, 3 for 1h/4h).

### Step 3: Form Ranges
Pair each support zone with each resistance zone:
- Resistance must be above support, zones must not overlap
- Range height must be within configured bounds (e.g., 0.3%–6% for 15m, 0.5%–12% for 1h/4h)
- **Time concentration check** (causal): from the earliest zone bar, count what fraction of the next `min_bars` candles close inside the range
  - Must meet threshold: 80% for 15m, 70% for 1h, 65% for 4h

### Step 4: Break Detection + Post-Break Ranges
When a range breaks (close outside boundary):
- Mark the range as broken (cannot generate more signals)
- Form a **post-break range**: the old boundary becomes one side, the extreme of the breakout move becomes the other
- Post-break ranges have relaxed concentration requirements (60% of normal threshold)

### Step 5: Active Ranges Per Bar
For each bar, list all ranges that are:
- Confirmed (past `start + min_bars`)
- Not yet broken
- Within `max_bars` of start

### Temporal Windows
To prevent stale levels from contaminating detection (e.g., $68K in 2021 vs 2024), ranges are detected in overlapping temporal windows (`window_size = max(max_bars*2, 500)`, `step = max(max_bars, 250)`), then deduped across windows.

### Per-TF Parameters

| Parameter | 15m | 1h | 4h |
|---|---|---|---|
| `n_swing` | 3 | 3 | 3 |
| `min_bars` | 30 | 15 | 20 |
| `max_bars` | 400 | 300 | 300 |
| `min_zone_count` | 4 | 3 | 3 |
| `min_time_concentration` | 80% | 70% | 65% |
| Range height (% of price) | 0.3%–6% | 0.5%–12% | 0.5%–12% |

---

## 4. Boundary Filter

**File:** `src/labels/liq_sfp_labels.py:generate_labels()` (lines 140–195)

Only SFPs that occur **at a range boundary** are kept. This is the primary quality filter.

### Long Signal Requirements
1. Swept level is within the **support zone** (±buffer)
2. Candle low penetrates below `support.level` (actually sweeps into the zone)
3. Candle close reclaims above the zone (strictness depends on TF)

### Short Signal Requirements
1. Swept level is within the **resistance zone** (±buffer)
2. Candle high penetrates above `resistance.level`
3. Candle close reclaims below the zone

### Per-TF Strictness

| TF | Zone Buffer | Reclaim Requirement |
|---|---|---|
| 15m | `0` (exact match) | Close must beat `zone.top` / `zone.bottom` |
| 1h | `ATR * 0.1` | Close >= `zone.level` is enough |
| 4h | `ATR * 0.15` | Close >= `zone.level` is enough |

### Best Range Selection
If multiple active ranges qualify, the one with the highest **quality score** is selected:
```
score = concentration × min(touches_high, touches_low)
```

---

## 5. Liquidation Features (Soft)

**File:** `src/labels/liq_labels.py` + `src/labels/liq_sfp_labels.py`

Liquidation data is computed for each signal bar but is **NOT used as a filter** — it's provided as features to the model, which learns that liq confluence correlates with higher `P(win)`.

### Liquidation Price Estimation
No exchange API needed — estimated from swing levels (assumed entries) and common leverage tiers:

```python
LEVERAGE_TIERS = [(3, 0.05), (5, 0.15), (10, 0.30), (25, 0.25), (50, 0.15), (100, 0.10)]

# Long liquidation (below entry):
liq_price = entry × (1 - 1/leverage + maintenance_margin_rate)

# Short liquidation (above entry):
liq_price = entry × (1 + 1/leverage - maintenance_margin_rate)
```

Maintenance margin rate: 0.4% (Binance Tier 1).

### Per-Signal Liq Features (6 total)
Computed lazily, only for bars that pass the boundary filter:

| Feature | Description |
|---|---|
| `n_liq_swept` | Count of liquidation levels triggered by the SFP candle's wick |
| `weighted_liq_swept` | Sum of popularity weights of triggered levels |
| `max_leverage_swept` | Highest leverage tier that was triggered |
| `liq_cascade_depth` | Price range spanned by triggered levels (max - min) |
| `liq_cluster_density` | Triggered / total levels on that side |
| `n_swings_with_liq` | Unique swing entries whose liq levels were triggered |

**Proximity threshold**: A liq level counts as "swept" if it's within `0.5 × ATR` of the candle's wick extreme.

---

## 6. Feature Engineering (37 Features)

**File:** `src/train_liq_range_sfp.py:build_features()` / `server/pipeline.py:build_liq_range_sfp_features()`

All features are computed per bar after dropping 30 warmup bars (for ATR-14, MA-20, etc.).

### Range Features (6)
| # | Feature | Normalization |
|---|---|---|
| 1 | `range_height_pct` | range_height / mid_price |
| 2 | `range_touches_norm` | min(touches, 5) / 5 |
| 3 | `range_concentration` | bars_inside / total_bars |
| 4 | `range_age` | (bar - confirmed) / 200, clipped [0, 5] |
| 5 | `sweep_depth_range` | wick beyond boundary / range_height, clipped [0, 2] |
| 6 | `reclaim_strength_range` | (close - boundary) / range_height, clipped [0, 2] |

### Liquidation Features (6)
| # | Feature | Normalization |
|---|---|---|
| 7 | `n_liq_swept_norm` | min(n, 30) / 30 |
| 8 | `weighted_liq_swept` | min(w, 3.0) / 3.0 |
| 9 | `max_leverage_norm` | leverage / 100 |
| 10 | `liq_cascade_depth` | cascade_depth / ATR, clipped [0, 5] |
| 11 | `liq_cluster_density` | raw [0, 1] |
| 12 | `n_swings_with_liq` | min(n, 10) / 10 |

### SFP Candle Features (6)
| # | Feature | Description |
|---|---|---|
| 13 | `body_ratio` | (close - open) / candle_range |
| 14 | `wick_ratio` | rejection wick / candle_range |
| 15 | `vol_spike` | volume / 20-bar MA volume, clipped [0, 5] |
| 16 | `close_position` | where close sits within candle range |
| 17 | `zone_sl_dist` | entry to zone boundary (SL side), clipped [0, 10%] |
| 18 | `zone_tp_dist` | entry to opposite boundary (TP side), clipped [0, 15%] |

### Context Features (6)
| # | Feature | Description |
|---|---|---|
| 19 | `rsi` | RSI / 100 |
| 20 | `trend_strength` | (close - EMA21) / close, clipped [-0.5, 0.5] |
| 21 | `ms_alignment` | +1 if signal aligns with market structure, -1 if against, 0 unclear |
| 22 | `ms_strength` | consecutive HH/HL or LH/LL count / 5 |
| 23 | `tf_hours` | timeframe in hours / 4 (0.0625 for 15m, 0.25 for 1h, 1.0 for 4h) |
| 24 | `asset_id` | numeric asset identifier |

### Range Fingerprint Features (6)
| # | Feature | Description |
|---|---|---|
| 25 | `signal_type` | 0=SFP boundary, 1=range approach |
| 26 | `is_recaptured` | 1.0 if range was broken then recaptured |
| 27 | `is_nested` | 1.0 if range is inside a macro range |
| 28 | `touch_symmetry` | min(touches_high, touches_low) / max(touches_high, touches_low) |
| 29 | `boundary_rejection_avg` | avg wick rejection at tested boundary / range_height, clipped [0, 2] |
| 30 | `range_position` | (close - support) / range_height, clipped [0, 1] |

### New Features (3)
| # | Feature | Description |
|---|---|---|
| 31 | `direction_feat` | 1.0 for long, -1.0 for short, 0 for non-signal bars |
| 32 | `vwap_distance` | (close - VWAP_20) / close, clipped [-0.05, 0.05] |
| 33 | `volume_imbalance` | up_vol_10 / total_vol_10 - 0.5, clipped [-0.5, 0.5] |

### HTF Alignment Features (4)
Computed by resampling current TF data 4:1 to simulate higher-TF context (15m→1h, 1h→4h, 4h→16h).

| # | Feature | Description |
|---|---|---|
| 34 | `htf_trend` | (close - EMA21) / close on resampled data, clipped [-0.5, 0.5] |
| 35 | `htf_rsi` | RSI-14 on resampled data, normalized [0, 1] |
| 36 | `htf_ms_direction` | market structure direction on resampled (-1, 0, +1) |
| 37 | `htf_ms_strength` | market structure strength on resampled |

---

## 7. Model Architecture

**File:** `src/models/liq_range_sfp_model.py`

`LiqRangeSFPClassifier` — 5-head model with direction-conditioned FiLM, positional encoding, and 2-layer transformer (~80K parameters).

```
Input: (batch, window, 37)
  |
Linear(37 -> 48) + ReLU            # per-bar projection
  |
+ pos_embed (learnable, 1×window×48)  # positional encoding
  |
MultiheadAttention(48, heads=1)    # self-attention layer 1
  + residual + LayerNorm
  |
FFN: Linear(48->96) -> ReLU -> Linear(96->48)
  + residual + LayerNorm
  |
MultiheadAttention(48, heads=2)    # self-attention layer 2
  + residual + LayerNorm
  |
FFN: Linear(48->96) -> ReLU -> Linear(96->48)
  + residual + LayerNorm
  |
Three pooling strategies:
  - mean(dim=1)  -> global context   (48)
  - last bar     -> signal bar       (48)
  - max(dim=1)   -> peak activation  (48)
  |
Concat -> (144)
  |
  +---> cls_head:      Linear(144->48) -> ReLU -> Dropout(0.15) -> Linear(48->1)  ->  logit     -> sigmoid -> P(profitable)
  |
  +---> tp1_head:      Linear(144->48) -> ReLU -> Dropout(0.15) -> Linear(48->1)  ->  softplus  -> TP1 distance (fixed τ=0.15)
  |
  +---> tp2_head:      Linear(144->48) -> ReLU -> Dropout(0.15) -> Linear(48->1)  ->  softplus  -> TP2 distance (fixed τ=0.40)
  |
  +---> ttp_head:      Linear(144->48) -> ReLU -> Dropout(0.15) -> Linear(48->1)  ->  sigmoid  -> time-to-peak ∈ [0, 1]
  |
  +---> sl_pred_head:  Linear(144->48) -> ReLU -> Dropout(0.15) -> Linear(48->1)  ->  softplus  -> SL distance (positive)
  |
  +--- FiLM conditioning on all 5 heads (scale ∈ [0.7, 1.3], bias ∈ [-0.1, 0.1])
```

### FiLM Conditioning (Direction-Aware)
**File:** `src/models/liq_range_sfp_model.py:ConditioningModule`

Applies per-asset/TF/direction scale and bias to all 5 heads:
- Asset embedding (dim=8) + TF embedding (dim=4) + Direction embedding (dim=4) → Linear(16→16→8) → constrained scale + bias
- Direction embedding: 0=none, 1=long, 2=short — allows asymmetric long/short TP targets
- Scale ∈ [0.7, 1.3] via `0.7 + 0.6 * sigmoid(x)`, bias ∈ [-0.1, 0.1] via `0.1 * tanh(x)`

### Positional Encoding
Learnable positional embeddings added after bar projection:
- `pos_embed = nn.Parameter(torch.randn(1, window, 48) * 0.02)`
- Allows the model to distinguish temporal position of patterns within the window

### Time-to-Peak Head
Predicts when the MFE peak occurs within the forward horizon:
- Label: `peak_bar / horizon` ∈ [0, 1]
- Trained with smooth L1 loss (weight=0.5), only on profitable signals
- Used by server for TTP-based trailing stop (see Section 10)

### SL Prediction Head (v5)
Predicts optimal stop-loss distance from MAE (Max Adverse Excursion) data:
- Trained with quantile loss at τ=0.85 on MAE target (all signals, not just profitable)
- Output: positive distance via softplus
- Server uses model SL floored by candle extreme (never tighter than structural SL)

**Output (B, 5):**
- `[0]` P(profitable) logit — classification gate (apply sigmoid externally)
- `[1]` TP1 distance — conservative (softplus, fixed quantile τ=0.15)
- `[2]` TP2 distance — aggressive (softplus, fixed quantile τ=0.40)
- `[3]` ttp — time-to-peak ∈ [0, 1] (sigmoid)
- `[4]` SL distance — predicted optimal stop-loss distance (softplus)

**Window sizes** (with learnable positional encoding):

| TF | Window | Lookback |
|---|---|---|
| 15m | 120 bars | 30 hours |
| 1h | 48 bars | 2 days |
| 4h | 30 bars | 5 days |

---

## 8. MFE Labels (Training)

**File:** `src/labels/range_sfp_labels.py:compute_range_tp_sl_labels()`

Evaluated over a per-TF forward horizon:
- **15m**: 36 bars (9 hours) — longer horizon gives 15m moves more room to develop
- **1h**: 18 bars (18 hours)
- **4h**: 18 bars (3 days)

### MFE (Max Favorable Excursion)
The maximum favorable price move before SL is hit or horizon expires:

**Long:** `MFE = max((High[j] - entry) / entry)` for `j` in `[i+1, i+horizon]`, stopping if `Low[j] <= stop_price`

**Short:** `MFE = max((entry - Low[j]) / entry)` for `j` in `[i+1, i+horizon]`, stopping if `High[j] >= stop_price`

### SL (Stop Loss)
Structural distance beyond the tested boundary with 0.2% buffer:
- **Long:** `SL = (entry - support.bottom * 0.998) / entry`
- **Short:** `SL = (resistance.top * 1.002 - entry) / entry`
- Clipped to [0.1%, 8%]

### Quality Label
`quality = 1` if `MFE > SL distance` (the trade moved favorably more than the risk), else `0`.

MFE clipped to [0%, 15%].

### Time-to-Peak Label
`ttp = peak_bar / horizon` where `peak_bar` is the bar index at which MFE was achieved. Gives 0.0 for early peaks, 1.0 for peaks at horizon end. Only meaningful for profitable signals.

### MAE (Max Adverse Excursion) (v5)
The maximum adverse price move before SL is hit or horizon expires:

**Long:** `MAE = max((entry - Low[j]) / entry)` for `j` in `[i+1, i+horizon]`, stopping if `Low[j] <= stop_price`

**Short:** `MAE = max((High[j] - entry) / entry)` for `j` in `[i+1, i+horizon]`, stopping if `High[j] >= stop_price`

MAE clipped to [0%, 15%]. Used to train the SL regression head.

### Returns
`(quality, mfe_labels, sl_labels, ttp_labels, mae_labels)` — 5 arrays

---

## 9. Training

**File:** `src/train_liq_range_sfp.py`

### Data
- **Assets**: BTC, ETH, SOL, Gold
- **Timeframes**: 15m, 1h, 4h (from 2018+)
- **Split**: 80% train / 20% test (chronological)
- **Scaler**: `StandardScaler` fit on all training data combined

### Loss Function
Combined classification + regression + time-to-peak + SL:

```
loss = cls_loss + reg_loss + 0.5 * ttp_loss + 0.2 * sl_loss
```

**Classification (gate):**
- `BCEWithLogitsLoss` on quality target with `pos_weight = n_neg / n_pos`

**Regression (TP prediction):**
- Fixed quantile (pinball) loss on MFE target
- TP1: fixed τ=0.15 — conservative target, higher win rate
- TP2: fixed τ=0.40 — aggressive target, larger moves
- **Only computed for profitable signals** (quality=1) — no regression signal from losing trades

**Time-to-Peak:**
- Smooth L1 loss on ttp target (peak_bar / horizon)
- Weight: 0.5
- **Only computed for profitable signals** (quality=1)

**SL Regression (v5):**
- Quantile loss at τ=0.85 on MAE target (max adverse excursion)
- Weight: 0.2 (low weight to avoid interfering with TP heads)
- **Computed on ALL signals** (not just profitable) — learns adverse excursion for all setups

### Optimizer
- AdamW (lr=1e-3, weight_decay=1e-3)
- OneCycleLR scheduler
- Gradient clipping: max_norm=1.0

### Model Selection
- Weighted EV score across thresholds 0.4/0.5/0.6
- EV = (TP1_WR%) * avg_TP1 - (1 - TP1_WR%) * avg_SL
- Early stopping: patience=50 epochs on test loss

---

## 10. Inference (Server)

**File:** `server/app.py`, `server/inference.py`

Per scheduled job (each asset x TF):
1. Fetch 1000 candles from Binance, merge with bar cache (up to 5000 bars)
2. Recompute indicators (OBV, BB, EMA21, RSI)
3. Run `run_liq_range_sfp_detection()` -> signals with boundary filter
4. Build features (37) -> scale with saved `StandardScaler`
5. For each signal bar: `predict_bar(model, scaled, bar_idx, tf_key, asset_id)` -> `(P(win), tp1_dist, tp2_dist, ttp, sl_dist)`
6. If `P(win) >= 0.3`: emit signal with entry, model-predicted TP/SL levels, and ttp

### Ensemble Inference (v5)
If multiple ensemble model files exist (`best_model_liq_range_sfp_*.pth`), the server loads all of them:
- `P(win)` = mean of sigmoid outputs (not sigmoid of mean logit)
- `TP1/TP2/TTP/SL` = mean of regression outputs
- Falls back to single model if no ensemble files found

### Signal Structure
- **Entry**: swept swing level
- **TP1**: `entry * (1 ± tp1_dist)` — conservative target from model regression
- **TP2**: `entry * (1 ± tp2_dist)` — aggressive target from model regression
- **SL**: model-predicted SL distance, floored by candle extreme (never tighter than structural)
  - Long: `sl_price = min(candle_low, entry * (1 - sl_dist))`
  - Short: `sl_price = max(candle_high, entry * (1 + sl_dist))`
- **Best TP**: TP2 if its R:R exceeds TP1's by 1.5x, else TP1
- **TTP**: predicted time-to-peak ∈ [0, 1] (fraction of forward horizon)
- **Expiry**: 3 bars
- **Resolution horizon**: per-TF (36 bars for 15m, 18 bars for 1h/4h)

### Partial TP Execution with TTP Trailing Stop
When TP1 is hit before TP2:
1. Signal status changes from `open` → `partial`
2. **TTP-based trailing stop** engages (replaces fixed breakeven):
   - Model's TTP prediction determines trail tightness:
     - Early peak (ttp ≤ 0.3): 0.3% trail — tight, lock in gains quickly
     - Mid peak (ttp ≤ 0.6): 0.6% trail — moderate room
     - Late peak (ttp > 0.6): 1.0% trail — wide, let it run
   - `best_price` tracks the high-water mark (persisted across job runs)
   - Trail SL = `best_price × (1 - trail_pct)` for longs, `× (1 + trail_pct)` for shorts
3. If TP2 is hit → `win` (reward = 0.5 × TP1_R + 0.5 × TP2_R)
4. If trailing stop hit → `partial_win` (reward = 0.5 × TP1_R + 0.5 × max(trailing_R, 0))
5. If horizon expires → `partial_win` (close at market)
6. `partial_win` counts as a win in stats

**Config** (`server/config.py:TTP_TRAILING`):
```python
TTP_TRAILING = {
    "early": {"max_ttp": 0.3, "trail_pct": 0.003},   # 0.3% trail
    "mid":   {"max_ttp": 0.6, "trail_pct": 0.006},    # 0.6% trail
    "late":  {"trail_pct": 0.010},                      # 1.0% trail
}
```

---

## Benchmark Performance (2024 OOS, P>0.7)

### v4 (current) — 37 features, hidden=48, per-TF horizon, ~74K params

| Asset/TF | Trades | Gate WR | TP1 WR | TP1 EV% | Total R |
|----------|--------|---------|--------|---------|---------|
| SOL/4h | 928 | 83% | 62% | +1.69% | +1503R |
| ETH/4h | 887 | 82% | 66% | +1.39% | +997R |
| SOL/15m | 2,083 | 84% | 74% | +1.06% | +1726R |
| SOL/1h | 2,879 | 78% | 62% | +1.00% | +2184R |
| BTC/4h | 677 | 76% | 56% | +0.88% | +455R |
| ETH/1h | 1,844 | 76% | 64% | +0.86% | +1175R |
| ETH/15m | 1,114 | 79% | 67% | +0.78% | +812R |
| BTC/15m | 552 | 78% | 63% | +0.55% | +362R |
| BTC/1h | 1,250 | 65% | 55% | +0.41% | +494R |
| GOLD/15m | 63 | 78% | 83% | +0.32% | +49R |
| GOLD/1h | 116 | 52% | 43% | -0.13% | +21R |
| GOLD/4h | 199 | 60% | 41% | -0.10% | +3R |

10/12 asset/TF combos positive EV. Aggregate at P>0.7:
```
P > 0.3: 24772 trades | TP1 EV=+0.361%
P > 0.5: 18360 trades | TP1 EV=+0.624%
P > 0.7: 12597 trades | TP1 EV=+0.905%
```

### v3 — 33 features, hidden=32, fixed 18-bar horizon, ~35K params

| Asset/TF | Trades | Gate WR | TP1 WR | TP1 EV% | Total R |
|----------|--------|---------|--------|---------|---------|
| SOL/4h | 725 | 82% | 73% | +1.58% | +1158R |
| ETH/4h | 695 | 79% | 71% | +1.20% | +782R |
| SOL/1h | 2,337 | 80% | 63% | +1.03% | +1921R |
| BTC/4h | 409 | 79% | 76% | +0.97% | +338R |
| SOL/15m | 1,233 | 83% | 73% | +0.86% | +940R |
| ETH/1h | 1,504 | 74% | 59% | +0.70% | +924R |
| ETH/15m | 530 | 76% | 61% | +0.51% | +330R |
| GOLD/4h | 59 | 70% | 93% | +0.53% | +24R |
| BTC/1h | 972 | 66% | 56% | +0.37% | +419R |
| GOLD/1h | 72 | 68% | 99% | +0.26% | +5R |
| GOLD/15m | 46 | 63% | 70% | +0.25% | +26R |
| BTC/15m | 233 | 68% | 52% | +0.20% | +106R |

12/12 positive EV. Aggregate at P>0.7:
```
P > 0.3: 12612 trades | TP1 EV=+0.253%
P > 0.5:  9287 trades | TP1 EV=+0.465%
P > 0.7:  5311 trades | TP1 EV=+0.758%
```

### Key Changes (v3 → v4)
- **15m massively improved** (main goal): BTC +0.20→+0.55%, ETH +0.51→+0.78%, SOL +0.86→+1.06%, all with 2x trades
- **More trades everywhere**: 12,597 vs 5,311 at P>0.7 (2.4x)
- **Aggregate TP1 EV up**: +0.758% → +0.905% at P>0.7
- **GOLD 1h/4h regressed** to negative EV (v3 had 72/59 trades with 99%/93% WR — likely overfit)

---

## Implemented Improvements

### v5 (current)
4 improvements targeting SL prediction, validation robustness, and inference reliability:

1. ~~**SL regression head**~~ — **Done.** 5th model output predicts optimal SL distance from MAE data. Trained with quantile loss at τ=0.85 on all signals. Server uses model SL floored by candle extreme.
2. ~~**Feature importance audit**~~ — **Done.** `src/feature_importance.py` performs permutation importance: shuffles each of 37 features, measures TP1 EV degradation. Identifies dead-weight features for removal.
3. ~~**Walk-forward validation**~~ — **Done.** `--walk-forward` flag: train 2018-2021→test 2022, train 2018-2022→test 2023, train 2018-2023→test 2024. Reports per-fold and averaged EV.
4. ~~**Ensemble training**~~ — **Done.** `--ensemble N` flag: trains N models with different seeds, saves as `best_model_liq_range_sfp_0.pth`, etc. Server auto-detects ensemble files and averages predictions (P(win) = mean of sigmoids, TP/SL = mean of regressions).

### v4
3 improvements targeting model capacity and 15m performance:

1. ~~**Per-TF horizon**~~ — **Done.** 15m horizon increased 18→36 bars (9 hours) to give moves more room to develop. 1h/4h keep 18 bars.
2. ~~**Hidden dim 48**~~ — **Done.** Hidden dimension increased 32→48 across all layers (~73K params, up from ~35K). More capacity for 37-feature input.
3. ~~**Multi-TF alignment features**~~ — **Done.** 4 new HTF features (33→37 total) computed by resampling current TF data 4:1: htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength.

### v3
4 architecture changes + server-side trailing stop:

1. ~~**Remove tau heads**~~ — **Done.** Adaptive tau heads always pinned at floor values (wasted 6,274 params). Replaced with fixed τ=0.15/0.40 in loss function.
2. ~~**Positional encoding**~~ — **Done.** Learnable `pos_embed` (1×window×32) after bar projection. Model can now distinguish temporal position of patterns.
3. ~~**2nd transformer layer**~~ — **Done.** Added `attn2` (2 heads) + `ffn2` + norms. Deeper feature extraction before pooling.
4. ~~**Direction-conditioned FiLM**~~ — **Done.** Added `direction_emb` (Embedding(3, 4)) to ConditioningModule. FiLM input expanded from 12→16. Enables asymmetric long/short TP targets.
5. ~~**TTP trailing stop**~~ — **Done.** Server uses model TTP prediction to set trailing stop tightness after TP1 hit (early peak → tight trail, late peak → wide trail).

### v2
1. ~~**Adaptive tau heads**~~ — tau1/tau2 predict per-sample quantile (removed in v3 — always pinned at minimums).
2. ~~**FiLM conditioning**~~ — asset+TF embeddings → scale+bias on cls/tp1/tp2 heads.
3. ~~**Partial TP execution**~~ — TP1→partial→TP2/breakeven flow (upgraded to trailing stop in v3).
4. ~~**Direction-aware features**~~ — `direction_feat` (+1 long, -1 short).
5. ~~**Volume profile features**~~ — `vwap_distance` and `volume_imbalance`.
6. ~~**Time-to-peak head**~~ — predicts when MFE peak occurs (now used for trailing stop in v3).

## Potential Next Improvements

1. **Attention-based feature selection** — replace fixed 37-feature set with learnable feature attention weights. Let the model learn which features matter per-signal rather than treating all equally.

2. **Multi-task curriculum learning** — start training with only classification loss, gradually introduce regression and SL losses. May help the model learn a better representation before splitting into task-specific heads.

3. **Asymmetric SL by direction** — currently SL prediction treats long/short symmetrically. Long drawdowns and short squeezes have different dynamics. Consider direction-specific MAE targets or separate SL tau values.

4. **Online adaptation** — periodically fine-tune the model on recent live signals to adapt to changing market regimes. Risk: overfitting to recent noise. Mitigation: use walk-forward validation to set the fine-tuning window size.
