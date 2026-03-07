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

## 6. Feature Engineering (33 Features)

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

---

## 7. Model Architecture

**File:** `src/models/liq_range_sfp_model.py`

`LiqRangeSFPClassifier` — 6-head model with FiLM conditioning (~29K parameters).

```
Input: (batch, window, 33)
  |
Linear(33 -> 32) + ReLU            # per-bar projection
  |
MultiheadAttention(32, heads=1)    # self-attention across bars
  + residual + LayerNorm
  |
FFN: Linear(32->64) -> ReLU -> Linear(64->32)
  + residual + LayerNorm
  |
Three pooling strategies:
  - mean(dim=1)  -> global context   (32)
  - last bar     -> signal bar       (32)
  - max(dim=1)   -> peak activation  (32)
  |
Concat -> (96)
  |
  +---> cls_head:  Linear(96->32) -> ReLU -> Dropout(0.15) -> Linear(32->1)  ->  logit     -> sigmoid -> P(profitable)
  |
  +---> tp1_head:  Linear(96->32) -> ReLU -> Dropout(0.15) -> Linear(32->1)  ->  softplus  -> TP1 distance (adaptive quantile)
  |
  +---> tp2_head:  Linear(96->32) -> ReLU -> Dropout(0.15) -> Linear(32->1)  ->  softplus  -> TP2 distance (adaptive quantile)
  |
  +--- FiLM conditioning on cls/tp1/tp2 (scale ∈ [0.7, 1.3], bias ∈ [-0.1, 0.1])
  |
  +---> tau1_head: Linear(96->32) -> ReLU -> Linear(32->1) -> sigmoid * 0.20 + 0.15  ->  tau1 ∈ [0.15, 0.35]
  |
  +---> tau2_head: Linear(96->32) -> ReLU -> Linear(32->1) -> sigmoid * 0.20 + 0.40  ->  tau2 ∈ [0.40, 0.60]
  |
  +---> ttp_head:  Linear(96->32) -> ReLU -> Dropout(0.15) -> Linear(32->1)  ->  sigmoid  -> time-to-peak ∈ [0, 1]
```

### FiLM Conditioning
**File:** `src/models/liq_range_sfp_model.py:ConditioningModule`

Applies per-asset/TF scale and bias to cls, tp1, tp2 heads (NOT tau or ttp):
- Asset embedding (dim=8) + TF embedding (dim=4) → Linear(12→16→6) → constrained scale + bias
- Scale ∈ [0.7, 1.3] via `0.7 + 0.6 * sigmoid(x)`, bias ∈ [-0.1, 0.1] via `0.1 * tanh(x)`
- Tau and ttp heads are NOT FiLM-conditioned — they must stay within their sigmoid-constrained ranges

### Adaptive Tau Heads
Two heads predict per-sample quantile levels for the quantile regression loss:
- tau1 ∈ [0.15, 0.35] — controls how conservative TP1 is
- tau2 ∈ [0.40, 0.60] — controls how aggressive TP2 is
- Gradient flows from quantile loss through tau — self-supervised
- In practice, both tau heads pin to their minimums (tau1≈0.15, tau2≈0.40), which produces better TP WR

### Time-to-Peak Head
Predicts when the MFE peak occurs within the forward horizon:
- Label: `peak_bar / horizon` ∈ [0, 1]
- Trained with smooth L1 loss (weight=0.5), only on profitable signals
- Used by server for time-aware position management

**Output (B, 6):**
- `[0]` P(profitable) logit — classification gate (apply sigmoid externally)
- `[1]` TP1 distance — conservative (softplus, adaptive quantile)
- `[2]` TP2 distance — aggressive (softplus, adaptive quantile)
- `[3]` tau1 — adaptive quantile for TP1 ∈ [0.15, 0.35]
- `[4]` tau2 — adaptive quantile for TP2 ∈ [0.40, 0.60]
- `[5]` ttp — time-to-peak ∈ [0, 1] (sigmoid)

**Window sizes** (no positional encoding — fully agnostic):

| TF | Window | Lookback |
|---|---|---|
| 15m | 120 bars | 30 hours |
| 1h | 48 bars | 2 days |
| 4h | 30 bars | 5 days |

---

## 8. MFE Labels (Training)

**File:** `src/labels/range_sfp_labels.py:compute_range_tp_sl_labels()`

Evaluated over an 18-bar forward horizon.

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

### Returns
`(quality, mfe_labels, sl_labels, ttp_labels)` — 4 arrays

---

## 9. Training

**File:** `src/train_liq_range_sfp.py`

### Data
- **Assets**: BTC, ETH, SOL, Gold
- **Timeframes**: 15m, 1h, 4h (from 2018+)
- **Split**: 80% train / 20% test (chronological)
- **Scaler**: `StandardScaler` fit on all training data combined

### Loss Function
Combined classification + regression + time-to-peak:

```
loss = cls_loss + reg_loss + 0.5 * ttp_loss
```

**Classification (gate):**
- `BCEWithLogitsLoss` on quality target with `pos_weight = n_neg / n_pos`

**Regression (TP prediction):**
- Adaptive quantile (pinball) loss on MFE target
- TP1: τ predicted by tau1 head (∈ [0.15, 0.35]) — gradient flows through tau
- TP2: τ predicted by tau2 head (∈ [0.40, 0.60]) — gradient flows through tau
- **Only computed for profitable signals** (quality=1) — no regression signal from losing trades

**Time-to-Peak:**
- Smooth L1 loss on ttp target (peak_bar / horizon)
- Weight: 0.5
- **Only computed for profitable signals** (quality=1)

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
4. Build features (33) -> scale with saved `StandardScaler`
5. For each signal bar: `predict_bar(model, scaled, bar_idx, tf_key, asset_id)` -> `(P(win), tp1_dist, tp2_dist, ttp)`
6. If `P(win) >= 0.3`: emit signal with entry, model-predicted TP levels, and ttp

### Signal Structure
- **Entry**: swept swing level
- **TP1**: `entry * (1 ± tp1_dist)` — conservative target from model regression
- **TP2**: `entry * (1 ± tp2_dist)` — aggressive target from model regression
- **SL**: SFP candle extreme (Low for longs, High for shorts)
- **Best TP**: TP2 if its R:R exceeds TP1's by 1.5x, else TP1
- **TTP**: predicted time-to-peak ∈ [0, 1] (fraction of forward horizon)
- **Expiry**: 3 bars
- **Resolution horizon**: 18 bars (TP hit, SL hit, or mark-to-market at expiry)

### Partial TP Execution
When TP1 is hit before TP2:
1. Signal status changes from `open` → `partial`
2. SL moves to breakeven (entry price)
3. If TP2 is hit → `win` (reward = 0.5 × TP1_R + 0.5 × TP2_R)
4. If stopped at breakeven → `partial_win` (reward = 0.5 × TP1_R)
5. `partial_win` counts as a win in stats

---

## Current Performance (all assets, all TFs)

```
P > 0.3: 14134 trades | Gate WR: 58% | TP1: 46%WR EV=+0.14% | TP2: 29%WR EV=-0.23%
P > 0.5: 9505 trades  | Gate WR: 68% | TP1: 53%WR EV=+0.49% | TP2: 34%WR EV=+0.16%
P > 0.7: 5133 trades  | Gate WR: 78% | TP1: 61%WR EV=+0.89% | TP2: 40%WR EV=+0.63%
```

### Previous Performance (v1: 30 features, fixed τ, no FiLM)
```
P > 0.3: 14403 trades | Gate WR: 58% | TP1: 43%WR EV=+0.12% | TP2: 14%WR EV=-0.70%
P > 0.5: 9478 trades  | Gate WR: 68% | TP1: 49%WR EV=+0.47% | TP2: 17%WR EV=-0.36%
P > 0.7: 4035 trades  | Gate WR: 79% | TP1: 56%WR EV=+0.92% | TP2: 20%WR EV=+0.13%
```

### Key Improvements (v1 → v2)
- **TP2 WR**: 14-20% → 29-40% (doubled across all thresholds)
- **TP2 EV**: negative at all thresholds → positive at P>0.5+
- **TP1 WR**: 43-56% → 46-61% (steady improvement)
- **Both EVs positive** at P>0.5 — both TP targets are now usable

---

## Implemented Improvements (v2)

All 8 improvements from v1 were implemented simultaneously (required full retrain):

1. ~~**TP2 quantile too aggressive**~~ — **Done.** Adaptive tau2 head pins at τ≈0.40 (was fixed at 0.7). TP2 WR doubled.
2. ~~**MFE+ gap**~~ — **Done.** Adaptive tau1 head pins at τ≈0.15 (was fixed at 0.3). More trades reach TP1.
3. ~~**Adaptive τ per signal context**~~ — **Done.** tau1/tau2 heads predict per-sample quantile. In practice both pin to minimums, which is optimal behavior.
4. ~~**Partial TP execution**~~ — **Done.** Server implements TP1→partial→TP2/breakeven flow.
5. ~~**Per-asset/TF model heads**~~ — **Done.** FiLM conditioning (asset+TF embeddings → scale+bias on cls/tp1/tp2 heads).
6. ~~**Direction-aware features**~~ — **Done.** Added `direction_feat` (+1 long, -1 short).
7. ~~**Volume profile features**~~ — **Done.** Added `vwap_distance` and `volume_imbalance`.
8. ~~**Multi-horizon MFE**~~ — **Done.** Time-to-peak head predicts when MFE peak occurs.

## Potential Next Improvements

1. **Tau regularization** — both adaptive tau heads pin at their floor values (tau1=0.15, tau2=0.40). Adding a regularization term (e.g., entropy bonus or KL toward a target distribution) could encourage the model to actually vary tau per sample, producing context-adaptive TP targets.

2. **Trailing stop using TTP** — the time-to-peak head is trained but not yet used for exit logic. If TTP predicts an early peak (ttp < 0.3), could tighten trailing stop after TP1 instead of waiting for TP2.

3. **Multi-horizon training** — current MFE uses fixed 18-bar horizon. Training with multiple horizons (6, 12, 18, 36) and letting the model select optimal horizon per signal could capture both quick scalps and extended moves.

4. **Asymmetric long/short heads** — crypto markets have asymmetric distributions (sharper drops, slower grinds up). Separate TP/tau heads for longs vs shorts could capture this.

5. **15m timeframe performance** — 15m shows weaker performance (38% WR at P>0.5). Could benefit from TF-specific training hyperparameters (e.g., more aggressive filtering, tighter SL, or shorter horizon).

6. **Ensemble/stacking** — train multiple models with different random seeds, use consensus voting for the gate and average for TP regression. Would reduce variance at the cost of inference speed.

7. **Online learning** — periodically fine-tune on recent signals with known outcomes to adapt to regime changes. Would need careful implementation to prevent catastrophic forgetting.

8. **Order book features** — if available, add spread, depth imbalance, and trade flow metrics at signal time. Would require live data integration but could significantly improve gate accuracy.
