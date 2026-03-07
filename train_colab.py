"""Self-contained Colab training script for Liq+Range+SFP v5 model.

Usage in Google Colab:
    1. Upload this file + data/*.csv files to Colab
    2. Run: !python train_colab.py 4h 1h 15min
    3. Download: best_model_liq_range_sfp.pth + liq_range_sfp_scaler.joblib

GPU optimizations: AMP mixed precision, torch.compile, larger batch size.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
USE_AMP = device == "cuda"
BATCH_SIZE = 256 if device == "cuda" else 128
NUM_WORKERS = 2 if device == "cuda" else 0

# ============================================================================
# CLI
# ============================================================================
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

ASSETS = {
    "btc": {"prefix": "btc", "asset_id": 1.0},
    "gold": {"prefix": "gold", "asset_id": 2.0},
    "sol": {"prefix": "sol", "asset_id": 4.0},
    "eth": {"prefix": "eth", "asset_id": 5.0},
}

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
N_FEATURES = 37
HORIZON_MAP = {"15m": 36, "1h": 18, "4h": 18}

print(f"Training on: {TIMEFRAMES} | Assets: {SELECTED_ASSETS} | Batch: {BATCH_SIZE} | AMP: {USE_AMP}")


# ============================================================================
# CORE UTILITIES
# ============================================================================

def compute_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def detect_swings(highs, lows, n):
    length = len(highs)
    swing_highs = np.zeros(length, dtype=bool)
    swing_lows = np.zeros(length, dtype=bool)
    for i in range(n, length - n):
        window_h = highs[i-n:i+n+1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            swing_highs[i] = True
        window_l = lows[i-n:i+n+1]
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            swing_lows[i] = True
    return swing_highs, swing_lows


def build_swing_level_series(highs, lows, swing_highs, swing_lows, n, max_age=150):
    length = len(highs)
    active_sh = [[] for _ in range(length)]
    active_sl = [[] for _ in range(length)]
    active_sh_ages = [[] for _ in range(length)]
    active_sl_ages = [[] for _ in range(length)]
    for i in range(length):
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_highs[j]:
                active_sh[i].append(highs[j])
                active_sh_ages[i].append(i - j)
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_lows[j]:
                active_sl[i].append(lows[j])
                active_sl_ages[i].append(i - j)
    return active_sh, active_sl, active_sh_ages, active_sl_ages


def detect_sfp(df_high, df_low, df_close, df_open, active_sh, active_sl,
               max_sweep_pct=0.05, reclaim_window=1):
    length = len(df_high)
    actions = np.zeros(length, dtype=np.int64)
    swept_levels = np.zeros(length, dtype=np.float64)
    for i in range(length):
        for sl in active_sl[i]:
            if sl <= 0 or df_low[i] >= sl:
                continue
            if (sl - df_low[i]) / sl > max_sweep_pct:
                continue
            if df_open[i] <= sl:
                continue
            for k in range(i, min(i + reclaim_window + 1, length)):
                if df_close[k] > sl and actions[k] == 0:
                    actions[k] = 1
                    swept_levels[k] = sl
                    break
            break
        for sh in active_sh[i]:
            if sh <= 0 or df_high[i] <= sh:
                continue
            if (df_high[i] - sh) / sh > max_sweep_pct:
                continue
            if df_open[i] >= sh:
                continue
            for k in range(i, min(i + reclaim_window + 1, length)):
                if df_close[k] < sh:
                    if actions[k] == 1:
                        actions[k] = 0
                        swept_levels[k] = 0
                    elif actions[k] == 0:
                        actions[k] = 2
                        swept_levels[k] = sh
                    break
            break
    return actions, swept_levels


# ============================================================================
# LIQUIDATION
# ============================================================================

LEVERAGE_TIERS = [(3, 0.05), (5, 0.15), (10, 0.30), (25, 0.25), (50, 0.15), (100, 0.10)]
MM_RATE = 0.004

def compute_liq_price(entry, leverage, side, mm_rate=MM_RATE):
    if side == "long":
        return entry * (1 - 1.0 / leverage + mm_rate)
    else:
        return entry * (1 + 1.0 / leverage - mm_rate)


# ============================================================================
# MARKET STRUCTURE
# ============================================================================

def detect_market_structure(highs, lows, n=10):
    length = len(highs)
    ms_direction = np.zeros(length, dtype=np.float32)
    ms_strength = np.zeros(length, dtype=np.float32)
    last_structure_high = np.zeros(length, dtype=np.float64)
    last_structure_low = np.zeros(length, dtype=np.float64)
    swing_highs, swing_lows = detect_swings(highs, lows, n)
    prev_sh = prev_sl = 0.0
    bullish_count = bearish_count = 0
    current_sh = current_sl = 0.0
    for i in range(length):
        ci = i - n
        if ci >= 0:
            if swing_highs[ci]:
                new_sh = highs[ci]
                if prev_sh > 0:
                    if new_sh > prev_sh:
                        bullish_count += 1
                        bearish_count = max(bearish_count - 1, 0)
                    elif new_sh < prev_sh:
                        bearish_count += 1
                        bullish_count = max(bullish_count - 1, 0)
                prev_sh = new_sh
                current_sh = new_sh
            if swing_lows[ci]:
                new_sl = lows[ci]
                if prev_sl > 0:
                    if new_sl > prev_sl:
                        bullish_count += 1
                        bearish_count = max(bearish_count - 1, 0)
                    elif new_sl < prev_sl:
                        bearish_count += 1
                        bullish_count = max(bullish_count - 1, 0)
                prev_sl = new_sl
                current_sl = new_sl
        if bullish_count > bearish_count:
            ms_direction[i] = 1.0
            ms_strength[i] = min(bullish_count, 5) / 5.0
        elif bearish_count > bullish_count:
            ms_direction[i] = -1.0
            ms_strength[i] = min(bearish_count, 5) / 5.0
        last_structure_high[i] = current_sh if current_sh > 0 else highs[i]
        last_structure_low[i] = current_sl if current_sl > 0 else lows[i]
    return ms_direction, ms_strength, last_structure_high, last_structure_low


# ============================================================================
# RANGE DETECTION (zone-based v2)
# ============================================================================

@dataclass
class PriceZone:
    bottom: float
    top: float
    level: float
    count: int
    bars: list = field(default_factory=list)

@dataclass
class ZoneRange:
    resistance: PriceZone
    support: PriceZone
    start: int
    confirmed: int
    bars_inside: int
    concentration: float
    high: float = 0.0
    low: float = 0.0
    touches_high: int = 0
    touches_low: int = 0
    broken: bool = False
    broken_dir: int = 0
    broken_bar: int = -1
    recaptured: bool = False
    recapture_bar: int = -1
    is_macro: bool = False

RANGE_HEIGHT_PCT = {
    "15m": (0.003, 0.06), "1h": (0.005, 0.12), "4h": (0.005, 0.12),
}
RANGE_SFP_PARAMS = {
    "15m": {"n_swing": 3, "min_bars": 30, "max_bars": 400, "min_zone_count": 4, "min_time_concentration": 0.80, "recapture_window": 10},
    "1h":  {"n_swing": 3, "min_bars": 15, "max_bars": 300, "min_zone_count": 3, "min_time_concentration": 0.70, "recapture_window": 5},
    "4h":  {"n_swing": 3, "min_bars": 20, "max_bars": 300, "min_zone_count": 3, "min_time_concentration": 0.65, "recapture_window": 3},
}


def cluster_levels(prices_with_bars, pct=0.015):
    if not prices_with_bars:
        return []
    sorted_levels = sorted(prices_with_bars, key=lambda x: x[0])
    clusters = []
    current = [sorted_levels[0]]
    current_mean = sorted_levels[0][0]
    current_min = sorted_levels[0][0]
    for price, bar_idx in sorted_levels[1:]:
        width_ok = (price - current_min) / current_mean <= pct * 2
        if current_mean > 0 and abs(price - current_mean) / current_mean <= pct and width_ok:
            current.append((price, bar_idx))
            current_mean = sum(p for p, _ in current) / len(current)
        else:
            clusters.append(current)
            current = [(price, bar_idx)]
            current_mean = price
            current_min = price
    clusters.append(current)
    zones = []
    for cluster in clusters:
        prices = [p for p, _ in cluster]
        bars = [b for _, b in cluster]
        zones.append(PriceZone(bottom=min(prices), top=max(prices),
                               level=sum(prices)/len(prices), count=len(prices), bars=bars))
    return zones


def _make_zone_range(rz, sz, earliest_bar, min_bars, min_height_pct, max_height_pct):
    range_height = rz.level - sz.level
    mid_price = (rz.level + sz.level) / 2.0
    if mid_price <= 0 or range_height <= 0:
        return None
    height_pct = range_height / mid_price
    if height_pct < min_height_pct or height_pct > max_height_pct:
        return None
    confirmed_bar = earliest_bar + min_bars
    zr = ZoneRange(resistance=rz, support=sz, start=earliest_bar, confirmed=confirmed_bar,
                   bars_inside=0, concentration=0.0, high=rz.top, low=sz.bottom,
                   touches_high=rz.count, touches_low=sz.count)
    zr._vol_compression = 1.0
    zr._height_pct = height_pct
    return zr


def _detect_ranges_in_window(highs, lows, closes, opens, w_start, w_end, global_offset,
                              n_swing, min_bars, max_bars, min_height_pct, max_height_pct,
                              min_time_concentration, min_zone_count, cluster_pct):
    wh = highs[w_start:w_end]
    wl = lows[w_start:w_end]
    wc = closes[w_start:w_end]
    wo = opens[w_start:w_end]
    wn = len(wh)
    if wn < min_bars:
        return []
    swing_highs, swing_lows = detect_swings(wh, wl, n_swing)
    resistance_cands = []
    support_cands = []
    for i in range(wn):
        if swing_highs[i]:
            resistance_cands.append((wh[i], i))
        if swing_lows[i]:
            support_cands.append((wl[i], i))
        cr = wh[i] - wl[i]
        if cr <= 0:
            continue
        bt = max(wo[i], wc[i])
        bb = min(wo[i], wc[i])
        if (wh[i] - bt) / cr > 0.30:
            resistance_cands.append((wh[i], i))
        if (bb - wl[i]) / cr > 0.30:
            support_cands.append((wl[i], i))
    rz_list = [z for z in cluster_levels(resistance_cands, pct=cluster_pct) if z.count >= min_zone_count]
    sz_list = [z for z in cluster_levels(support_cands, pct=cluster_pct) if z.count >= min_zone_count]
    window_ranges = []
    for sz in sz_list:
        for rz in rz_list:
            if rz.level <= sz.level or sz.top >= rz.bottom:
                continue
            h_pct = (rz.level - sz.level) / ((rz.level + sz.level) / 2.0)
            if h_pct < min_height_pct or h_pct > max_height_pct:
                continue
            earliest = min(min(sz.bars), min(rz.bars))
            check_end = earliest + min_bars
            if check_end > wn:
                continue
            bi = sum(1 for j in range(earliest, check_end) if wc[j] >= sz.bottom and wc[j] <= rz.top)
            conc = bi / min_bars
            if conc < min_time_concentration:
                continue
            g_earliest = earliest + w_start
            g_confirmed = check_end - 1 + w_start
            g_sz = PriceZone(sz.bottom, sz.top, sz.level, sz.count, [b + w_start for b in sz.bars])
            g_rz = PriceZone(rz.bottom, rz.top, rz.level, rz.count, [b + w_start for b in rz.bars])
            zr = ZoneRange(resistance=g_rz, support=g_sz, start=g_earliest, confirmed=g_confirmed,
                           bars_inside=bi, concentration=conc, high=g_rz.top, low=g_sz.bottom,
                           touches_high=g_rz.count, touches_low=g_sz.count)
            zr._time_concentration = conc
            zr._vol_compression = 1.0
            zr._height_pct = h_pct
            is_dup = False
            for ex in window_ranges:
                if (abs(ex.high - zr.high) / (zr.high + 1e-8) < 0.03
                        and abs(ex.low - zr.low) / (zr.low + 1e-8) < 0.03):
                    if zr.concentration > ex.concentration:
                        window_ranges.remove(ex)
                    else:
                        is_dup = True
                    break
            if not is_dup:
                window_ranges.append(zr)
    return window_ranges


def detect_ranges_v2(highs, lows, closes, opens, n_swing=3, min_bars=15, max_bars=250,
                     min_height_pct=0.005, max_height_pct=0.08, min_time_concentration=0.80,
                     min_zone_count=2, cluster_pct=0.015, recapture_window=5):
    n = len(highs)
    window_size = max(max_bars * 2, 500)
    window_step = max(max_bars, 250)
    raw_ranges = []
    for w_start in range(0, max(1, n - min_bars), window_step):
        w_end = min(w_start + window_size, n)
        window_ranges = _detect_ranges_in_window(
            highs, lows, closes, opens, w_start, w_end, w_start,
            n_swing, min_bars, max_bars, min_height_pct, max_height_pct,
            min_time_concentration, min_zone_count, cluster_pct)
        raw_ranges.extend(window_ranges)
    ranges = []
    for zr in raw_ranges:
        is_dup = False
        for existing in ranges:
            if (abs(existing.high - zr.high) / (zr.high + 1e-8) < 0.03
                    and abs(existing.low - zr.low) / (zr.low + 1e-8) < 0.03
                    and abs(existing.start - zr.start) <= max_bars):
                if zr.concentration > existing.concentration:
                    ranges.remove(existing)
                else:
                    is_dup = True
                break
        if not is_dup:
            ranges.append(zr)

    # Break detection + recapture + post-break ranges
    post_break_ranges = []
    for r in ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True
                r.broken_dir = 2
                r.broken_bar = i
                recaptured = False
                for j in range(i+1, min(i+1+recapture_window, n)):
                    if closes[j] >= r.support.bottom and closes[j] <= r.resistance.top:
                        r.broken = False
                        r.recaptured = True
                        r.recapture_bar = j
                        recaptured = True
                        break
                break_low = lows[i]
                break_low_bar = i
                for j in range(i+1, min(i + max_bars // 2, n)):
                    if lows[j] < break_low:
                        break_low = lows[j]
                        break_low_bar = j
                    if closes[j] > r.support.top:
                        break
                new_support = PriceZone(break_low, break_low * 1.005, break_low, 1, [break_low_bar])
                new_resistance = r.support
                pb = _make_zone_range(new_resistance, new_support, break_low_bar, min_bars, min_height_pct, max_height_pct)
                if pb is not None:
                    bi = sum(1 for j in range(break_low_bar, min(break_low_bar + max_bars, n))
                             if closes[j] >= new_support.bottom and closes[j] <= new_resistance.top)
                    total = min(max_bars, n - break_low_bar)
                    if total >= min_bars:
                        conc = bi / total
                        if conc >= min_time_concentration * 0.6:
                            pb.bars_inside = bi
                            pb.concentration = conc
                            pb._time_concentration = conc
                            post_break_ranges.append(pb)
                if recaptured:
                    continue
                break
            elif closes[i] > r.resistance.top:
                r.broken = True
                r.broken_dir = 1
                r.broken_bar = i
                recaptured = False
                for j in range(i+1, min(i+1+recapture_window, n)):
                    if closes[j] >= r.support.bottom and closes[j] <= r.resistance.top:
                        r.broken = False
                        r.recaptured = True
                        r.recapture_bar = j
                        recaptured = True
                        break
                break_high = highs[i]
                break_high_bar = i
                for j in range(i+1, min(i + max_bars // 2, n)):
                    if highs[j] > break_high:
                        break_high = highs[j]
                        break_high_bar = j
                    if closes[j] < r.resistance.bottom:
                        break
                new_resistance = PriceZone(break_high * 0.995, break_high, break_high, 1, [break_high_bar])
                new_support = r.resistance
                pb = _make_zone_range(new_resistance, new_support, break_high_bar, min_bars, min_height_pct, max_height_pct)
                if pb is not None:
                    bi = sum(1 for j in range(break_high_bar, min(break_high_bar + max_bars, n))
                             if closes[j] >= new_support.bottom and closes[j] <= new_resistance.top)
                    total = min(max_bars, n - break_high_bar)
                    if total >= min_bars:
                        conc = bi / total
                        if conc >= min_time_concentration * 0.6:
                            pb.bars_inside = bi
                            pb.concentration = conc
                            pb._time_concentration = conc
                            post_break_ranges.append(pb)
                if recaptured:
                    continue
                break

    for r in post_break_ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True; r.broken_dir = 2; r.broken_bar = i; break
            elif closes[i] > r.resistance.top:
                r.broken = True; r.broken_dir = 1; r.broken_bar = i; break
    ranges.extend(post_break_ranges)

    # Macro range detection
    macro_cluster_pct = 0.03
    macro_min_zone_count = max(min_zone_count - 1, 2)
    macro_max_height_pct = 0.18 if max_height_pct >= 0.10 else 0.10
    macro_min_time_conc = min_time_concentration * 0.90
    macro_raw = []
    for w_start in range(0, max(1, n - min_bars), max(max_bars, 250)):
        w_end = min(w_start + max(max_bars * 2, 500), n)
        wr = _detect_ranges_in_window(
            highs, lows, closes, opens, w_start, w_end, w_start,
            n_swing, min_bars, max_bars, min_height_pct, macro_max_height_pct,
            macro_min_time_conc, macro_min_zone_count, macro_cluster_pct)
        for w in wr:
            w.is_macro = True
        macro_raw.extend(wr)
    macro_ranges = []
    for zr in macro_raw:
        is_dup = False
        for existing in macro_ranges:
            if (abs(existing.high - zr.high) / (zr.high + 1e-8) < 0.03
                    and abs(existing.low - zr.low) / (zr.low + 1e-8) < 0.03
                    and abs(existing.start - zr.start) <= max_bars):
                if zr.concentration > existing.concentration:
                    macro_ranges.remove(existing)
                else:
                    is_dup = True
                break
        if not is_dup:
            macro_ranges.append(zr)
    for r in macro_ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True; r.broken_dir = 2; r.broken_bar = i; break
            elif closes[i] > r.resistance.top:
                r.broken = True; r.broken_dir = 1; r.broken_bar = i; break
    ranges.extend(macro_ranges)

    # Build active_ranges_per_bar
    active_ranges = [[] for _ in range(n)]
    for r in ranges:
        end_bar = min(r.start + max_bars, n)
        for i in range(r.confirmed, end_bar):
            if r.recaptured:
                if r.broken_bar <= i < r.recapture_bar:
                    continue
            elif r.broken and i >= r.broken_bar:
                break
            active_ranges[i].append(r)
    return active_ranges, ranges


# ============================================================================
# RANGE TP/SL/MAE LABELS
# ============================================================================

def compute_range_tp_sl_labels(highs, lows, closes, actions, swept_levels, signal_map, horizon=18):
    length = len(actions)
    quality = np.zeros(length, dtype=np.int64)
    mfe_labels = np.zeros(length, dtype=np.float32)
    sl_labels = np.zeros(length, dtype=np.float32)
    ttp_labels = np.zeros(length, dtype=np.float32)
    mae_labels = np.zeros(length, dtype=np.float32)
    stop_buffer = 0.002
    for i in range(length):
        if actions[i] == 0:
            continue
        if i + horizon >= length:
            actions[i] = 0; continue
        entry = swept_levels[i]
        if entry <= 0:
            actions[i] = 0; continue
        sig = signal_map.get(i)
        if sig is None:
            actions[i] = 0; continue
        r = sig.range_ref
        range_height = r.resistance.level - r.support.level
        if range_height <= 0:
            actions[i] = 0; continue
        if actions[i] == 1:
            stop_price = r.support.bottom * (1 - stop_buffer)
            sl = max((entry - stop_price) / entry, 0.001)
        else:
            stop_price = r.resistance.top * (1 + stop_buffer)
            sl = max((stop_price - entry) / entry, 0.001)
        sl_labels[i] = np.clip(sl, 0.001, 0.08)
        mfe = 0.0
        mae = 0.0
        peak_bar = 0
        for j in range(i+1, min(i+1+horizon, length)):
            if actions[i] == 1:
                if lows[j] <= stop_price:
                    break
                favorable = (highs[j] - entry) / entry
                adverse = (entry - lows[j]) / entry
            else:
                if highs[j] >= stop_price:
                    break
                favorable = (entry - lows[j]) / entry
                adverse = (highs[j] - entry) / entry
            if favorable > mfe:
                mfe = favorable
                peak_bar = j - i
            if adverse > mae:
                mae = adverse
        mfe_labels[i] = np.clip(mfe, 0.0, 0.15)
        mae_labels[i] = np.clip(mae, 0.0, 0.15)
        ttp_labels[i] = np.clip(peak_bar / horizon, 0.0, 1.0)
        quality[i] = 1 if mfe > sl else 0
    return quality, mfe_labels, sl_labels, ttp_labels, mae_labels


# ============================================================================
# LIQ+RANGE+SFP SIGNAL (combined label generator)
# ============================================================================

@dataclass
class LiqRangeSFPSignal:
    bar_idx: int
    direction: int
    swept_level: float
    range_ref: ZoneRange = field(repr=False)
    range_height_pct: float = 0.0
    range_touches: float = 0.0
    range_concentration: float = 0.0
    range_age: float = 0.0
    sweep_depth_range: float = 0.0
    reclaim_strength_range: float = 0.0
    n_liq_swept: int = 0
    weighted_liq_swept: float = 0.0
    max_leverage_swept: int = 0
    liq_cascade_depth: float = 0.0
    liq_cluster_density: float = 0.0
    n_swings_with_liq: int = 0
    ms_alignment: float = 0.0
    ms_strength: float = 0.0
    signal_type: int = 0
    is_recaptured: float = 0.0
    is_nested: float = 0.0
    touch_symmetry: float = 0.0
    boundary_rejection_avg: float = 0.0
    range_position: float = 0.0


def generate_labels(highs, lows, closes, opens, volumes=None, tf_key="4h"):
    n = len(highs)
    atr = compute_atr(highs, lows, closes, period=14)
    params = RANGE_SFP_PARAMS.get(tf_key, RANGE_SFP_PARAMS["4h"])
    height_min, height_max = RANGE_HEIGHT_PCT.get(tf_key, (0.005, 0.08))
    print(f"  [LiqRangeSFP/{tf_key}] Detecting ranges...")
    active_per_bar, all_ranges = detect_ranges_v2(
        highs, lows, closes, opens,
        n_swing=params["n_swing"], min_bars=params["min_bars"], max_bars=params["max_bars"],
        min_height_pct=height_min, max_height_pct=height_max,
        min_zone_count=params.get("min_zone_count", 2),
        min_time_concentration=params.get("min_time_concentration", 0.80),
        recapture_window=params.get("recapture_window", 5))
    n_post = sum(1 for r in all_ranges if r.support.count == 1 or r.resistance.count == 1)
    n_macro = sum(1 for r in all_ranges if r.is_macro)
    n_recaptured = sum(1 for r in all_ranges if r.recaptured)
    n_primary = len(all_ranges) - n_post - n_macro
    print(f"    Found {len(all_ranges)} ranges ({n_primary} primary, {n_post} post-break, {n_macro} macro, {n_recaptured} recaptured)")

    # SFP detection
    print(f"  [LiqRangeSFP/{tf_key}] Detecting swings + SFPs...")
    reclaim_windows = {5: 1, 10: 3}
    results = {}
    liq_swing_data = {}
    for ns in [5, 10]:
        sh, sl = detect_swings(highs, lows, ns)
        active_sh, active_sl, _, _ = build_swing_level_series(highs, lows, sh, sl, ns, max_age=150)
        sfp_actions, sfp_swept = detect_sfp(highs, lows, closes, opens, active_sh, active_sl, reclaim_window=reclaim_windows[ns])
        results[ns] = (sfp_actions, sfp_swept)
        active_sh_liq, active_sl_liq, _, _ = build_swing_level_series(highs, lows, sh, sl, ns, max_age=200)
        liq_swing_data[ns] = (active_sh_liq, active_sl_liq)

    actions_5, swept_5 = results[5]
    actions_10, swept_10 = results[10]
    raw_actions = np.zeros(n, dtype=np.int64)
    raw_swept = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a5, a10 = actions_5[i], actions_10[i]
        if a5 == a10:
            raw_actions[i] = a5
            raw_swept[i] = swept_5[i] if swept_5[i] > 0 else swept_10[i]
        elif a5 != 0 and a10 == 0:
            raw_actions[i] = a5; raw_swept[i] = swept_5[i]
        elif a10 != 0 and a5 == 0:
            raw_actions[i] = a10; raw_swept[i] = swept_10[i]
    print(f"    Total SFPs detected: {int(np.sum(raw_actions != 0))}")

    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    # Boundary filter + liq features
    actions = np.zeros(n, dtype=np.int64)
    swept_levels = np.zeros(n, dtype=np.float64)
    signal_map = {}

    for i in range(n):
        if raw_actions[i] == 0 or not active_per_bar[i]:
            continue
        swept = raw_swept[i]
        direction = raw_actions[i]
        best_range = None
        best_score = -1.0
        if tf_key == "15m":
            zone_buffer = 0.0; strict_reclaim = True
        elif tf_key == "4h":
            zone_buffer = atr[i] * 0.15; strict_reclaim = False
        else:
            zone_buffer = atr[i] * 0.1; strict_reclaim = False

        for r in active_per_bar[i]:
            range_height = r.high - r.low
            if range_height <= 0:
                continue
            is_boundary = False
            if direction == 1:
                reclaim_ok = closes[i] > r.support.top if strict_reclaim else closes[i] >= r.support.level
                if (swept >= r.support.bottom - zone_buffer and swept <= r.support.top + zone_buffer
                        and lows[i] < r.support.level and reclaim_ok):
                    is_boundary = True
            elif direction == 2:
                reclaim_ok = closes[i] < r.resistance.bottom if strict_reclaim else closes[i] <= r.resistance.level
                if (swept >= r.resistance.bottom - zone_buffer and swept <= r.resistance.top + zone_buffer
                        and highs[i] > r.resistance.level and reclaim_ok):
                    is_boundary = True
            if not is_boundary:
                continue
            score = r.concentration * min(r.touches_high, r.touches_low)
            if score > best_score:
                best_score = score; best_range = r

        if best_range is None:
            continue

        # Liq features
        n_liq_swept = 0; weighted_liq = 0.0; max_leverage = 0
        liq_prices_swept = []; swings_with_liq = set(); total_side = 0
        local_atr = atr[i] if atr[i] > 0 else closes[i] * 0.01
        proximity = local_atr * 0.5; price = closes[i]
        for ns in [5, 10]:
            active_sh, active_sl = liq_swing_data[ns]
            if direction == 1:
                for sl_price in active_sl[i]:
                    for lev, weight in LEVERAGE_TIERS:
                        liq_p = compute_liq_price(sl_price, lev, "long")
                        if liq_p < price:
                            total_side += 1
                            if liq_p >= lows[i] - proximity:
                                n_liq_swept += 1; weighted_liq += weight
                                max_leverage = max(max_leverage, lev)
                                liq_prices_swept.append(liq_p)
                                swings_with_liq.add(round(sl_price, 2))
            else:
                for sh_price in active_sh[i]:
                    for lev, weight in LEVERAGE_TIERS:
                        liq_p = compute_liq_price(sh_price, lev, "short")
                        if liq_p > price:
                            total_side += 1
                            if liq_p <= highs[i] + proximity:
                                n_liq_swept += 1; weighted_liq += weight
                                max_leverage = max(max_leverage, lev)
                                liq_prices_swept.append(liq_p)
                                swings_with_liq.add(round(sh_price, 2))

        r = best_range
        range_height = r.high - r.low
        mid_price = (r.high + r.low) / 2.0
        if direction == 1:
            sweep_depth = (r.support.top - lows[i]) / range_height
            reclaim_strength = (closes[i] - r.support.top) / range_height
        else:
            sweep_depth = (highs[i] - r.resistance.bottom) / range_height
            reclaim_strength = (r.resistance.bottom - closes[i]) / range_height
        cascade_depth = max(liq_prices_swept) - min(liq_prices_swept) if liq_prices_swept else 0.0
        cluster_density = n_liq_swept / max(total_side, 1)
        ms_dir = ms_direction[i]
        if direction == 1:
            ms_align = 1.0 if ms_dir > 0 else (-1.0 if ms_dir < 0 else 0.0)
        else:
            ms_align = 1.0 if ms_dir < 0 else (-1.0 if ms_dir > 0 else 0.0)

        actions[i] = direction
        swept_levels[i] = swept
        t_high = r.touches_high; t_low = r.touches_low
        touch_sym = min(t_high, t_low) / max(t_high, t_low, 1)
        if direction == 1:
            rej_bars = [b for b in r.support.bars if b <= i]
            rej_avg = sum(r.support.level - lows[b] for b in rej_bars if b < n) / max(len(rej_bars), 1)
        else:
            rej_bars = [b for b in r.resistance.bars if b <= i]
            rej_avg = sum(highs[b] - r.resistance.level for b in rej_bars if b < n) / max(len(rej_bars), 1)
        boundary_rej = rej_avg / (range_height + 1e-8)
        range_pos = (closes[i] - r.low) / (range_height + 1e-8)
        is_nested = 1.0 if r.is_macro else 0.0
        if not r.is_macro:
            for ar in active_per_bar[i]:
                if ar.is_macro and ar.low <= r.low and ar.high >= r.high:
                    is_nested = 1.0; break

        signal_map[i] = LiqRangeSFPSignal(
            bar_idx=i, direction=direction, swept_level=swept, range_ref=r,
            range_height_pct=range_height / (mid_price + 1e-8),
            range_touches=min(r.touches_high, r.touches_low),
            range_concentration=r.concentration,
            range_age=(i - r.confirmed) / 200.0,
            sweep_depth_range=sweep_depth, reclaim_strength_range=reclaim_strength,
            n_liq_swept=n_liq_swept, weighted_liq_swept=weighted_liq,
            max_leverage_swept=max_leverage, liq_cascade_depth=cascade_depth,
            liq_cluster_density=cluster_density, n_swings_with_liq=len(swings_with_liq),
            ms_alignment=ms_align, ms_strength=ms_strength_arr[i],
            signal_type=0, is_recaptured=1.0 if r.recaptured else 0.0,
            is_nested=is_nested, touch_symmetry=touch_sym,
            boundary_rejection_avg=boundary_rej, range_position=np.clip(range_pos, 0.0, 1.0))

    n_sfp = int(np.sum(actions != 0))
    n_with_liq = sum(1 for s in signal_map.values() if s.n_liq_swept > 0)
    print(f"    SFP boundary: {n_sfp} signals | With liq: {n_with_liq}")

    # Range approach signals (second pass)
    n_approach = 0
    for i in range(n):
        if actions[i] != 0 or not active_per_bar[i]:
            continue
        if tf_key == "15m":
            zone_buffer = 0.0
        elif tf_key == "4h":
            zone_buffer = atr[i] * 0.15
        else:
            zone_buffer = atr[i] * 0.1
        best_range = None; best_score = -1.0; best_dir = 0
        for r in active_per_bar[i]:
            range_height = r.high - r.low
            if range_height <= 0:
                continue
            if (lows[i] <= r.support.top + zone_buffer and lows[i] < r.support.level and closes[i] > r.support.level):
                score = r.concentration * min(r.touches_high, r.touches_low)
                if score > best_score:
                    best_score = score; best_range = r; best_dir = 1
            if (highs[i] >= r.resistance.bottom - zone_buffer and highs[i] > r.resistance.level and closes[i] < r.resistance.level):
                score = r.concentration * min(r.touches_high, r.touches_low)
                if score > best_score:
                    best_score = score; best_range = r; best_dir = 2
        if best_range is None:
            continue
        r = best_range; direction = best_dir
        range_height = r.high - r.low; mid_price = (r.high + r.low) / 2.0
        entry = r.support.level if direction == 1 else r.resistance.level
        if direction == 1:
            sweep_depth = (r.support.top - lows[i]) / range_height
            reclaim_strength = (closes[i] - r.support.top) / range_height
        else:
            sweep_depth = (highs[i] - r.resistance.bottom) / range_height
            reclaim_strength = (r.resistance.bottom - closes[i]) / range_height
        ms_dir = ms_direction[i]
        if direction == 1:
            ms_align = 1.0 if ms_dir > 0 else (-1.0 if ms_dir < 0 else 0.0)
        else:
            ms_align = 1.0 if ms_dir < 0 else (-1.0 if ms_dir > 0 else 0.0)
        t_high = r.touches_high; t_low = r.touches_low
        touch_sym = min(t_high, t_low) / max(t_high, t_low, 1)
        if direction == 1:
            rej_bars = [b for b in r.support.bars if b <= i]
            rej_avg = sum(r.support.level - lows[b] for b in rej_bars if b < n) / max(len(rej_bars), 1)
        else:
            rej_bars = [b for b in r.resistance.bars if b <= i]
            rej_avg = sum(highs[b] - r.resistance.level for b in rej_bars if b < n) / max(len(rej_bars), 1)
        boundary_rej = rej_avg / (range_height + 1e-8)
        range_pos = (closes[i] - r.low) / (range_height + 1e-8)
        is_nested = 1.0 if r.is_macro else 0.0
        if not r.is_macro:
            for ar in active_per_bar[i]:
                if ar.is_macro and ar.low <= r.low and ar.high >= r.high:
                    is_nested = 1.0; break
        actions[i] = direction
        swept_levels[i] = entry
        signal_map[i] = LiqRangeSFPSignal(
            bar_idx=i, direction=direction, swept_level=entry, range_ref=r,
            range_height_pct=range_height / (mid_price + 1e-8),
            range_touches=min(r.touches_high, r.touches_low),
            range_concentration=r.concentration,
            range_age=(i - r.confirmed) / 200.0,
            sweep_depth_range=sweep_depth, reclaim_strength_range=reclaim_strength,
            n_liq_swept=0, weighted_liq_swept=0.0, max_leverage_swept=0,
            liq_cascade_depth=0.0, liq_cluster_density=0.0, n_swings_with_liq=0,
            ms_alignment=ms_align, ms_strength=ms_strength_arr[i],
            signal_type=1, is_recaptured=1.0 if r.recaptured else 0.0,
            is_nested=is_nested, touch_symmetry=touch_sym,
            boundary_rejection_avg=boundary_rej, range_position=np.clip(range_pos, 0.0, 1.0))
        n_approach += 1

    total = int(np.sum(actions != 0))
    print(f"    Range approach: {n_approach} | Total: {total}")

    quality, mfe_labels, sl_labels, ttp_labels, mae_labels = compute_range_tp_sl_labels(
        highs, lows, closes, actions, swept_levels, signal_map, horizon=HORIZON_MAP.get(tf_key, 18))
    if total > 0:
        n_prof = int(np.sum((actions != 0) & (quality == 1)))
        mask = actions != 0
        print(f"  [LiqRangeSFP/{tf_key}] {total} signals -> profitable: {n_prof} ({n_prof/total*100:.0f}%)")
    return actions, quality, mfe_labels, sl_labels, ttp_labels, swept_levels, signal_map, mae_labels


# ============================================================================
# HTF FEATURES
# ============================================================================

def _compute_rsi(closes, period=14):
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_gain = np.zeros_like(closes, dtype=np.float64)
    avg_loss = np.zeros_like(closes, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    for i in range(period+1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = 50.0
    return rsi


def compute_htf_features(highs, lows, closes, n, resample_factor=4):
    rf = resample_factor
    n_groups = n // rf
    remainder = n - n_groups * rf
    htf_trend = np.zeros(n, dtype=np.float32)
    htf_rsi = np.zeros(n, dtype=np.float32) + 0.5
    htf_ms_direction = np.zeros(n, dtype=np.float32)
    htf_ms_strength = np.zeros(n, dtype=np.float32)
    if n_groups < 22:
        return htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength
    start = remainder
    rs_highs = np.array([highs[start + i*rf: start + (i+1)*rf].max() for i in range(n_groups)])
    rs_lows = np.array([lows[start + i*rf: start + (i+1)*rf].min() for i in range(n_groups)])
    rs_closes = np.array([closes[start + (i+1)*rf - 1] for i in range(n_groups)])
    ema = np.zeros(n_groups, dtype=np.float64)
    ema[0] = rs_closes[0]
    alpha = 2.0 / 22.0
    for i in range(1, n_groups):
        ema[i] = alpha * rs_closes[i] + (1 - alpha) * ema[i-1]
    htf_trend_rs = np.clip((rs_closes - ema) / (rs_closes + 1e-8), -0.5, 0.5).astype(np.float32)
    htf_rsi_rs = (_compute_rsi(rs_closes, period=14) / 100.0).astype(np.float32)
    htf_ms_dir_rs, htf_ms_str_rs, _, _ = detect_market_structure(rs_highs, rs_lows, n=10)
    for g in range(n_groups):
        bar_start = start + g * rf
        bar_end = start + (g+1) * rf
        htf_trend[bar_start:bar_end] = htf_trend_rs[g]
        htf_rsi[bar_start:bar_end] = htf_rsi_rs[g]
        htf_ms_direction[bar_start:bar_end] = htf_ms_dir_rs[g]
        htf_ms_strength[bar_start:bar_end] = htf_ms_str_rs[g]
    if remainder > 0 and n_groups > 0:
        htf_trend[:start] = htf_trend_rs[0]
        htf_rsi[:start] = htf_rsi_rs[0]
        htf_ms_direction[:start] = htf_ms_dir_rs[0]
        htf_ms_strength[:start] = htf_ms_str_rs[0]
    return htf_trend, htf_rsi, htf_ms_direction, htf_ms_strength


# ============================================================================
# FEATURE ENGINEERING (37 features)
# ============================================================================

def build_features(df, actions, signal_map, tf_hours, asset_id=1.0):
    n = len(df)
    highs = df["High"].values; lows = df["Low"].values
    closes = df["Close"].values; opens = df["Open"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.ones(n)
    atr = compute_atr(highs, lows, closes, period=14)
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)
    feat = pd.DataFrame()

    # 6 range features
    range_height_pct = np.zeros(n, dtype=np.float32)
    range_touches_norm = np.zeros(n, dtype=np.float32)
    range_concentration = np.zeros(n, dtype=np.float32)
    range_age = np.zeros(n, dtype=np.float32)
    sweep_depth_range = np.zeros(n, dtype=np.float32)
    reclaim_strength_range = np.zeros(n, dtype=np.float32)
    # 6 liq features
    n_liq_swept_norm = np.zeros(n, dtype=np.float32)
    weighted_liq_swept = np.zeros(n, dtype=np.float32)
    max_leverage_norm = np.zeros(n, dtype=np.float32)
    liq_cascade_depth = np.zeros(n, dtype=np.float32)
    liq_cluster_density = np.zeros(n, dtype=np.float32)
    n_swings_with_liq_norm = np.zeros(n, dtype=np.float32)
    # 6 SFP candle features
    body_ratio = np.zeros(n, dtype=np.float32)
    wick_ratio = np.zeros(n, dtype=np.float32)
    vol_spike = np.zeros(n, dtype=np.float32)
    close_position = np.zeros(n, dtype=np.float32)
    zone_sl_dist = np.zeros(n, dtype=np.float32)
    zone_tp_dist = np.zeros(n, dtype=np.float32)
    vol_ma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i, sig in signal_map.items():
        r = sig.range_ref; entry = sig.swept_level
        range_height_pct[i] = sig.range_height_pct
        range_touches_norm[i] = min(sig.range_touches, 5) / 5.0
        range_concentration[i] = sig.range_concentration
        range_age[i] = sig.range_age
        sweep_depth_range[i] = sig.sweep_depth_range
        reclaim_strength_range[i] = sig.reclaim_strength_range
        n_liq_swept_norm[i] = min(sig.n_liq_swept, 30) / 30.0
        weighted_liq_swept[i] = min(sig.weighted_liq_swept, 3.0) / 3.0
        max_leverage_norm[i] = sig.max_leverage_swept / 100.0
        local_atr = atr[i] if atr[i] > 0 else 1e-8
        liq_cascade_depth[i] = np.clip(sig.liq_cascade_depth / local_atr, 0, 5)
        liq_cluster_density[i] = sig.liq_cluster_density
        n_swings_with_liq_norm[i] = min(sig.n_swings_with_liq, 10) / 10.0
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
    # Context features (6)
    feat["rsi"] = df["rsi"].values / 100.0 if "rsi" in df.columns else 0.5
    feat["trend_strength"] = ((df["Close"] - df["ema_21"]) / df["Close"]).values if "ema_21" in df.columns else 0.0
    feat["ms_alignment"] = np.zeros(n, dtype=np.float32)
    feat["ms_strength"] = ms_strength_arr
    for i, sig in signal_map.items():
        feat.at[i, "ms_alignment"] = sig.ms_alignment
    feat["tf_hours"] = tf_hours / 4.0
    feat["asset_id"] = asset_id
    # Range fingerprint (6)
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
        boundary_rejection_avg_arr[i] = np.clip(sig.boundary_rejection_avg, 0, 2.0)
        range_position_arr[i] = sig.range_position
    feat["signal_type"] = signal_type_arr
    feat["is_recaptured"] = is_recaptured_arr
    feat["is_nested"] = is_nested_arr
    feat["touch_symmetry"] = touch_symmetry_arr
    feat["boundary_rejection_avg"] = boundary_rejection_avg_arr
    feat["range_position"] = range_position_arr
    # New features (3)
    direction_arr = np.zeros(n, dtype=np.float32)
    for i, sig in signal_map.items():
        direction_arr[i] = 1.0 if sig.direction == 1 else -1.0
    feat["direction_feat"] = direction_arr
    vwap_20 = (df["Close"] * volumes).rolling(20, min_periods=1).sum() / pd.Series(volumes).rolling(20, min_periods=1).sum()
    vwap_dist = ((closes - vwap_20.values) / (closes + 1e-8)).astype(np.float32)
    feat["vwap_distance"] = np.clip(vwap_dist, -0.05, 0.05)
    up_vol = np.where(closes >= opens, volumes, 0.0)
    up_vol_10 = pd.Series(up_vol).rolling(10, min_periods=1).sum().values
    total_vol_10 = pd.Series(volumes).rolling(10, min_periods=1).sum().values
    vol_imbalance = (up_vol_10 / (total_vol_10 + 1e-8) - 0.5).astype(np.float32)
    feat["volume_imbalance"] = np.clip(vol_imbalance, -0.5, 0.5)
    # HTF features (4)
    htf_trend, htf_rsi, htf_ms_dir, htf_ms_str = compute_htf_features(highs, lows, closes, n)
    feat["htf_trend"] = htf_trend
    feat["htf_rsi"] = htf_rsi
    feat["htf_ms_direction"] = htf_ms_dir
    feat["htf_ms_strength"] = htf_ms_str

    # Drop warmup + cleanup
    drop_n = 30
    feat = feat.iloc[drop_n:].reset_index(drop=True)
    actions = actions[drop_n:]
    signal_map_shifted = {k - drop_n: v for k, v in signal_map.items() if k >= drop_n}
    feat = feat.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    feat["vol_spike"] = feat["vol_spike"].clip(0, 5.0)
    feat["trend_strength"] = feat["trend_strength"].clip(-0.5, 0.5)
    feat["sweep_depth_range"] = feat["sweep_depth_range"].clip(0, 2.0)
    feat["reclaim_strength_range"] = feat["reclaim_strength_range"].clip(0, 2.0)
    feat["range_age"] = feat["range_age"].clip(0, 5.0)
    feat["zone_sl_dist"] = feat["zone_sl_dist"].clip(0, 0.10)
    feat["zone_tp_dist"] = feat["zone_tp_dist"].clip(0, 0.15)
    return feat, actions, signal_map_shifted


# ============================================================================
# DATASET
# ============================================================================

class SFPDataset(Dataset):
    def __init__(self, scaled_data, actions, quality, mfe, sl_labels,
                 ttp=None, asset_ids=None, tf_ids=None, mae=None, window=30):
        self.data = scaled_data
        self.window = window
        valid_start = window - 1
        self.indices = np.array([i for i in range(valid_start, len(actions)) if actions[i] != 0])
        self.actions = actions
        self.quality = quality
        self.mfe = mfe
        self.sl_labels = sl_labels
        self.ttp = ttp if ttp is not None else np.zeros(len(actions), dtype=np.float32)
        self.asset_ids = asset_ids if asset_ids is not None else np.zeros(len(actions), dtype=np.int64)
        self.tf_ids = tf_ids if tf_ids is not None else np.zeros(len(actions), dtype=np.int64)
        self.mae = mae if mae is not None else np.zeros(len(actions), dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri = self.indices[idx]
        x = self.data[ri - self.window + 1 : ri + 1]
        return (
            torch.FloatTensor(x),
            torch.LongTensor([self.actions[ri]]).squeeze(),
            torch.FloatTensor([self.quality[ri]]).squeeze(),
            torch.FloatTensor([self.mfe[ri]]).squeeze(),
            torch.FloatTensor([self.sl_labels[ri]]).squeeze(),
            torch.FloatTensor([self.ttp[ri]]).squeeze(),
            torch.LongTensor([self.asset_ids[ri]]).squeeze(),
            torch.LongTensor([self.tf_ids[ri]]).squeeze(),
            torch.FloatTensor([self.mae[ri]]).squeeze(),
        )


# ============================================================================
# MODEL
# ============================================================================

class ConditioningModule(nn.Module):
    def __init__(self, n_assets=6, n_tfs=4, asset_dim=8, tf_dim=4, n_heads=4):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_dim)
        self.tf_emb = nn.Embedding(n_tfs, tf_dim)
        self.direction_emb = nn.Embedding(3, 4)
        self.film = nn.Sequential(nn.Linear(asset_dim + tf_dim + 4, 16), nn.ReLU(), nn.Linear(16, n_heads * 2))
        self.n_heads = n_heads

    def forward(self, asset_ids, tf_ids, direction_ids=None):
        a = self.asset_emb(asset_ids)
        t = self.tf_emb(tf_ids)
        if direction_ids is None:
            direction_ids = torch.zeros_like(asset_ids)
        d = self.direction_emb(direction_ids)
        h = self.film(torch.cat([a, t, d], dim=-1))
        raw_scale = h[:, :self.n_heads]
        raw_bias = h[:, self.n_heads:]
        scale = 0.7 + 0.6 * torch.sigmoid(raw_scale)
        bias = 0.1 * torch.tanh(raw_bias)
        return scale, bias


class LiqRangeSFPClassifier(nn.Module):
    def __init__(self, n_features=33, window=30, hidden=32):
        super().__init__()
        self.bar_proj = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.pos_embed = nn.Parameter(torch.randn(1, window, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads=1, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.ffn_norm = nn.LayerNorm(hidden)
        self.attn2 = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True, dropout=0.1)
        self.attn2_norm = nn.LayerNorm(hidden)
        self.ffn2 = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.ffn2_norm = nn.LayerNorm(hidden)
        self.cls_head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
        self.tp1_head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
        self.tp2_head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
        self.ttp_head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
        self.sl_pred_head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
        self.conditioning = ConditioningModule(n_assets=6, n_tfs=4, n_heads=5)

    def forward(self, x, asset_ids=None, tf_ids=None, direction_ids=None):
        B, seq_len = x.shape[0], x.shape[1]
        h = self.bar_proj(x)
        h = h + self.pos_embed[:, :seq_len, :]
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))
        attn_out2, _ = self.attn2(h, h, h)
        h = self.attn2_norm(h + attn_out2)
        h = self.ffn2_norm(h + self.ffn2(h))
        pooled = h.mean(dim=1)
        last = h[:, -1, :]
        mx = h.max(dim=1).values
        combined = torch.cat([pooled, last, mx], dim=-1)
        cls_logit = self.cls_head(combined)
        tp1 = F.softplus(self.tp1_head(combined))
        tp2 = F.softplus(self.tp2_head(combined))
        ttp = torch.sigmoid(self.ttp_head(combined))
        sl_pred = F.softplus(self.sl_pred_head(combined.detach()))
        if asset_ids is not None and tf_ids is not None:
            scale, bias = self.conditioning(asset_ids, tf_ids, direction_ids)
            cls_logit = cls_logit * scale[:, 0:1] + bias[:, 0:1]
            tp1 = tp1 * scale[:, 1:2] + bias[:, 1:2]
            tp2 = tp2 * scale[:, 2:3] + bias[:, 2:3]
            ttp = ttp * scale[:, 3:4] + bias[:, 3:4]
            sl_pred = sl_pred * scale[:, 4:5] + bias[:, 4:5]
        return torch.cat([cls_logit, tp1, tp2, ttp, sl_pred], dim=-1)


# ============================================================================
# DATA LOADING
# ============================================================================

def _process_one_tf(item):
    asset_name, prefix, asset_id, tf, tf_key, tf_hours = item
    data_file = f"data/{prefix}_{tf}.csv"
    if not os.path.exists(data_file):
        return None
    df = pd.read_csv(data_file).reset_index(drop=True)
    if "timestamp" in df.columns:
        df = df[df["timestamp"] >= TRAIN_START].reset_index(drop=True)
    actions, quality, mfe, sl_labels, ttp_labels, swept_levels, signal_map, mae_labels = generate_labels(
        df["High"].values, df["Low"].values, df["Close"].values, df["Open"].values,
        volumes=df["Volume"].values if "Volume" in df.columns else None, tf_key=tf_key)
    feat, actions, signal_map_shifted = build_features(df, actions, signal_map, tf_hours, asset_id=asset_id)
    drop_n = 30
    quality = quality[drop_n:]; mfe = mfe[drop_n:]; sl_labels = sl_labels[drop_n:]
    ttp_labels = ttp_labels[drop_n:]; mae_labels = mae_labels[drop_n:]
    feat_values = feat.values.astype(np.float32)
    signal_mask = actions != 0
    total_signals = int(np.sum(signal_mask))
    if total_signals == 0:
        return None
    n_profitable = int(np.sum(quality[signal_mask] == 1))
    TF_ID_MAP = {"15m": 0, "1h": 1, "4h": 2}
    ASSET_ID_MAP = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
    n_bars = len(feat_values)
    asset_id_arr = np.full(n_bars, ASSET_ID_MAP.get(asset_id, 0), dtype=np.int64)
    tf_id_arr = np.full(n_bars, TF_ID_MAP.get(tf_key, 0), dtype=np.int64)
    split_idx = int(n_bars * 0.8)
    return {
        "tf_key": tf_key, "label": f"{asset_name}/{tf_key}",
        "n_bars": n_bars, "n_signals": total_signals, "n_profitable": n_profitable,
        "train_feat": feat_values[:split_idx], "train_actions": actions[:split_idx],
        "train_quality": quality[:split_idx], "train_mfe": mfe[:split_idx],
        "train_sl": sl_labels[:split_idx], "train_ttp": ttp_labels[:split_idx],
        "train_mae": mae_labels[:split_idx], "train_asset_ids": asset_id_arr[:split_idx],
        "train_tf_ids": tf_id_arr[:split_idx],
        "test_feat": feat_values[split_idx:], "test_actions": actions[split_idx:],
        "test_quality": quality[split_idx:], "test_mfe": mfe[split_idx:],
        "test_sl": sl_labels[split_idx:], "test_ttp": ttp_labels[split_idx:],
        "test_mae": mae_labels[split_idx:], "test_asset_ids": asset_id_arr[split_idx:],
        "test_tf_ids": tf_id_arr[split_idx:],
    }


def load_data_set():
    from multiprocessing import Pool
    work_items = []
    for asset_name in SELECTED_ASSETS:
        cfg = ASSETS[asset_name]
        for tf in TIMEFRAMES:
            work_items.append((asset_name, cfg["prefix"], cfg["asset_id"], tf, TF_KEYS[tf], TF_HOURS[tf]))
    print(f"\nProcessing {len(work_items)} asset/TF combos...")
    # Use sequential processing on Colab (avoids multiprocessing issues)
    results = [_process_one_tf(item) for item in work_items]

    tf_groups = {}
    keys = ["train_feat", "train_actions", "train_quality", "train_mfe", "train_sl",
            "train_ttp", "train_mae", "train_asset_ids", "train_tf_ids",
            "test_feat", "test_actions", "test_quality", "test_mfe", "test_sl",
            "test_ttp", "test_mae", "test_asset_ids", "test_tf_ids"]
    for result in results:
        if result is None:
            continue
        label = result["label"]
        print(f"  {label}: {result['n_bars']} bars, {result['n_signals']} signals, "
              f"{result['n_profitable']} profitable ({result['n_profitable']/result['n_signals']*100:.0f}%)")
        tk = result["tf_key"]
        if tk not in tf_groups:
            tf_groups[tk] = {k: [] for k in keys}
        g = tf_groups[tk]
        for split in ["train", "test"]:
            g[f"{split}_feat"].append(result[f"{split}_feat"])
            g[f"{split}_actions"].append(result[f"{split}_actions"])
            for k in ["quality", "mfe", "sl", "ttp", "mae", "asset_ids", "tf_ids"]:
                g[f"{split}_{k}"].append(result[f"{split}_{k}"])
    if not tf_groups:
        print("ERROR: No training data!"); sys.exit(1)

    all_train = np.concatenate([np.concatenate(g["train_feat"]) for g in tf_groups.values()])
    n_features = all_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(all_train)
    joblib.dump(scaler, "liq_range_sfp_scaler.joblib")
    print(f"\nScaler fit on {len(all_train)} train bars")

    train_loaders = {}; test_loaders = {}
    for tk, g in tf_groups.items():
        window = WINDOW_BY_TF.get(tk, 30)
        train_feat = scaler.transform(np.concatenate(g["train_feat"]))
        test_feat = scaler.transform(np.concatenate(g["test_feat"]))
        train_set = SFPDataset(train_feat, np.concatenate(g["train_actions"]),
                               np.concatenate(g["train_quality"]), np.concatenate(g["train_mfe"]),
                               np.concatenate(g["train_sl"]), ttp=np.concatenate(g["train_ttp"]),
                               asset_ids=np.concatenate(g["train_asset_ids"]),
                               tf_ids=np.concatenate(g["train_tf_ids"]),
                               mae=np.concatenate(g["train_mae"]), window=window)
        test_set = SFPDataset(test_feat, np.concatenate(g["test_actions"]),
                              np.concatenate(g["test_quality"]), np.concatenate(g["test_mfe"]),
                              np.concatenate(g["test_sl"]), ttp=np.concatenate(g["test_ttp"]),
                              asset_ids=np.concatenate(g["test_asset_ids"]),
                              tf_ids=np.concatenate(g["test_tf_ids"]),
                              mae=np.concatenate(g["test_mae"]), window=window)
        print(f"  {tk} (window={window}): {len(train_set)} train, {len(test_set)} test signals")
        train_loaders[tk] = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
        test_loaders[tk] = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    return train_loaders, test_loaders, n_features


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def quantile_loss(pred, target, tau):
    err = target - pred
    return torch.max(tau * err, (tau - 1) * err).mean()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, test_loaders):
    model.eval()
    all_cls_prob = []; all_tp1_pred = []; all_tp2_pred = []; all_sl_pred = []
    all_quality = []; all_mfe = []; all_sl = []
    total_loss = 0; n_batches = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
        for loader in test_loaders.values():
            for x, direction, q, mfe, sl, ttp, asset_id, tf_id, mae in loader:
                x = x.to(device); q_t = q.to(device).float()
                asset_id = asset_id.to(device); tf_id = tf_id.to(device)
                direction = direction.to(device)
                out = model(x, asset_ids=asset_id, tf_ids=tf_id, direction_ids=direction)
                cls_loss = nn.functional.binary_cross_entropy_with_logits(out[:, 0], q_t)
                total_loss += cls_loss.item(); n_batches += 1
                all_cls_prob.append(torch.sigmoid(out[:, 0]).cpu())
                all_tp1_pred.append(out[:, 1].cpu())
                all_tp2_pred.append(out[:, 2].cpu())
                all_sl_pred.append(out[:, 4].cpu())
                all_quality.append(q.cpu()); all_mfe.append(mfe.cpu()); all_sl.append(sl.cpu())
    cls_prob = torch.cat(all_cls_prob); tp1_pred = torch.cat(all_tp1_pred)
    tp2_pred = torch.cat(all_tp2_pred); sl_pred = torch.cat(all_sl_pred)
    quality = torch.cat(all_quality); mfe = torch.cat(all_mfe); sl = torch.cat(all_sl)

    results = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        take = cls_prob > thresh; n_take = take.sum().item()
        if n_take == 0:
            results[thresh] = {"n": 0, "wr": 0, "tp1_wr": 0, "tp2_wr": 0, "ev1": 0, "ev2": 0}; continue
        q_taken = quality[take]; wr = (q_taken == 1).float().mean().item() * 100
        mfe_taken = mfe[take]; sl_taken = sl[take]
        tp1_taken = tp1_pred[take]; tp2_taken = tp2_pred[take]
        tp1_wr = (mfe_taken >= tp1_taken).float().mean().item() * 100
        tp2_wr = (mfe_taken >= tp2_taken).float().mean().item() * 100
        avg_tp1 = tp1_taken.mean().item() * 100; avg_tp2 = tp2_taken.mean().item() * 100
        avg_sl_taken = sl_taken.mean().item() * 100
        ev1 = (tp1_wr / 100) * avg_tp1 - (1 - tp1_wr / 100) * avg_sl_taken
        ev2 = (tp2_wr / 100) * avg_tp2 - (1 - tp2_wr / 100) * avg_sl_taken
        results[thresh] = {"n": n_take, "wr": wr, "tp1_wr": tp1_wr, "tp2_wr": tp2_wr,
                           "avg_tp1": avg_tp1, "avg_tp2": avg_tp2, "avg_sl": avg_sl_taken,
                           "ev1": ev1, "ev2": ev2}
    return total_loss / max(n_batches, 1), results


# ============================================================================
# TRAINING (with AMP + torch.compile)
# ============================================================================

def train():
    t0 = time.time()
    train_loaders, test_loaders, n_features = load_data_set()
    WINDOW = max(WINDOW_BY_TF.values())
    model = LiqRangeSFPClassifier(n_features=n_features, window=WINDOW, hidden=48).to(device)

    if RESUME and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        print(f"Resumed from {MODEL_FILE}")

    # torch.compile for PyTorch 2.x speedup
    if hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), using eager mode")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | Device: {device} | AMP: {USE_AMP}")

    # Class weight
    all_q = []
    for loader in train_loaders.values():
        for x, direction, q, mfe, sl, ttp, asset_id, tf_id, mae in loader:
            all_q.append(q)
    all_q = torch.cat(all_q)
    n_pos = (all_q == 1).sum().item(); n_neg = (all_q == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Gate: wins={n_pos}, losses={n_neg}, pos_weight={pos_weight.item():.2f}")

    resume_lr = 1e-4 if RESUME else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=resume_lr, weight_decay=1e-3)
    total_batches = sum(len(loader) for loader in train_loaders.values())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=resume_lr, epochs=200, steps_per_epoch=total_batches)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    best_score = -float("inf"); best_loss = float("inf"); counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0; n_batches = 0
        for loader in train_loaders.values():
            for x, direction, q, mfe, sl, ttp, asset_id, tf_id, mae in loader:
                x = x.to(device); q_t = q.to(device).float()
                mfe_t = mfe.to(device).float(); sl_t = sl.to(device).float()
                ttp_t = ttp.to(device).float(); mae_t = mae.to(device).float()
                asset_id = asset_id.to(device); tf_id = tf_id.to(device)
                direction = direction.to(device)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    out = model(x, asset_ids=asset_id, tf_ids=tf_id, direction_ids=direction)
                    cls_logit = out[:, 0]; tp1_pred = out[:, 1]; tp2_pred = out[:, 2]
                    ttp_pred = out[:, 3]; sl_pred = out[:, 4]
                    cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_logit, q_t, pos_weight=pos_weight)
                    profitable_mask = q_t > 0.5
                    if profitable_mask.any():
                        mfe_prof = mfe_t[profitable_mask]
                        reg_loss = quantile_loss(tp1_pred[profitable_mask], mfe_prof, tau=0.15) + \
                                   quantile_loss(tp2_pred[profitable_mask], mfe_prof, tau=0.40)
                        ttp_loss = nn.functional.smooth_l1_loss(ttp_pred[profitable_mask], ttp_t[profitable_mask])
                    else:
                        reg_loss = torch.tensor(0.0, device=device)
                        ttp_loss = torch.tensor(0.0, device=device)
                    sl_loss = quantile_loss(sl_pred, mae_t, tau=0.85)
                    loss = cls_loss + reg_loss + 0.5 * ttp_loss + 0.2 * sl_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item(); n_batches += 1

        test_loss, results = evaluate(model, test_loaders)
        avg_train = train_loss / max(n_batches, 1)
        r50 = results.get(0.5, {"n": 0, "tp1_wr": 0, "ev1": 0})
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d} [{elapsed:.0f}s] | Loss: {avg_train:.4f}/{test_loss:.4f} | "
              f"n={r50['n']} TP1: {r50['tp1_wr']:.0f}%WR EV={r50['ev1']:+.2f}%")

        if (epoch + 1) % 10 == 0:
            for thresh in sorted(results.keys()):
                r = results[thresh]
                if r["n"] == 0:
                    continue
                print(f"    P>{thresh}: {r['n']} trades | WR: {r['wr']:.0f}% | "
                      f"TP1: {r['tp1_wr']:.0f}%WR TP={r.get('avg_tp1',0):.2f}% EV={r['ev1']:+.3f}% | "
                      f"TP2: {r['tp2_wr']:.0f}%WR EV={r['ev2']:+.3f}% | SL={r.get('avg_sl',0):.2f}%")

        score = 0.0
        for thresh, weight in [(0.4, 1.0), (0.5, 2.0), (0.6, 3.0)]:
            r = results.get(thresh, {"n": 0, "ev1": 0})
            if r["n"] >= 10 and r["ev1"] > 0:
                score += weight * r["ev1"] * min(r["n"], 200)
        if score > best_score:
            best_score = score
            # Save uncompiled model state dict
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), MODEL_FILE)
            print(f"  -> Saved best (score: {score:.1f})")
        if test_loss < best_loss:
            best_loss = test_loss; counter = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"\nEarly stopping at epoch {epoch+1}"); break

    # Final eval
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    _, results = evaluate(raw_model, test_loaders)
    print(f"\n{'='*60}\nBest model — MFE regression results\n{'='*60}")
    for thresh in sorted(results.keys()):
        r = results[thresh]
        if r["n"] == 0:
            continue
        print(f"  P>{thresh}: {r['n']} trades | WR: {r['wr']:.0f}% | "
              f"TP1: {r['tp1_wr']:.0f}%WR TP={r.get('avg_tp1',0):.2f}% EV={r['ev1']:+.3f}% | "
              f"TP2: {r['tp2_wr']:.0f}%WR EV={r['ev2']:+.3f}% | SL={r.get('avg_sl',0):.2f}%")
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Scaler saved: liq_range_sfp_scaler.joblib")


if __name__ == "__main__":
    train()
