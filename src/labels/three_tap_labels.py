"""Three-Tap Strategy label generator.

Detects: Range → Deviation + MSS → FVG Demand Zone → Retest Entry.
Two layers: wide range (macro TP target) and small range (nested, trade setup).
Also detects breakout retests when wide range breaks.
"""

import numpy as np
from dataclasses import dataclass, field
from src.labels.sfp_labels import detect_swings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Range:
    high: float
    low: float
    start: int          # first bar of the range
    confirmed: int      # bar when range is confirmed (min_bars reached)
    touches_high: int
    touches_low: int
    broken: bool = False
    broken_dir: int = 0  # 1=bullish breakout, 2=bearish breakout
    broken_bar: int = -1


@dataclass
class Deviation:
    bar_idx: int
    direction: int      # 1=bullish (sweep below), 2=bearish (sweep above)
    range_ref: Range = field(repr=False)
    wick_extreme: float  # lowest low (bull) or highest high (bear)


@dataclass
class MSS:
    bar_idx: int
    deviation: Deviation = field(repr=False)
    direction: int       # 1=bullish MSS, 2=bearish MSS


@dataclass
class FVG:
    bar_idx: int         # middle candle index
    top: float
    bottom: float
    direction: int       # 1=bullish FVG, 2=bearish FVG


@dataclass
class DemandZone:
    top: float
    bottom: float
    direction: int       # 1=long demand, 2=short supply
    mss_bar: int
    deviation_wick: float
    tp_target: float     # wide range opposite boundary or measured move
    expiry: int          # max bar to wait for retest


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------

def compute_atr(highs, lows, closes, period=14):
    """Wilder's ATR."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Range detection
# ---------------------------------------------------------------------------

def detect_ranges(
    highs, lows, closes,
    atr,
    n_swing,
    min_bars,
    max_bars,
    min_atr_mult,
    max_atr_mult,
    tolerance=0.01,
    touch_tolerance=0.003,
    min_touches=2,
):
    """Detect consolidation ranges.

    Returns active_ranges: list[list[Range]] — active ranges at each bar.
    """
    n = len(highs)
    swing_highs, swing_lows = detect_swings(highs, lows, n_swing)

    # Collect confirmed swing levels with their bar indices
    sh_levels = []  # (bar_idx, price)
    sl_levels = []
    for i in range(n):
        if swing_highs[i]:
            sh_levels.append((i, highs[i]))
        if swing_lows[i]:
            sl_levels.append((i, lows[i]))

    ranges: list[Range] = []

    # Try to form ranges from pairs of swing high + swing low
    # Only consider pairs within max_bars distance to avoid O(N^2) blowup
    for sl_idx, sl_price in sl_levels:
        for sh_idx, sh_price in sh_levels:
            if abs(sh_idx - sl_idx) > max_bars:
                continue
            if sh_price <= sl_price:
                continue

            range_height = sh_price - sl_price
            start = min(sl_idx, sh_idx)

            # Check ATR at the start of the range
            local_atr = atr[start] if atr[start] > 0 else atr[max(0, start - 1)]
            if local_atr <= 0:
                continue
            atr_mult = range_height / local_atr
            if atr_mult < min_atr_mult or atr_mult > max_atr_mult:
                continue

            # Count bars where price stays inside (with tolerance)
            upper_bound = sh_price * (1 + tolerance)
            lower_bound = sl_price * (1 - tolerance)

            # Find how far the range extends
            bars_inside = 0
            end = start
            for j in range(start, min(start + max_bars, n)):
                if highs[j] <= upper_bound and lows[j] >= lower_bound:
                    bars_inside += 1
                    end = j
                else:
                    # Allow occasional wick outside if body stays inside
                    body_high = max(closes[j], highs[j] * 0.5 + closes[j] * 0.5)
                    body_low = min(closes[j], lows[j] * 0.5 + closes[j] * 0.5)
                    if body_high <= upper_bound and body_low >= lower_bound:
                        bars_inside += 1
                        end = j
                    else:
                        break

            if bars_inside < min_bars:
                continue

            # Count touches of high and low zones
            high_zone_top = sh_price * (1 + touch_tolerance)
            high_zone_bot = sh_price * (1 - touch_tolerance)
            low_zone_top = sl_price * (1 + touch_tolerance)
            low_zone_bot = sl_price * (1 - touch_tolerance)

            touches_h = 0
            touches_l = 0
            last_touch_h = -10
            last_touch_l = -10
            for j in range(start, end + 1):
                if highs[j] >= high_zone_bot and highs[j] <= high_zone_top:
                    if j - last_touch_h >= 3:  # min 3 bars between touches
                        touches_h += 1
                        last_touch_h = j
                if lows[j] >= low_zone_bot and lows[j] <= low_zone_top:
                    if j - last_touch_l >= 3:
                        touches_l += 1
                        last_touch_l = j

            if touches_h < min_touches or touches_l < min_touches:
                continue

            r = Range(
                high=sh_price,
                low=sl_price,
                start=start,
                confirmed=start + min_bars,
                touches_high=touches_h,
                touches_low=touches_l,
            )

            # Check for duplicate/overlapping ranges
            is_dup = False
            for existing in ranges:
                high_overlap = abs(existing.high - r.high) / (r.high + 1e-8) < 0.01
                low_overlap = abs(existing.low - r.low) / (r.low + 1e-8) < 0.01
                if high_overlap and low_overlap:
                    # Keep the one with more touches
                    if r.touches_high + r.touches_low > existing.touches_high + existing.touches_low:
                        ranges.remove(existing)
                    else:
                        is_dup = True
                    break
            if not is_dup:
                ranges.append(r)

    # Build per-bar active ranges list
    # Only iterate over each range's active window instead of all bars
    active_ranges = [[] for _ in range(n)]
    for r in ranges:
        end_bar = min(r.start + max_bars, n)
        for i in range(r.confirmed, end_bar):
            if not r.broken:
                active_ranges[i].append(r)

    return active_ranges, ranges


def detect_nested_ranges(
    highs, lows, closes, atr,
    wide_active_ranges,
    n_swing=5, min_bars=15, max_bars=120,
    min_atr_mult=0.5, max_atr_mult=6.0,
):
    """Detect small ranges nested inside wide ranges.

    Must be fully contained within an active wide range.
    """
    small_active, small_all = detect_ranges(
        highs, lows, closes, atr,
        n_swing=n_swing,
        min_bars=min_bars,
        max_bars=max_bars,
        min_atr_mult=min_atr_mult,
        max_atr_mult=max_atr_mult,
        tolerance=0.01,
        touch_tolerance=0.005,
        min_touches=2,
    )

    # Filter: small range must be inside some wide range at its confirmed bar
    n = len(highs)
    filtered_active = [[] for _ in range(n)]
    filtered_all = []

    for sr in small_all:
        inside_wide = False
        for wr_list in wide_active_ranges[sr.confirmed: sr.confirmed + 1]:
            for wr in wr_list:
                if sr.high <= wr.high * (1 + 0.005) and sr.low >= wr.low * (1 - 0.005):
                    inside_wide = True
                    # Store the parent wide range for TP targeting
                    sr._wide_range = wr
                    break
            if inside_wide:
                break

        if inside_wide:
            filtered_all.append(sr)

    for sr in filtered_all:
        end_bar = min(sr.start + max_bars, n)
        for i in range(sr.confirmed, end_bar):
            if not sr.broken:
                filtered_active[i].append(sr)

    return filtered_active, filtered_all


# ---------------------------------------------------------------------------
# Deviation detection
# ---------------------------------------------------------------------------

def detect_deviation(
    highs, lows, closes, opens,
    active_ranges,
    min_dev_pct=0.001,
    max_dev_pct=0.03,
    allow_multi=False,
    min_gap_bars=20,
):
    """Detect deviations (liquidity sweeps) beyond range boundaries.

    If allow_multi=True, the same range can produce multiple deviations
    per direction as long as they are min_gap_bars apart.

    Returns list of Deviation objects.
    """
    n = len(highs)
    deviations = []
    # Track (range_id, direction) → last deviation bar index
    last_dev_bar: dict[tuple, int] = {}

    for i in range(n):
        for r in active_ranges[i]:
            range_id = id(r)

            # Bullish deviation: wick below range low
            if lows[i] < r.low:
                dev_pct = (r.low - lows[i]) / r.low
                if min_dev_pct <= dev_pct <= max_dev_pct:
                    key = (range_id, 1)
                    prev_bar = last_dev_bar.get(key, -999)
                    if allow_multi:
                        if i - prev_bar >= min_gap_bars:
                            deviations.append(Deviation(
                                bar_idx=i, direction=1,
                                range_ref=r, wick_extreme=lows[i],
                            ))
                            last_dev_bar[key] = i
                    else:
                        if key not in last_dev_bar:
                            deviations.append(Deviation(
                                bar_idx=i, direction=1,
                                range_ref=r, wick_extreme=lows[i],
                            ))
                            last_dev_bar[key] = i

            # Bearish deviation: wick above range high
            if highs[i] > r.high:
                dev_pct = (highs[i] - r.high) / r.high
                if min_dev_pct <= dev_pct <= max_dev_pct:
                    key = (range_id, 2)
                    prev_bar = last_dev_bar.get(key, -999)
                    if allow_multi:
                        if i - prev_bar >= min_gap_bars:
                            deviations.append(Deviation(
                                bar_idx=i, direction=2,
                                range_ref=r, wick_extreme=highs[i],
                            ))
                            last_dev_bar[key] = i
                    else:
                        if key not in last_dev_bar:
                            deviations.append(Deviation(
                                bar_idx=i, direction=2,
                                range_ref=r, wick_extreme=highs[i],
                            ))
                            last_dev_bar[key] = i

    return deviations


# ---------------------------------------------------------------------------
# MSS / BOS detection
# ---------------------------------------------------------------------------

def detect_mss(
    highs, lows, closes,
    deviations,
    n_mss=3,
    max_bars=10,
    mode="strict",
):
    """Detect Market Structure Shift after deviations.

    Modes:
      "strict": Close breaks above recent swing high (n=3) for bullish,
                below recent swing low for bearish. Original behavior.
      "soft":   Close reclaims back inside range with a strong body
                (body > 50% of candle range). Easier to trigger but still
                confirms buyer/seller response after the sweep.

    Returns list of MSS objects.
    """
    n = len(highs)
    mss_list = []

    if mode == "strict":
        swing_highs, swing_lows = detect_swings(highs, lows, n_mss)

        for dev in deviations:
            start = dev.bar_idx + 1
            end = min(dev.bar_idx + max_bars + 1, n)

            if dev.direction == 1:  # bullish deviation → bullish MSS
                recent_sh = None
                for j in range(dev.bar_idx, max(dev.bar_idx - 30, -1), -1):
                    if swing_highs[j]:
                        recent_sh = highs[j]
                        break
                if recent_sh is None:
                    lookback = highs[max(0, dev.bar_idx - 10): dev.bar_idx + 1]
                    if len(lookback) > 0:
                        recent_sh = np.max(lookback)
                    else:
                        continue

                for j in range(start, end):
                    if closes[j] > recent_sh:
                        mss_list.append(MSS(bar_idx=j, deviation=dev, direction=1))
                        break

            elif dev.direction == 2:  # bearish deviation → bearish MSS
                recent_sl = None
                for j in range(dev.bar_idx, max(dev.bar_idx - 30, -1), -1):
                    if swing_lows[j]:
                        recent_sl = lows[j]
                        break
                if recent_sl is None:
                    lookback = lows[max(0, dev.bar_idx - 10): dev.bar_idx + 1]
                    if len(lookback) > 0:
                        recent_sl = np.min(lookback)
                    else:
                        continue

                for j in range(start, end):
                    if closes[j] < recent_sl:
                        mss_list.append(MSS(bar_idx=j, deviation=dev, direction=2))
                        break

    elif mode == "soft":
        for dev in deviations:
            r = dev.range_ref
            start = dev.bar_idx + 1
            end = min(dev.bar_idx + max_bars + 1, n)

            for j in range(start, end):
                candle_range = highs[j] - lows[j]
                if candle_range <= 0:
                    continue
                body = abs(closes[j] - closes[j - 1] if j > 0 else closes[j] - lows[j])
                body_ratio = body / candle_range

                if dev.direction == 1:  # bullish: close reclaims above range low
                    if closes[j] > r.low and body_ratio > 0.4:
                        mss_list.append(MSS(bar_idx=j, deviation=dev, direction=1))
                        break
                elif dev.direction == 2:  # bearish: close reclaims below range high
                    if closes[j] < r.high and body_ratio > 0.4:
                        mss_list.append(MSS(bar_idx=j, deviation=dev, direction=2))
                        break

    return mss_list


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------

def detect_fvg(highs, lows, closes, min_gap_pct=0.001):
    """Detect Fair Value Gaps (3-candle imbalance).

    Bullish FVG: highs[i] < lows[i+2] — gap below impulsive up move.
    Bearish FVG: lows[i] > highs[i+2] — gap above impulsive down move.

    Returns list of FVG objects.
    """
    n = len(highs)
    fvgs = []

    for i in range(n - 2):
        # Bullish FVG
        if highs[i] < lows[i + 2]:
            gap = lows[i + 2] - highs[i]
            gap_pct = gap / (closes[i + 1] + 1e-8)
            if gap_pct >= min_gap_pct:
                fvgs.append(FVG(
                    bar_idx=i + 1,
                    top=lows[i + 2],
                    bottom=highs[i],
                    direction=1,
                ))

        # Bearish FVG
        if lows[i] > highs[i + 2]:
            gap = lows[i] - highs[i + 2]
            gap_pct = gap / (closes[i + 1] + 1e-8)
            if gap_pct >= min_gap_pct:
                fvgs.append(FVG(
                    bar_idx=i + 1,
                    top=lows[i],
                    bottom=highs[i + 2],
                    direction=2,
                ))

    return fvgs


# ---------------------------------------------------------------------------
# Demand zone construction
# ---------------------------------------------------------------------------

def build_demand_zones(mss_list, fvgs, wide_active_ranges, max_zone_atr=3.0, atr=None):
    """Build demand/supply zones from MSS + FVG confluence.

    If FVG exists between deviation and MSS: zone = FVG top → deviation wick.
    If no FVG (optional): zone = range boundary → deviation wick.

    Returns list of DemandZone objects.
    """
    zones = []

    for mss in mss_list:
        dev = mss.deviation
        dev_bar = dev.bar_idx
        mss_bar = mss.bar_idx
        r = dev.range_ref

        # Find FVGs that formed between deviation and MSS (inclusive range +2)
        matching_fvgs = []
        for fvg in fvgs:
            if fvg.bar_idx < dev_bar or fvg.bar_idx > mss_bar + 2:
                continue
            if fvg.direction != mss.direction:
                continue
            matching_fvgs.append(fvg)

        best_fvg = min(matching_fvgs, key=lambda f: abs(f.bar_idx - dev_bar)) if matching_fvgs else None

        # TP target: small range opposite boundary
        tp_target = 0.0
        if mss.direction == 1:  # bullish
            if best_fvg:
                zone_top = best_fvg.top
            else:
                zone_top = r.low  # fallback: range low as zone top
            zone_bottom = dev.wick_extreme
            tp_target = r.high
        else:  # bearish
            zone_top = dev.wick_extreme
            if best_fvg:
                zone_bottom = best_fvg.bottom
            else:
                zone_bottom = r.high  # fallback: range high as zone bottom
            tp_target = r.low

        # Validate zone size
        zone_height = zone_top - zone_bottom
        if zone_height <= 0:
            continue
        if atr is not None and atr[mss_bar] > 0:
            if zone_height / atr[mss_bar] > max_zone_atr:
                continue

        zone = DemandZone(
            top=zone_top,
            bottom=zone_bottom,
            direction=mss.direction,
            mss_bar=mss_bar,
            deviation_wick=dev.wick_extreme,
            tp_target=tp_target,
            expiry=mss_bar + 30,
        )
        # Attach setup metadata for feature engineering
        zone._range_high = r.high
        zone._range_low = r.low
        zone._range_touches = r.touches_high + r.touches_low
        zone._range_confirmed = r.confirmed
        zone._deviation_bar = dev.bar_idx
        zone._has_fvg = best_fvg is not None
        zone._mss_bar = mss_bar
        zones.append(zone)

    return zones


# ---------------------------------------------------------------------------
# Retest detection (Tap 3 — the trade)
# ---------------------------------------------------------------------------

def detect_retest(highs, lows, closes, demand_zones, max_wait=30, min_rr=1.5):
    """Detect when price retests a demand/supply zone.

    Only takes trades where structural R:R >= min_rr.
    Returns (actions, entry_levels, sl_levels, tp_levels, signal_zones) arrays.
    signal_zones maps signal bar index → DemandZone (for feature engineering).
    """
    n = len(highs)
    actions = np.zeros(n, dtype=np.int64)
    entry_levels = np.zeros(n, dtype=np.float64)
    sl_levels = np.zeros(n, dtype=np.float64)
    tp_levels = np.zeros(n, dtype=np.float64)
    signal_zones = {}  # bar_idx → DemandZone

    used_zones = set()

    for zone in demand_zones:
        zone_id = id(zone)
        if zone_id in used_zones:
            continue

        start = zone.mss_bar + 1
        end = min(zone.expiry + 1, n)

        for i in range(start, end):
            if actions[i] != 0:
                continue  # already has a signal

            if zone.direction == 1:  # bullish demand — price dips into zone
                if lows[i] <= zone.top:
                    entry = zone.top
                    sl = zone.bottom  # below deviation wick
                    tp = zone.tp_target
                    if entry <= sl or tp <= entry:
                        break
                    # R:R check
                    tp_dist = tp - entry
                    sl_dist = entry - sl
                    if sl_dist > 0 and tp_dist / sl_dist >= min_rr:
                        actions[i] = 1
                        entry_levels[i] = entry
                        sl_levels[i] = sl
                        tp_levels[i] = tp
                        signal_zones[i] = zone
                        used_zones.add(zone_id)
                    break  # zone used either way

            elif zone.direction == 2:  # bearish supply — price pokes into zone
                if highs[i] >= zone.bottom:
                    entry = zone.bottom
                    sl = zone.top  # above deviation wick
                    tp = zone.tp_target
                    if entry >= sl or tp >= entry:
                        break
                    tp_dist = entry - tp
                    sl_dist = sl - entry
                    if sl_dist > 0 and tp_dist / sl_dist >= min_rr:
                        actions[i] = 2
                        entry_levels[i] = entry
                        sl_levels[i] = sl
                        tp_levels[i] = tp
                        signal_zones[i] = zone
                        used_zones.add(zone_id)
                    break

    return actions, entry_levels, sl_levels, tp_levels, signal_zones


# ---------------------------------------------------------------------------
# Breakout retest detection
# ---------------------------------------------------------------------------

def detect_breakout_retest(
    highs, lows, closes,
    wide_ranges,
    fvgs,
    n_mss=3,
    max_wait=30,
):
    """Detect breakout of wide range + retest of broken level.

    When price breaks wide range AND MSS confirms in break direction,
    wait for retest of the broken level.

    Returns (actions, entry_levels, sl_levels, tp_levels) arrays.
    """
    n = len(highs)
    actions = np.zeros(n, dtype=np.int64)
    entry_levels = np.zeros(n, dtype=np.float64)
    sl_levels = np.zeros(n, dtype=np.float64)
    tp_levels = np.zeros(n, dtype=np.float64)

    swing_highs, swing_lows = detect_swings(highs, lows, n_mss)

    for r in wide_ranges:
        if r.broken:
            continue

        range_height = r.high - r.low

        # Check for bullish breakout (close above range high)
        for i in range(r.confirmed, min(r.start + 500, n)):
            if closes[i] > r.high * 1.001:
                # Confirm MSS above
                recent_sh = None
                for j in range(i, max(i - 20, -1), -1):
                    if swing_highs[j]:
                        recent_sh = highs[j]
                        break
                if recent_sh is None or closes[i] <= recent_sh:
                    continue

                r.broken = True
                r.broken_dir = 1
                r.broken_bar = i

                # Wait for retest of range high (now support)
                retest_zone_top = r.high * 1.005
                retest_zone_bot = r.high * 0.995
                tp = r.high + range_height  # measured move

                for j in range(i + 1, min(i + max_wait + 1, n)):
                    if actions[j] != 0:
                        continue
                    if lows[j] <= retest_zone_top and lows[j] >= retest_zone_bot * 0.99:
                        entry = r.high
                        sl_price = r.high - range_height * 0.3  # SL below broken level
                        if tp > entry and entry > sl_price:
                            actions[j] = 1
                            entry_levels[j] = entry
                            sl_levels[j] = sl_price
                            tp_levels[j] = tp
                        break
                break

            # Check for bearish breakout (close below range low)
            if closes[i] < r.low * 0.999:
                recent_sl = None
                for j in range(i, max(i - 20, -1), -1):
                    if swing_lows[j]:
                        recent_sl = lows[j]
                        break
                if recent_sl is None or closes[i] >= recent_sl:
                    continue

                r.broken = True
                r.broken_dir = 2
                r.broken_bar = i

                retest_zone_top = r.low * 1.005
                retest_zone_bot = r.low * 0.995
                tp = r.low - range_height  # measured move down

                for j in range(i + 1, min(i + max_wait + 1, n)):
                    if actions[j] != 0:
                        continue
                    if highs[j] >= retest_zone_bot and highs[j] <= retest_zone_top * 1.01:
                        entry = r.low
                        sl_price = r.low + range_height * 0.3
                        if tp < entry and entry < sl_price:
                            actions[j] = 2
                            entry_levels[j] = entry
                            sl_levels[j] = sl_price
                            tp_levels[j] = tp
                        break
                break

    return actions, entry_levels, sl_levels, tp_levels


# ---------------------------------------------------------------------------
# TP / SL label computation
# ---------------------------------------------------------------------------

def compute_tp_sl_labels(
    highs, lows, closes,
    actions, entry_levels, sl_levels, tp_levels,
    horizon=18,
):
    """Compute quality and actual TP/SL labels.

    Uses structural entry/SL/TP from the three-tap setup.
    Quality = 1 if TP hit before SL within horizon, else 0.
    """
    n = len(actions)
    quality = np.zeros(n, dtype=np.int64)
    tp_labels = np.zeros(n, dtype=np.float32)
    sl_labels = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if actions[i] == 0:
            continue
        if i + horizon >= n:
            actions[i] = 0
            continue

        entry = entry_levels[i]
        sl = sl_levels[i]
        tp = tp_levels[i]
        if entry <= 0 or sl <= 0:
            actions[i] = 0
            continue

        future_highs = highs[i + 1: i + 1 + horizon]
        future_lows = lows[i + 1: i + 1 + horizon]

        if actions[i] == 1:  # long
            tp_pct = (tp - entry) / entry
            sl_pct = (entry - sl) / entry

            # Check outcome bar by bar
            tp_hit = False
            sl_hit = False
            for j in range(len(future_highs)):
                if future_lows[j] <= sl:
                    sl_hit = True
                    break
                if future_highs[j] >= tp:
                    tp_hit = True
                    break

            if tp_hit:
                quality[i] = 1
            else:
                quality[i] = 0

        else:  # short
            tp_pct = (entry - tp) / entry
            sl_pct = (sl - entry) / entry

            tp_hit = False
            sl_hit = False
            for j in range(len(future_lows)):
                if future_highs[j] >= sl:
                    sl_hit = True
                    break
                if future_lows[j] <= tp:
                    tp_hit = True
                    break

            if tp_hit:
                quality[i] = 1
            else:
                quality[i] = 0

        tp_labels[i] = np.clip(tp_pct, 0.001, 0.10)
        sl_labels[i] = np.clip(sl_pct, 0.001, 0.10)

    return quality, tp_labels, sl_labels


# ---------------------------------------------------------------------------
# Range detection on 4h (called once, reused across TFs)
# ---------------------------------------------------------------------------

def detect_ranges_4h(highs_4h, lows_4h, closes_4h, timestamps_4h):
    """Detect wide + small ranges on 4h data.

    Returns list of Range objects with confirmed_time set for cross-TF mapping.
    Each Range also carries ._wide_range for nested ranges.
    """
    atr = compute_atr(highs_4h, lows_4h, closes_4h, period=14)

    print("  [Three-Tap] Detecting wide ranges on 4h (n=20, min_bars=60)...")
    wide_active, wide_all = detect_ranges(
        highs_4h, lows_4h, closes_4h, atr,
        n_swing=20, min_bars=60, max_bars=500,
        min_atr_mult=2.0, max_atr_mult=15.0,
        tolerance=0.01, touch_tolerance=0.005, min_touches=2,
    )
    print(f"    Found {len(wide_all)} wide ranges")

    # Stamp confirmation time on each range
    for r in wide_all:
        if r.confirmed < len(timestamps_4h):
            r._confirmed_time = timestamps_4h[r.confirmed]
        else:
            r._confirmed_time = timestamps_4h[-1]

    print("  [Three-Tap] Detecting small ranges on 4h (n=5, min_bars=15)...")
    small_active, small_all = detect_nested_ranges(
        highs_4h, lows_4h, closes_4h, atr, wide_active,
    )
    print(f"    Found {len(small_all)} small ranges (nested inside wide)")

    for r in small_all:
        if r.confirmed < len(timestamps_4h):
            r._confirmed_time = timestamps_4h[r.confirmed]
        else:
            r._confirmed_time = timestamps_4h[-1]

    return small_all, wide_all


def build_active_ranges_for_tf(ranges, timestamps_tf):
    """Map 4h-detected ranges to per-bar active list for any timeframe.

    A range is active at bar i if timestamps_tf[i] >= range._confirmed_time.
    """
    n = len(timestamps_tf)
    active = [[] for _ in range(n)]

    import bisect
    ts_list = list(timestamps_tf)

    for r in ranges:
        conf_time = getattr(r, '_confirmed_time', None)
        if conf_time is None or r.broken:
            continue
        # Binary search for the first bar >= conf_time
        start_i = bisect.bisect_left(ts_list, conf_time)
        for i in range(start_i, n):
            active[i].append(r)

    return active


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Horizon per TF: swing trades need more bars on lower TFs
HORIZON_BY_TF = {
    "15m": 240,  # 60 hours
    "1h": 60,    # 60 hours
    "4h": 60,    # 240 hours
}

MSS_MAX_BARS = {"15m": 40, "1h": 20, "4h": 10}

# Per-TF range detection parameters
RANGE_PARAMS = {
    "15m": {
        "wide": {"n_swing": 10, "min_bars": 200, "max_bars": 800,
                 "min_atr_mult": 2.0, "max_atr_mult": 10.0},
        "small": {"n_swing": 3, "min_bars": 40, "max_bars": 250},
    },
    "1h": {
        "wide": {"n_swing": 15, "min_bars": 80, "max_bars": 500,
                 "min_atr_mult": 2.0, "max_atr_mult": 12.0},
        "small": {"n_swing": 5, "min_bars": 20, "max_bars": 150},
    },
    "4h": {
        "wide": {"n_swing": 20, "min_bars": 60, "max_bars": 500,
                 "min_atr_mult": 2.0, "max_atr_mult": 15.0},
        "small": {"n_swing": 5, "min_bars": 15, "max_bars": 120},
    },
}


def generate_labels(highs, lows, closes, opens, volumes=None,
                    precomputed_ranges=None, timestamps=None, tf_key="4h",
                    require_mss=True, allow_multi_dev=False, mss_mode="strict"):
    """Top-level label generator for the Three-Tap strategy.

    Args:
        highs, lows, closes, opens: price arrays for the signal TF
        precomputed_ranges: (small_all, wide_all) from detect_ranges_4h().
            If None, detects ranges on the provided data directly (4h mode).
        timestamps: timestamp strings for each bar (needed for cross-TF mapping)
        tf_key: "15m", "1h", or "4h" — controls horizon and MSS window
        require_mss: if False, skip MSS confirmation (every deviation → zone)
        allow_multi_dev: if True, allow multiple deviations per range (min 20 bars apart)

    Returns:
        (actions, quality, tp_labels, sl_labels, entry_levels)
    """
    n = len(highs)
    atr = compute_atr(highs, lows, closes, period=14)
    horizon = HORIZON_BY_TF.get(tf_key, 60)
    mss_max = MSS_MAX_BARS.get(tf_key, 10)

    if precomputed_ranges is not None:
        small_all, wide_all = precomputed_ranges
        if timestamps is None:
            raise ValueError("timestamps required when using precomputed_ranges")
        small_active = build_active_ranges_for_tf(small_all, timestamps)
    else:
        # Per-TF range detection parameters
        rp = RANGE_PARAMS.get(tf_key, RANGE_PARAMS["4h"])
        wp = rp["wide"]
        sp = rp["small"]

        print(f"  [{tf_key}] Detecting wide ranges (n={wp['n_swing']}, min_bars={wp['min_bars']})...")
        wide_active, wide_all = detect_ranges(
            highs, lows, closes, atr,
            n_swing=wp["n_swing"], min_bars=wp["min_bars"], max_bars=wp["max_bars"],
            min_atr_mult=wp["min_atr_mult"], max_atr_mult=wp["max_atr_mult"],
            tolerance=0.01, touch_tolerance=0.005, min_touches=2,
        )
        print(f"    Found {len(wide_all)} wide ranges")

        print(f"  [{tf_key}] Detecting small ranges (n={sp['n_swing']}, min_bars={sp['min_bars']})...")
        small_active, small_all = detect_nested_ranges(
            highs, lows, closes, atr, wide_active,
            n_swing=sp["n_swing"], min_bars=sp["min_bars"], max_bars=sp["max_bars"],
        )
        print(f"    Found {len(small_all)} small ranges")

    print(f"  [{tf_key}] Detecting deviations (multi={allow_multi_dev})...")
    deviations = detect_deviation(
        highs, lows, closes, opens, small_active,
        allow_multi=allow_multi_dev, min_gap_bars=20,
    )
    print(f"    Found {len(deviations)} deviations")

    print(f"  [{tf_key}] Detecting FVGs...")
    fvgs = detect_fvg(highs, lows, closes, min_gap_pct=0.0005)
    print(f"    Found {len(fvgs)} FVGs")

    if require_mss:
        print(f"  [{tf_key}] Detecting MSS/BOS (mode={mss_mode}, max_bars={mss_max})...")
        mss_list = detect_mss(highs, lows, closes, deviations, n_mss=3, max_bars=mss_max, mode=mss_mode)
        print(f"    Found {len(mss_list)} MSS confirmations")
    else:
        print(f"  [{tf_key}] Skipping MSS — all deviations pass")
        mss_list = [
            MSS(bar_idx=dev.bar_idx, deviation=dev, direction=dev.direction)
            for dev in deviations
        ]
        print(f"    {len(mss_list)} pseudo-MSS from deviations")

    print(f"  [{tf_key}] Building demand zones...")
    zones = build_demand_zones(mss_list, fvgs, small_active, max_zone_atr=2.0, atr=atr)
    print(f"    Built {len(zones)} demand/supply zones")

    print(f"  [{tf_key}] Detecting retests (Tap 3)...")
    actions, entry_levels, sl_levels, tp_levels, signal_zones = detect_retest(
        highs, lows, closes, zones, max_wait=30, min_rr=1.2,
    )
    n_retest = int(np.sum(actions != 0))
    n_long = int(np.sum(actions == 1))
    n_short = int(np.sum(actions == 2))
    print(f"    Found {n_retest} retest entries ({n_long} long, {n_short} short)")

    # Compute TP/SL labels
    quality, tp_labels, sl_labels = compute_tp_sl_labels(
        highs, lows, closes, actions, entry_levels, sl_levels, tp_levels,
        horizon=horizon,
    )
    n_profitable = int(np.sum((actions != 0) & (quality == 1)))
    total_final = int(np.sum(actions != 0))
    if total_final > 0:
        print(f"  [{tf_key}] Funnel: {total_final} signals → {n_profitable} profitable ({n_profitable/total_final*100:.1f}%)")
    else:
        print(f"  [{tf_key}] No signals detected")

    return actions, quality, tp_labels, sl_labels, entry_levels, signal_zones
