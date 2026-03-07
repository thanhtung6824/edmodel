"""Range-SFP Strategy label generator.

Detects SFPs that sweep a range boundary (high or low) and reclaim back inside.
Combines zone-based range detection with SFP boundary filtering for higher win-rate signals.
"""

import numpy as np
from dataclasses import dataclass, field

from src.labels.sfp_labels import (
    detect_swings,
    build_swing_level_series,
    detect_sfp,
)
from src.labels.three_tap_labels import compute_atr


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceZone:
    bottom: float       # min price in cluster
    top: float          # max price in cluster
    level: float        # mean price
    count: int          # number of touches/rejections
    bars: list = field(default_factory=list)  # bar indices of touches


@dataclass
class ZoneRange:
    resistance: PriceZone   # upper zone
    support: PriceZone      # lower zone
    start: int              # first bar of consolidation
    confirmed: int          # start + min_bars
    bars_inside: int        # count of bars inside
    concentration: float    # bars_inside / total_bars
    # Backward-compat flat fields (set from zones at creation)
    high: float = 0.0
    low: float = 0.0
    touches_high: int = 0
    touches_low: int = 0
    broken: bool = False
    broken_dir: int = 0     # 1=bullish breakout, 2=bearish breakdown
    broken_bar: int = -1
    recaptured: bool = False
    recapture_bar: int = -1
    is_macro: bool = False


@dataclass
class RangeSFPSignal:
    bar_idx: int
    direction: int            # 1=long, 2=short
    swept_level: float
    range_ref: ZoneRange = field(repr=False)
    sweep_depth: float        # wick beyond boundary / range_height
    reclaim_strength: float   # (close - boundary) / range_height
    ms_alignment: float       # +1 aligned with trend, -1 against, 0 unclear
    ms_strength: float        # trend clarity (consecutive HH/HL or LH/LL count)


# ---------------------------------------------------------------------------
# Market structure detection
# ---------------------------------------------------------------------------

def detect_market_structure(highs, lows, n=10):
    """Detect market structure (HH/HL = bullish, LH/LL = bearish).

    Returns per-bar arrays:
        ms_direction: +1 bullish, -1 bearish, 0 unclear
        ms_strength: consecutive confirmations / 5 (normalized)
        last_structure_high: most recent swing high price
        last_structure_low: most recent swing low price
    """
    length = len(highs)
    ms_direction = np.zeros(length, dtype=np.float32)
    ms_strength = np.zeros(length, dtype=np.float32)
    last_structure_high = np.zeros(length, dtype=np.float64)
    last_structure_low = np.zeros(length, dtype=np.float64)

    swing_highs, swing_lows = detect_swings(highs, lows, n)

    # Track swing sequence
    prev_sh = 0.0
    prev_sl = 0.0
    bullish_count = 0
    bearish_count = 0
    current_sh = 0.0
    current_sl = 0.0

    for i in range(length):
        # Update swing levels when confirmed (at i, swing at i-n is confirmed)
        confirm_idx = i - n
        if confirm_idx >= 0:
            if swing_highs[confirm_idx]:
                new_sh = highs[confirm_idx]
                if prev_sh > 0:
                    if new_sh > prev_sh:  # Higher High
                        bullish_count += 1
                        bearish_count = max(bearish_count - 1, 0)
                    elif new_sh < prev_sh:  # Lower High
                        bearish_count += 1
                        bullish_count = max(bullish_count - 1, 0)
                prev_sh = new_sh
                current_sh = new_sh

            if swing_lows[confirm_idx]:
                new_sl = lows[confirm_idx]
                if prev_sl > 0:
                    if new_sl > prev_sl:  # Higher Low
                        bullish_count += 1
                        bearish_count = max(bearish_count - 1, 0)
                    elif new_sl < prev_sl:  # Lower Low
                        bearish_count += 1
                        bullish_count = max(bullish_count - 1, 0)
                prev_sl = new_sl
                current_sl = new_sl

        # Determine direction
        if bullish_count > bearish_count:
            ms_direction[i] = 1.0
            ms_strength[i] = min(bullish_count, 5) / 5.0
        elif bearish_count > bullish_count:
            ms_direction[i] = -1.0
            ms_strength[i] = min(bearish_count, 5) / 5.0
        else:
            ms_direction[i] = 0.0
            ms_strength[i] = 0.0

        last_structure_high[i] = current_sh if current_sh > 0 else highs[i]
        last_structure_low[i] = current_sl if current_sl > 0 else lows[i]

    return ms_direction, ms_strength, last_structure_high, last_structure_low


# ---------------------------------------------------------------------------
# Zone-based range detection (v2)
# ---------------------------------------------------------------------------

# Height filter: range height as % of price
RANGE_HEIGHT_PCT = {
    "15m": (0.003, 0.06),   # tighter max — quality filter
    "1h":  (0.005, 0.12),   # keep relaxed
    "4h":  (0.005, 0.12),   # keep relaxed
}

# Range detection parameters per TF
RANGE_SFP_PARAMS = {
    "15m": {"n_swing": 3, "min_bars": 30, "max_bars": 400, "min_zone_count": 4, "min_time_concentration": 0.80, "recapture_window": 10},
    "1h":  {"n_swing": 3, "min_bars": 15, "max_bars": 300, "min_zone_count": 3, "min_time_concentration": 0.70, "recapture_window": 5},
    "4h":  {"n_swing": 3, "min_bars": 20, "max_bars": 300, "min_zone_count": 3, "min_time_concentration": 0.65, "recapture_window": 3},
}


def cluster_levels(prices_with_bars, pct=0.015):
    """Cluster prices within pct of running cluster mean.

    Args:
        prices_with_bars: list of (price, bar_idx) tuples
        pct: maximum distance from cluster mean as fraction

    Returns:
        list of PriceZone
    """
    if not prices_with_bars:
        return []

    sorted_levels = sorted(prices_with_bars, key=lambda x: x[0])
    clusters = []
    current = [sorted_levels[0]]
    current_mean = sorted_levels[0][0]
    current_min = sorted_levels[0][0]

    for price, bar_idx in sorted_levels[1:]:
        # Check distance from mean AND cap total cluster width at 2*pct
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
        zones.append(PriceZone(
            bottom=min(prices),
            top=max(prices),
            level=sum(prices) / len(prices),
            count=len(prices),
            bars=bars,
        ))
    return zones


def _make_zone_range(rz, sz, earliest_bar, min_bars, min_height_pct, max_height_pct):
    """Helper to create a ZoneRange skeleton from resistance/support zones, or None.

    Only checks height constraints. Caller must fill bars_inside/concentration."""
    range_height = rz.level - sz.level
    mid_price = (rz.level + sz.level) / 2.0
    if mid_price <= 0 or range_height <= 0:
        return None
    height_pct = range_height / mid_price
    if height_pct < min_height_pct or height_pct > max_height_pct:
        return None

    confirmed_bar = earliest_bar + min_bars
    zr = ZoneRange(
        resistance=rz,
        support=sz,
        start=earliest_bar,
        confirmed=confirmed_bar,
        bars_inside=0,
        concentration=0.0,
        high=rz.top,
        low=sz.bottom,
        touches_high=rz.count,
        touches_low=sz.count,
    )
    zr._vol_compression = 1.0
    zr._height_pct = height_pct
    return zr


def _detect_ranges_in_window(
    highs, lows, closes, opens,
    w_start, w_end, global_offset,
    n_swing, min_bars, max_bars,
    min_height_pct, max_height_pct,
    min_time_concentration, min_zone_count, cluster_pct,
):
    """Detect ranges within a single temporal window.

    All bar indices in returned ranges are in global coordinates.
    """
    wh = highs[w_start:w_end]
    wl = lows[w_start:w_end]
    wc = closes[w_start:w_end]
    wo = opens[w_start:w_end]
    wn = len(wh)

    if wn < min_bars:
        return []

    swing_highs, swing_lows = detect_swings(wh, wl, n_swing)

    # Step 1: Collect price levels (local indices)
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

    # Step 2: Cluster
    rz_list = [z for z in cluster_levels(resistance_cands, pct=cluster_pct)
                if z.count >= min_zone_count]
    sz_list = [z for z in cluster_levels(support_cands, pct=cluster_pct)
                if z.count >= min_zone_count]

    # Step 3: Form ranges — causal confirmation
    # Check concentration over exactly min_bars from the earliest zone bar.
    # Confirmed at earliest + min_bars (no look-ahead: only uses bars up to that point).
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

            # Count bars inside during the confirmation window
            bi = 0
            for j in range(earliest, check_end):
                if wc[j] >= sz.bottom and wc[j] <= rz.top:
                    bi += 1
            conc = bi / min_bars
            if conc < min_time_concentration:
                continue

            # Convert to global bar indices
            g_earliest = earliest + w_start
            g_confirmed = check_end - 1 + w_start
            g_sz = PriceZone(sz.bottom, sz.top, sz.level, sz.count,
                             [b + w_start for b in sz.bars])
            g_rz = PriceZone(rz.bottom, rz.top, rz.level, rz.count,
                             [b + w_start for b in rz.bars])

            zr = ZoneRange(
                resistance=g_rz, support=g_sz,
                start=g_earliest, confirmed=g_confirmed,
                bars_inside=bi, concentration=conc,
                high=g_rz.top, low=g_sz.bottom,
                touches_high=g_rz.count, touches_low=g_sz.count,
            )
            zr._time_concentration = conc
            zr._vol_compression = 1.0
            zr._height_pct = h_pct

            # Dedup within window
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


def detect_ranges_v2(
    highs, lows, closes, opens,
    n_swing=3,
    min_bars=15,
    max_bars=250,
    min_height_pct=0.005,
    max_height_pct=0.08,
    min_time_concentration=0.80,
    min_zone_count=2,
    cluster_pct=0.015,
    recapture_window=5,
):
    """Detect ranges using zone-based boundaries in temporal windows.

    Uses overlapping windows to keep zones temporally relevant — a zone at $68k
    in 2021 won't contaminate range detection at $68k in 2024.

    Steps:
    1. Slide overlapping windows across the data
    2. In each window: collect swings + wick rejections, cluster, form ranges
    3. Dedup ranges across windows
    4. Detect breakouts and form post-break ranges
    5. Build active_ranges_per_bar

    Returns:
        (active_ranges_per_bar, all_ranges)
    """
    n = len(highs)
    window_size = max(max_bars * 2, 500)
    window_step = max(max_bars, 250)

    # --- Steps 1-3: Windowed zone detection ---
    raw_ranges: list[ZoneRange] = []
    for w_start in range(0, max(1, n - min_bars), window_step):
        w_end = min(w_start + window_size, n)
        window_ranges = _detect_ranges_in_window(
            highs, lows, closes, opens,
            w_start, w_end, w_start,
            n_swing, min_bars, max_bars,
            min_height_pct, max_height_pct,
            min_time_concentration, min_zone_count, cluster_pct,
        )
        raw_ranges.extend(window_ranges)

    # Dedup overlapping ranges across windows
    # Two ranges are duplicates if their boundaries are within 3% and they overlap in time
    ranges: list[ZoneRange] = []
    for zr in raw_ranges:
        is_dup = False
        for existing in ranges:
            high_close = abs(existing.high - zr.high) / (zr.high + 1e-8) < 0.03
            low_close = abs(existing.low - zr.low) / (zr.low + 1e-8) < 0.03
            time_close = abs(existing.start - zr.start) <= max_bars
            if high_close and low_close and time_close:
                if zr.concentration > existing.concentration:
                    ranges.remove(existing)
                else:
                    is_dup = True
                break
        if not is_dup:
            ranges.append(zr)

    # --- Step 4: Break detection + recapture + post-break ranges ---
    post_break_ranges = []

    for r in ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True
                r.broken_dir = 2
                r.broken_bar = i

                # Check for recapture within window
                recaptured = False
                for j in range(i + 1, min(i + 1 + recapture_window, n)):
                    if closes[j] >= r.support.bottom and closes[j] <= r.resistance.top:
                        r.broken = False
                        r.recaptured = True
                        r.recapture_bar = j
                        recaptured = True
                        break

                # Find lowest low after break → new support (post-break range)
                break_low = lows[i]
                break_low_bar = i
                for j in range(i + 1, min(i + max_bars // 2, n)):
                    if lows[j] < break_low:
                        break_low = lows[j]
                        break_low_bar = j
                    if closes[j] > r.support.top:
                        break

                new_support = PriceZone(break_low, break_low * 1.005,
                                        break_low, 1, [break_low_bar])
                new_resistance = r.support
                pb = _make_zone_range(new_resistance, new_support,
                                      break_low_bar, min_bars,
                                      min_height_pct, max_height_pct)
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
                    continue  # range is active again, keep scanning for next break
                break

            elif closes[i] > r.resistance.top:
                r.broken = True
                r.broken_dir = 1
                r.broken_bar = i

                # Check for recapture within window
                recaptured = False
                for j in range(i + 1, min(i + 1 + recapture_window, n)):
                    if closes[j] >= r.support.bottom and closes[j] <= r.resistance.top:
                        r.broken = False
                        r.recaptured = True
                        r.recapture_bar = j
                        recaptured = True
                        break

                break_high = highs[i]
                break_high_bar = i
                for j in range(i + 1, min(i + max_bars // 2, n)):
                    if highs[j] > break_high:
                        break_high = highs[j]
                        break_high_bar = j
                    if closes[j] < r.resistance.bottom:
                        break

                new_resistance = PriceZone(break_high * 0.995, break_high,
                                            break_high, 1, [break_high_bar])
                new_support = r.resistance
                pb = _make_zone_range(new_resistance, new_support,
                                      break_high_bar, min_bars,
                                      min_height_pct, max_height_pct)
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

    # Break detection on post-break ranges
    for r in post_break_ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True
                r.broken_dir = 2
                r.broken_bar = i
                break
            elif closes[i] > r.resistance.top:
                r.broken = True
                r.broken_dir = 1
                r.broken_bar = i
                break

    ranges.extend(post_break_ranges)

    # --- Step 4b: Macro range detection (second pass with wider params) ---
    macro_cluster_pct = 0.03
    macro_min_zone_count = max(min_zone_count - 1, 2)
    macro_max_height_pct = 0.18 if max_height_pct >= 0.10 else 0.10
    macro_min_time_conc = min_time_concentration * 0.90

    macro_raw: list[ZoneRange] = []
    for w_start in range(0, max(1, n - min_bars), max(max_bars, 250)):
        w_end = min(w_start + max(max_bars * 2, 500), n)
        window_ranges = _detect_ranges_in_window(
            highs, lows, closes, opens,
            w_start, w_end, w_start,
            n_swing, min_bars, max_bars,
            min_height_pct, macro_max_height_pct,
            macro_min_time_conc, macro_min_zone_count, macro_cluster_pct,
        )
        for wr in window_ranges:
            wr.is_macro = True
        macro_raw.extend(window_ranges)

    # Dedup macro ranges against each other only (not against inner ranges)
    macro_ranges: list[ZoneRange] = []
    for zr in macro_raw:
        is_dup = False
        for existing in macro_ranges:
            high_close = abs(existing.high - zr.high) / (zr.high + 1e-8) < 0.03
            low_close = abs(existing.low - zr.low) / (zr.low + 1e-8) < 0.03
            time_close = abs(existing.start - zr.start) <= max_bars
            if high_close and low_close and time_close:
                if zr.concentration > existing.concentration:
                    macro_ranges.remove(existing)
                else:
                    is_dup = True
                break
        if not is_dup:
            macro_ranges.append(zr)

    # Break detection on macro ranges
    for r in macro_ranges:
        for i in range(r.confirmed, min(r.start + max_bars, n)):
            if closes[i] < r.support.bottom:
                r.broken = True
                r.broken_dir = 2
                r.broken_bar = i
                break
            elif closes[i] > r.resistance.top:
                r.broken = True
                r.broken_dir = 1
                r.broken_bar = i
                break

    ranges.extend(macro_ranges)

    # --- Step 5: Build active_ranges_per_bar ---
    active_ranges = [[] for _ in range(n)]
    for r in ranges:
        end_bar = min(r.start + max_bars, n)
        for i in range(r.confirmed, end_bar):
            if r.recaptured:
                # Skip the brief broken period (broken_bar to recapture_bar)
                if r.broken_bar <= i < r.recapture_bar:
                    continue
            elif r.broken and i >= r.broken_bar:
                break
            active_ranges[i].append(r)

    return active_ranges, ranges


# ---------------------------------------------------------------------------
# Range-based TP/SL labels
# ---------------------------------------------------------------------------

def compute_range_tp_sl_labels(highs, lows, closes, actions, swept_levels, signal_map, horizon=18):
    """Compute MFE (max favorable excursion) + SL labels using range geometry.

    MFE = max favorable price move (as fraction of entry) before SL hit or
    horizon expiry.  quality = 1 if MFE > SL distance (profitable signal).

    SL = structural distance beyond tested boundary with 0.2% buffer.

    Returns: (quality, mfe_labels, sl_labels)
    """
    length = len(actions)
    quality = np.zeros(length, dtype=np.int64)
    mfe_labels = np.zeros(length, dtype=np.float32)
    sl_labels = np.zeros(length, dtype=np.float32)

    stop_buffer = 0.002  # 0.2% beyond zone boundary

    for i in range(length):
        if actions[i] == 0:
            continue
        if i + horizon >= length:
            actions[i] = 0
            continue

        entry = swept_levels[i]
        if entry <= 0:
            actions[i] = 0
            continue

        sig = signal_map.get(i)
        if sig is None:
            actions[i] = 0
            continue

        r = sig.range_ref
        range_height = r.resistance.level - r.support.level
        if range_height <= 0:
            actions[i] = 0
            continue

        if actions[i] == 1:  # long
            stop_price = r.support.bottom * (1 - stop_buffer)
            sl = max((entry - stop_price) / entry, 0.001)
        else:  # short
            stop_price = r.resistance.top * (1 + stop_buffer)
            sl = max((stop_price - entry) / entry, 0.001)

        sl_labels[i] = np.clip(sl, 0.001, 0.08)

        # Track MFE: max favorable move before SL hit or horizon expiry
        mfe = 0.0
        for j in range(i + 1, min(i + 1 + horizon, length)):
            if actions[i] == 1:  # long
                if lows[j] <= stop_price:
                    break
                favorable = (highs[j] - entry) / entry
            else:  # short
                if highs[j] >= stop_price:
                    break
                favorable = (entry - lows[j]) / entry
            mfe = max(mfe, favorable)

        mfe_labels[i] = np.clip(mfe, 0.0, 0.15)
        quality[i] = 1 if mfe > sl else 0

    return quality, mfe_labels, sl_labels


# ---------------------------------------------------------------------------
# Main label generator
# ---------------------------------------------------------------------------

def generate_labels(highs, lows, closes, opens, tf_key="4h"):
    """Top-level: detect ranges, filter SFPs at boundaries, compute labels.

    Returns:
        (actions, quality, tp_labels, sl_labels, swept_levels, signal_map, all_ranges, active_per_bar)
        signal_map: dict of bar_idx -> RangeSFPSignal
    """
    n = len(highs)
    atr = compute_atr(highs, lows, closes, period=14)

    # --- Range detection (zone-based v2) ---
    params = RANGE_SFP_PARAMS.get(tf_key, RANGE_SFP_PARAMS["4h"])
    height_min, height_max = RANGE_HEIGHT_PCT.get(tf_key, (0.005, 0.08))

    print(f"  [Range-SFP/{tf_key}] Detecting ranges (n_swing={params['n_swing']}, "
          f"min_bars={params['min_bars']})...")
    active_per_bar, all_ranges = detect_ranges_v2(
        highs, lows, closes, opens,
        n_swing=params["n_swing"],
        min_bars=params["min_bars"],
        max_bars=params["max_bars"],
        min_height_pct=height_min,
        max_height_pct=height_max,
        min_zone_count=params.get("min_zone_count", 2),
        min_time_concentration=params.get("min_time_concentration", 0.80),
        recapture_window=params.get("recapture_window", 5),
    )
    n_post = sum(1 for r in all_ranges if r.support.count == 1 or r.resistance.count == 1)
    n_macro = sum(1 for r in all_ranges if r.is_macro)
    n_recaptured = sum(1 for r in all_ranges if r.recaptured)
    n_primary = len(all_ranges) - n_post - n_macro
    print(f"    Found {len(all_ranges)} ranges ({n_primary} primary, {n_post} post-break, "
          f"{n_macro} macro, {n_recaptured} recaptured)")

    # --- SFP detection (n=5 and n=10, merge) ---
    reclaim_windows = {5: 1, 10: 3}
    results = {}
    for ns in [5, 10]:
        sh, sl = detect_swings(highs, lows, ns)
        active_sh, active_sl, _, _ = build_swing_level_series(
            highs, lows, sh, sl, ns, max_age=150
        )
        sfp_actions, sfp_swept = detect_sfp(
            highs, lows, closes, opens, active_sh, active_sl,
            reclaim_window=reclaim_windows[ns],
        )
        results[ns] = (sfp_actions, sfp_swept)

    # Merge SFP results
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
            raw_actions[i] = a5
            raw_swept[i] = swept_5[i]
        elif a10 != 0 and a5 == 0:
            raw_actions[i] = a10
            raw_swept[i] = swept_10[i]

    total_sfp = int(np.sum(raw_actions != 0))
    print(f"    Total SFPs detected: {total_sfp}")

    # --- Market structure ---
    ms_direction, ms_strength_arr, last_sh, last_sl = detect_market_structure(
        highs, lows, n=10
    )

    # --- Boundary filter: only keep SFPs at range boundaries (zone-based) ---
    actions = np.zeros(n, dtype=np.int64)
    swept_levels = np.zeros(n, dtype=np.float64)
    signal_map = {}

    for i in range(n):
        if raw_actions[i] == 0:
            continue
        if not active_per_bar[i]:
            continue

        swept = raw_swept[i]
        direction = raw_actions[i]

        # Find the BEST matching range (highest quality), not first
        best_range = None
        best_score = -1.0

        # Per-TF zone_buffer and reclaim strictness
        if tf_key == "15m":
            zone_buffer = 0.0              # no tolerance — strict
            strict_reclaim = True          # close must beat zone.top / zone.bottom
        elif tf_key == "4h":
            zone_buffer = atr[i] * 0.15   # aggressive tolerance
            strict_reclaim = False         # close >= zone.level is enough
        else:  # 1h
            zone_buffer = atr[i] * 0.1    # moderate tolerance
            strict_reclaim = False

        for r in active_per_bar[i]:
            range_height = r.high - r.low
            if range_height <= 0:
                continue

            is_boundary = False
            if direction == 1:  # long: swept at zone, deep wick, close above zone
                reclaim_ok = closes[i] > r.support.top if strict_reclaim else closes[i] >= r.support.level
                if (swept >= r.support.bottom - zone_buffer and swept <= r.support.top + zone_buffer
                        and lows[i] < r.support.level
                        and reclaim_ok):
                    is_boundary = True
            elif direction == 2:  # short: swept at zone, deep wick, close below zone
                reclaim_ok = closes[i] < r.resistance.bottom if strict_reclaim else closes[i] <= r.resistance.level
                if (swept >= r.resistance.bottom - zone_buffer and swept <= r.resistance.top + zone_buffer
                        and highs[i] > r.resistance.level
                        and reclaim_ok):
                    is_boundary = True

            if not is_boundary:
                continue

            # Range quality score: concentration * min touches
            score = r.concentration * min(r.touches_high, r.touches_low)
            if score > best_score:
                best_score = score
                best_range = r

        if best_range is None:
            continue

        r = best_range
        range_height = r.high - r.low

        # Compute signal features
        if direction == 1:
            sweep_depth = (r.support.top - lows[i]) / range_height
            reclaim_strength = (closes[i] - r.support.top) / range_height
        else:
            sweep_depth = (highs[i] - r.resistance.bottom) / range_height
            reclaim_strength = (r.resistance.bottom - closes[i]) / range_height

        # MS alignment: +1 if direction matches structure
        ms_dir = ms_direction[i]
        if direction == 1:  # long
            ms_align = 1.0 if ms_dir > 0 else (-1.0 if ms_dir < 0 else 0.0)
        else:  # short
            ms_align = 1.0 if ms_dir < 0 else (-1.0 if ms_dir > 0 else 0.0)

        actions[i] = direction
        swept_levels[i] = swept

        signal_map[i] = RangeSFPSignal(
            bar_idx=i,
            direction=direction,
            swept_level=swept,
            range_ref=r,
            sweep_depth=sweep_depth,
            reclaim_strength=reclaim_strength,
            ms_alignment=ms_align,
            ms_strength=ms_strength_arr[i],
        )

    n_boundary = int(np.sum(actions != 0))
    n_long = int(np.sum(actions == 1))
    n_short = int(np.sum(actions == 2))
    print(f"    Boundary-filtered: {n_boundary} signals ({n_long} long, {n_short} short)")

    # --- MFE/SL labels ---
    quality, mfe_labels, sl_labels = compute_range_tp_sl_labels(
        highs, lows, closes, actions, swept_levels, signal_map, horizon=18,
    )
    total_final = int(np.sum(actions != 0))
    if total_final > 0:
        n_prof = int(np.sum((actions != 0) & (quality == 1)))
        mask = actions != 0
        avg_mfe = float(np.mean(mfe_labels[mask]))
        avg_sl = float(np.mean(sl_labels[mask]))
        print(f"  [Range-SFP/{tf_key}] {total_final} signals -> "
              f"profitable: {n_prof} ({n_prof/total_final*100:.0f}%) | "
              f"avg MFE: {avg_mfe*100:.2f}% | avg SL: {avg_sl*100:.2f}%")
    else:
        print(f"  [Range-SFP/{tf_key}] No signals detected")

    return actions, quality, mfe_labels, sl_labels, swept_levels, signal_map, all_ranges, active_per_bar
