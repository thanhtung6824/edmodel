"""Combined Liquidation + Range + SFP label generator.

Same signal set as range_sfp (SFP at range boundary), enriched with 6 liquidation
features. Liq data is soft — the model learns that liq confluence = higher P(win),
rather than filtering out signals without liq sweeps.
"""

import numpy as np
from dataclasses import dataclass, field

from src.labels.sfp_labels import detect_swings, build_swing_level_series, detect_sfp
from src.labels.range_sfp_labels import (
    detect_ranges_v2,
    detect_market_structure,
    RANGE_SFP_PARAMS,
    RANGE_HEIGHT_PCT,
    ZoneRange,
    compute_range_tp_sl_labels,
)
from src.labels.liq_labels import compute_liq_price, LEVERAGE_TIERS
from src.labels.three_tap_labels import compute_atr


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LiqRangeSFPSignal:
    bar_idx: int
    direction: int              # 1=long, 2=short
    swept_level: float
    range_ref: ZoneRange = field(repr=False)
    # Range features
    range_height_pct: float = 0.0
    range_touches: float = 0.0
    range_concentration: float = 0.0
    range_age: float = 0.0
    sweep_depth_range: float = 0.0
    reclaim_strength_range: float = 0.0
    # Liquidation features (soft — 0 if no liq nearby)
    n_liq_swept: int = 0
    weighted_liq_swept: float = 0.0
    max_leverage_swept: int = 0
    liq_cascade_depth: float = 0.0
    liq_cluster_density: float = 0.0
    n_swings_with_liq: int = 0
    # SFP / context features (computed in build_features)
    ms_alignment: float = 0.0
    ms_strength: float = 0.0


# ---------------------------------------------------------------------------
# Main label generator
# ---------------------------------------------------------------------------

def generate_labels(highs, lows, closes, opens, volumes=None, tf_key="4h"):
    """Top-level: SFP at range boundary + soft liq features.

    Signal filter: SFP + range boundary (same as range_sfp_labels).
    Liq data is computed but NOT used as a filter — stored as features.

    Returns:
        (actions, quality, tp_labels, sl_labels, swept_levels, signal_map)
        signal_map: dict of bar_idx -> LiqRangeSFPSignal
    """
    n = len(highs)
    atr = compute_atr(highs, lows, closes, period=14)

    # --- 1. Range detection (zone-based v2) ---
    params = RANGE_SFP_PARAMS.get(tf_key, RANGE_SFP_PARAMS["4h"])
    height_min, height_max = RANGE_HEIGHT_PCT.get(tf_key, (0.005, 0.08))

    print(f"  [LiqRangeSFP/{tf_key}] Detecting ranges...")
    active_per_bar, all_ranges = detect_ranges_v2(
        highs, lows, closes, opens,
        n_swing=params["n_swing"],
        min_bars=params["min_bars"],
        max_bars=params["max_bars"],
        min_height_pct=height_min,
        max_height_pct=height_max,
        min_zone_count=params.get("min_zone_count", 2),
        min_time_concentration=params.get("min_time_concentration", 0.80),
    )
    n_post = sum(1 for r in all_ranges if r.support.count == 1 or r.resistance.count == 1)
    n_primary = len(all_ranges) - n_post
    print(f"    Found {len(all_ranges)} ranges ({n_primary} primary, {n_post} post-break)")

    # --- 2+3. Swing detection (shared for SFP + liq) ---
    print(f"  [LiqRangeSFP/{tf_key}] Detecting swings + SFPs...")
    reclaim_windows = {5: 1, 10: 3}
    results = {}
    liq_swing_data = {}
    for ns in [5, 10]:
        sh, sl = detect_swings(highs, lows, ns)
        # SFP uses max_age=150
        active_sh, active_sl, _, _ = build_swing_level_series(
            highs, lows, sh, sl, ns, max_age=150
        )
        sfp_actions, sfp_swept = detect_sfp(
            highs, lows, closes, opens, active_sh, active_sl,
            reclaim_window=reclaim_windows[ns],
        )
        results[ns] = (sfp_actions, sfp_swept)
        # Liq uses max_age=200 (wider window)
        active_sh_liq, active_sl_liq, _, _ = build_swing_level_series(
            highs, lows, sh, sl, ns, max_age=200
        )
        liq_swing_data[ns] = (active_sh_liq, active_sl_liq)

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

    # --- 4. Market structure ---
    ms_direction, ms_strength_arr, _, _ = detect_market_structure(highs, lows, n=10)

    # --- 5. Boundary filter (same as range_sfp) + soft liq features ---
    actions = np.zeros(n, dtype=np.int64)
    swept_levels = np.zeros(n, dtype=np.float64)
    signal_map = {}

    for i in range(n):
        # Filter 1: SFP exists?
        if raw_actions[i] == 0:
            continue

        # Filter 2: Inside active range?
        if not active_per_bar[i]:
            continue

        swept = raw_swept[i]
        direction = raw_actions[i]

        # Filter 3: SFP sweeps a range boundary? (zone-based)
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
            if direction == 1:  # long: swept at support zone
                reclaim_ok = closes[i] > r.support.top if strict_reclaim else closes[i] >= r.support.level
                if (swept >= r.support.bottom - zone_buffer and swept <= r.support.top + zone_buffer
                        and lows[i] < r.support.level
                        and reclaim_ok):
                    is_boundary = True
            elif direction == 2:  # short: swept at resistance zone
                reclaim_ok = closes[i] < r.resistance.bottom if strict_reclaim else closes[i] <= r.resistance.level
                if (swept >= r.resistance.bottom - zone_buffer and swept <= r.resistance.top + zone_buffer
                        and highs[i] > r.resistance.level
                        and reclaim_ok):
                    is_boundary = True

            if not is_boundary:
                continue

            score = r.concentration * min(r.touches_high, r.touches_low)
            if score > best_score:
                best_score = score
                best_range = r

        if best_range is None:
            continue

        # --- Compute liq features lazily (only for signal bars) ---
        n_liq_swept = 0
        weighted_liq = 0.0
        max_leverage = 0
        liq_prices_swept = []
        swings_with_liq = set()
        total_side = 0
        local_atr = atr[i] if atr[i] > 0 else closes[i] * 0.01
        proximity = local_atr * 0.5
        price = closes[i]

        for ns in [5, 10]:
            active_sh, active_sl = liq_swing_data[ns]
            if direction == 1:  # long: liq from swing lows (long entries) below
                for sl_price in active_sl[i]:
                    for lev, weight in LEVERAGE_TIERS:
                        liq_p = compute_liq_price(sl_price, lev, "long")
                        if liq_p < price:
                            total_side += 1
                            if liq_p >= lows[i] - proximity:
                                key = round(liq_p, 2)
                                n_liq_swept += 1
                                weighted_liq += weight
                                max_leverage = max(max_leverage, lev)
                                liq_prices_swept.append(liq_p)
                                swings_with_liq.add(round(sl_price, 2))
            else:  # short: liq from swing highs (short entries) above
                for sh_price in active_sh[i]:
                    for lev, weight in LEVERAGE_TIERS:
                        liq_p = compute_liq_price(sh_price, lev, "short")
                        if liq_p > price:
                            total_side += 1
                            if liq_p <= highs[i] + proximity:
                                key = round(liq_p, 2)
                                n_liq_swept += 1
                                weighted_liq += weight
                                max_leverage = max(max_leverage, lev)
                                liq_prices_swept.append(liq_p)
                                swings_with_liq.add(round(sh_price, 2))

        r = best_range
        range_height = r.high - r.low
        mid_price = (r.high + r.low) / 2.0

        # Range features
        if direction == 1:
            sweep_depth = (r.support.top - lows[i]) / range_height
            reclaim_strength = (closes[i] - r.support.top) / range_height
        else:
            sweep_depth = (highs[i] - r.resistance.bottom) / range_height
            reclaim_strength = (r.resistance.bottom - closes[i]) / range_height

        # Liq cascade depth
        if liq_prices_swept:
            cascade_depth = max(liq_prices_swept) - min(liq_prices_swept)
        else:
            cascade_depth = 0.0

        # Liq cluster density: triggered / total on that side
        cluster_density = n_liq_swept / max(total_side, 1)

        # MS alignment
        ms_dir = ms_direction[i]
        if direction == 1:
            ms_align = 1.0 if ms_dir > 0 else (-1.0 if ms_dir < 0 else 0.0)
        else:
            ms_align = 1.0 if ms_dir < 0 else (-1.0 if ms_dir > 0 else 0.0)

        actions[i] = direction
        swept_levels[i] = swept

        signal_map[i] = LiqRangeSFPSignal(
            bar_idx=i,
            direction=direction,
            swept_level=swept,
            range_ref=r,
            range_height_pct=range_height / (mid_price + 1e-8),
            range_touches=min(r.touches_high, r.touches_low),
            range_concentration=r.concentration,
            range_age=(i - r.confirmed) / 200.0,
            sweep_depth_range=sweep_depth,
            reclaim_strength_range=reclaim_strength,
            n_liq_swept=n_liq_swept,
            weighted_liq_swept=weighted_liq,
            max_leverage_swept=max_leverage,
            liq_cascade_depth=cascade_depth,
            liq_cluster_density=cluster_density,
            n_swings_with_liq=len(swings_with_liq),
            ms_alignment=ms_align,
            ms_strength=ms_strength_arr[i],
        )

    n_boundary = int(np.sum(actions != 0))
    n_long = int(np.sum(actions == 1))
    n_short = int(np.sum(actions == 2))
    n_with_liq = sum(1 for s in signal_map.values() if s.n_liq_swept > 0)
    print(f"    Boundary-filtered: {n_boundary} signals ({n_long} long, {n_short} short)")
    print(f"    With liq confluence: {n_with_liq}/{n_boundary} ({n_with_liq/max(n_boundary,1)*100:.0f}%)")

    # --- TP/SL labels (zone-based structural SL) ---
    quality, tp_labels, sl_labels = compute_range_tp_sl_labels(
        highs, lows, closes, actions, swept_levels, signal_map, horizon=18,
    )
    n_profitable = int(np.sum((actions != 0) & (quality == 1)))
    total_final = int(np.sum(actions != 0))
    if total_final > 0:
        print(f"  [LiqRangeSFP/{tf_key}] Funnel: {total_final} signals -> "
              f"{n_profitable} profitable ({n_profitable / total_final * 100:.1f}%)")
    else:
        print(f"  [LiqRangeSFP/{tf_key}] No signals detected")

    return actions, quality, tp_labels, sl_labels, swept_levels, signal_map
