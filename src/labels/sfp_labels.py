import numpy as np


def detect_swings(highs, lows, n):
    """Detect swing highs and swing lows using n-bar window on each side.

    A swing high at bar i means High[i] is the strict max of High[i-n : i+n+1].
    A swing low at bar i means Low[i] is the strict min of Low[i-n : i+n+1].
    """
    length = len(highs)
    swing_highs = np.zeros(length, dtype=bool)
    swing_lows = np.zeros(length, dtype=bool)

    for i in range(n, length - n):
        window_h = highs[i - n : i + n + 1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            swing_highs[i] = True

        window_l = lows[i - n : i + n + 1]
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            swing_lows[i] = True

    return swing_highs, swing_lows


def build_swing_level_series(highs, lows, swing_highs, swing_lows, n, max_age=150):
    """For each bar, collect all active confirmed swing high/low levels.

    A swing at index j is only confirmed at j+n (no look-ahead bias).
    max_age ignores stale levels older than max_age bars.

    Returns (active_sh, active_sl, active_sh_ages, active_sl_ages):
        active_sh[i] = [level1, level2, ...] all active swing highs at bar i
        active_sl[i] = [level1, level2, ...] all active swing lows at bar i
        active_sh_ages[i] = [age1, age2, ...] age in bars for each swing high
        active_sl_ages[i] = [age1, age2, ...] age in bars for each swing low
    """
    length = len(highs)
    active_sh = [[] for _ in range(length)]
    active_sl = [[] for _ in range(length)]
    active_sh_ages = [[] for _ in range(length)]
    active_sl_ages = [[] for _ in range(length)]

    for i in range(length):
        # Collect all confirmed swing highs within max_age
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_highs[j]:
                active_sh[i].append(highs[j])
                active_sh_ages[i].append(i - j)

        # Collect all confirmed swing lows within max_age
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_lows[j]:
                active_sl[i].append(lows[j])
                active_sl_ages[i].append(i - j)

    return active_sh, active_sl, active_sh_ages, active_sl_ages


def compute_swing_level_info(closes, active_sh, active_sl, active_sh_ages, active_sl_ages, max_age=150):
    """Compute per-bar swing level context features.

    Returns:
        nearest_age: age of nearest swing level, normalized by max_age
        confluence: count of levels within 1% of nearest, capped at 5, divided by 5
    """
    length = len(closes)
    nearest_age = np.zeros(length, dtype=np.float32)
    confluence = np.zeros(length, dtype=np.float32)

    for i in range(length):
        all_levels = list(active_sh[i]) + list(active_sl[i])
        all_ages = list(active_sh_ages[i]) + list(active_sl_ages[i])
        if not all_levels:
            continue

        # Find nearest level to current close
        distances = [abs(lvl - closes[i]) / (closes[i] + 1e-8) for lvl in all_levels]
        nearest_idx = np.argmin(distances)
        nearest_price = all_levels[nearest_idx]

        # Age of nearest level
        nearest_age[i] = min(all_ages[nearest_idx], max_age) / max_age

        # Confluence: count levels within 1% of nearest
        n_nearby = sum(1 for lvl in all_levels if abs(lvl - nearest_price) / (nearest_price + 1e-8) < 0.01)
        confluence[i] = min(n_nearby, 5) / 5.0

    return nearest_age, confluence


def detect_sfp(
    df_high,
    df_low,
    df_close,
    df_open,
    active_sh,
    active_sl,
    min_sweep_pct=0.003,
    max_sweep_pct=0.02,
    reclaim_window=1,
):
    """Detect Swing Failure Patterns (liquidity sweeps) with multi-candle reclaim.

    Checks sweeps against ALL active swing levels (not just most recent).
    Filters:
      - Body position: open must be on correct side of swing level
      - Sweep distance: 0.3% to 2%

    Bullish SFP: candle sweeps Low < swing_low, open > swing_low,
                 then reclaim Close > swing_low
    Bearish SFP: candle sweeps High > swing_high, open < swing_high,
                 then reclaim Close < swing_high

    Returns (actions, swept_levels):
        actions: 0=no-trade, 1=long, 2=short
        swept_levels: the swing level price that triggered the SFP (used as entry)
    """
    length = len(df_high)
    actions = np.zeros(length, dtype=np.int64)
    swept_levels = np.zeros(length, dtype=np.float64)

    for i in range(length):
        # Check bullish SFP: sweep below any active swing low
        for sl in active_sl[i]:
            if sl <= 0:
                continue
            sweep_dist = (sl - df_low[i]) / sl
            if sweep_dist < min_sweep_pct or sweep_dist > max_sweep_pct:
                continue
            # Body position filter: open must be above swing low
            if df_open[i] <= sl:
                continue
            # Look for reclaim
            for k in range(i, min(i + reclaim_window + 1, length)):
                if df_close[k] > sl and actions[k] == 0:
                    actions[k] = 1  # long on reclaim candle
                    swept_levels[k] = sl
                    break
            break  # only trigger on first matching level

        # Check bearish SFP: sweep above any active swing high
        for sh in active_sh[i]:
            if sh <= 0:
                continue
            sweep_dist = (df_high[i] - sh) / sh
            if sweep_dist < min_sweep_pct or sweep_dist > max_sweep_pct:
                continue
            # Body position filter: open must be below swing high
            if df_open[i] >= sh:
                continue
            # Look for reclaim
            for k in range(i, min(i + reclaim_window + 1, length)):
                if df_close[k] < sh:
                    if actions[k] == 1:
                        actions[k] = 0  # conflict → no trade
                        swept_levels[k] = 0  # clear conflicted level
                    elif actions[k] == 0:
                        actions[k] = 2  # short on reclaim candle
                        swept_levels[k] = sh
                    break
            break  # only trigger on first matching level

    return actions, swept_levels


def compute_tp_sl_labels(df_high, df_low, df_close, actions, swept_levels, horizon=18):
    """Compute TP/SL and quality labels for all SFP bars.

    Entry = swept swing level (not close), giving better R:R since entry
    is closer to the invalidation point.

    For longs: TP = (max future High - entry) / entry
               SL = (entry - min future Low) / entry
    For shorts: TP = (entry - min future Low) / entry
                SL = (max future High - entry) / entry

    Quality: 1 = profitable (end_close favorable AND stop not hit), 0 = losing.
    Clips TP/SL to [0.001, 0.10]. Non-SFP bars get tp=0, sl=0, quality=0.
    """
    length = len(actions)
    tp_labels = np.zeros(length, dtype=np.float32)
    sl_labels = np.zeros(length, dtype=np.float32)
    quality = np.zeros(length, dtype=np.int64)

    for i in range(length):
        if actions[i] == 0:
            continue
        if i + horizon > length:
            actions[i] = 0
            continue

        entry = swept_levels[i]
        if entry <= 0:
            actions[i] = 0
            continue

        future_highs = df_high[i + 1 : i + 1 + horizon]
        future_lows = df_low[i + 1 : i + 1 + horizon]
        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        end_close = df_close[i + horizon]

        # Determine profitability (quality label)
        if actions[i] == 1:  # long
            tp = (max_high - entry) / entry
            sl = (entry - min_low) / entry
            profitable = end_close > entry and min_low > df_low[i]
        else:  # short
            tp = (entry - min_low) / entry
            sl = (max_high - entry) / entry
            profitable = end_close < entry and max_high < df_high[i]

        quality[i] = 1 if profitable else 0
        tp_labels[i] = np.clip(tp, 0.001, 0.10)
        sl_labels[i] = np.clip(sl, 0.001, 0.10)

    return quality, tp_labels, sl_labels


def generate_labels(df_high, df_low, df_close, df_open):
    """Top-level function: run SFP pipeline for n=5 and n=10, merge results.

    Agreement → keep, conflict → no-trade.
    Returns (actions, quality, tp_labels, sl_labels) where:
      - actions: 0=no-trade, 1=long, 2=short (ALL detected SFPs)
      - quality: 1=profitable, 0=losing (only set for SFP bars)
    """
    highs = df_high
    lows = df_low

    reclaim_windows = {5: 1, 10: 3}  # LTF → 1, HTF → 3
    results = {}
    for n in [5, 10]:
        sh, sl = detect_swings(highs, lows, n)
        active_sh, active_sl, _, _ = build_swing_level_series(highs, lows, sh, sl, n, max_age=150)
        actions, swept = detect_sfp(
            df_high,
            df_low,
            df_close,
            df_open,
            active_sh,
            active_sl,
            reclaim_window=reclaim_windows[n],
        )
        results[n] = (actions, swept)

    # Merge: agreement keeps, conflict → no-trade
    actions_5, swept_5 = results[5]
    actions_10, swept_10 = results[10]
    merged = np.zeros(len(highs), dtype=np.int64)
    merged_swept = np.zeros(len(highs), dtype=np.float64)

    for i in range(len(highs)):
        a5 = actions_5[i]
        a10 = actions_10[i]

        if a5 == a10:
            merged[i] = a5
            merged_swept[i] = swept_5[i] if swept_5[i] > 0 else swept_10[i]
        elif a5 != 0 and a10 == 0:
            merged[i] = a5
            merged_swept[i] = swept_5[i]
        elif a10 != 0 and a5 == 0:
            merged[i] = a10
            merged_swept[i] = swept_10[i]
        else:
            merged[i] = 0  # conflict

    total_sfp = int(np.sum(merged != 0))
    quality, tp_labels, sl_labels = compute_tp_sl_labels(df_high, df_low, df_close, merged, merged_swept)
    n_profitable = int(np.sum((merged != 0) & (quality == 1)))
    n_losing = total_sfp - n_profitable
    print(f"  SFP funnel: {total_sfp} detected → {n_profitable} profitable, {n_losing} losing")

    return merged, quality, tp_labels, sl_labels
