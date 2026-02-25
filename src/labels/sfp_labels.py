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

    Returns (active_sh, active_sl) arrays of lists:
        active_sh[i] = [level1, level2, ...] all active swing highs at bar i
        active_sl[i] = [level1, level2, ...] all active swing lows at bar i
    """
    length = len(highs)
    active_sh = [[] for _ in range(length)]
    active_sl = [[] for _ in range(length)]

    for i in range(length):
        # Collect all confirmed swing highs within max_age
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_highs[j]:
                active_sh[i].append(highs[j])

        # Collect all confirmed swing lows within max_age
        for j in range(i - n, -1, -1):
            if i - j > max_age + n:
                break
            if swing_lows[j]:
                active_sl[i].append(lows[j])

    return active_sh, active_sl


def detect_sfp(df_high, df_low, df_close, df_open,
               active_sh, active_sl,
               min_sweep_pct=0.003, max_sweep_pct=0.02,
               reclaim_window=1):
    """Detect Swing Failure Patterns (liquidity sweeps) with multi-candle reclaim.

    Checks sweeps against ALL active swing levels (not just most recent).
    Filters:
      - Body position: open must be on correct side of swing level
      - Sweep distance: 0.3% to 2%

    Bullish SFP: candle sweeps Low < swing_low, open > swing_low,
                 then reclaim Close > swing_low
    Bearish SFP: candle sweeps High > swing_high, open < swing_high,
                 then reclaim Close < swing_high

    Returns actions array: 0=no-trade, 1=long, 2=short.
    """
    length = len(df_high)
    actions = np.zeros(length, dtype=np.int64)

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
                    elif actions[k] == 0:
                        actions[k] = 2  # short on reclaim candle
                    break
            break  # only trigger on first matching level

    return actions


def compute_tp_sl_labels(df_high, df_low, df_close, actions, horizon=18):
    """Compute TP/SL and quality labels for all SFP bars.

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

        entry = df_close[i]
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
        active_sh, active_sl = build_swing_level_series(highs, lows, sh, sl, n, max_age=150)
        actions = detect_sfp(df_high, df_low, df_close, df_open,
                             active_sh, active_sl,
                             reclaim_window=reclaim_windows[n])
        results[n] = actions

    # Merge: agreement keeps, conflict → no-trade
    actions_5 = results[5]
    actions_10 = results[10]
    merged = np.zeros(len(highs), dtype=np.int64)

    for i in range(len(highs)):
        a5 = actions_5[i]
        a10 = actions_10[i]

        if a5 == a10:
            merged[i] = a5
        elif a5 != 0 and a10 == 0:
            merged[i] = a5
        elif a10 != 0 and a5 == 0:
            merged[i] = a10
        else:
            merged[i] = 0  # conflict

    total_sfp = int(np.sum(merged != 0))
    quality, tp_labels, sl_labels = compute_tp_sl_labels(df_high, df_low, df_close, merged)
    n_profitable = int(np.sum((merged != 0) & (quality == 1)))
    n_losing = total_sfp - n_profitable
    print(f"  SFP funnel: {total_sfp} detected → {n_profitable} profitable, {n_losing} losing")

    return merged, quality, tp_labels, sl_labels
