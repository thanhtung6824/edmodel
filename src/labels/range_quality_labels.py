"""Per-bar range detection labels and features.

Predicts P(in_tradeable_range) per bar — 100K+ samples vs ~360 range-level.

Two main functions:
  - build_per_bar_features(): (N, 14) float32 features for every bar
  - generate_per_bar_labels(): binary label per bar (1=inside good range, 0=not)

Also retains label_ranges() for the range-level quality classifier.
"""

import numpy as np

from src.labels.range_sfp_labels import generate_labels


def build_per_bar_features(highs, lows, closes, opens, volumes, rsi, bb, ema_21, tf_hours, asset_id):
    """Build 14 features per bar for range detection.

    Features (all float32):
        0: close_return     — close-to-close pct change
        1: high_return      — high-to-high pct change
        2: low_return       — low-to-low pct change
        3: open_return      — open-to-open pct change
        4: body_ratio       — (close-open) / candle_range, signed
        5: upper_wick_ratio — upper_wick / candle_range
        6: lower_wick_ratio — lower_wick / candle_range
        7: vol_rel_20       — volume / 20-bar mean volume
        8: rsi_norm         — rsi / 100
        9: bb_width         — bollinger bandwidth (bb / close)
       10: trend_strength   — (close - ema_21) / close
       11: atr_rel          — ATR(14) / close
       12: tf_hours_norm    — tf_hours / 4.0
       13: asset_id         — raw asset id
    """
    n = len(highs)
    features = np.zeros((n, 14), dtype=np.float32)

    # Returns (shift by 1, first bar = 0)
    for i in range(1, n):
        if closes[i - 1] > 0:
            features[i, 0] = (closes[i] - closes[i - 1]) / closes[i - 1]
        if highs[i - 1] > 0:
            features[i, 1] = (highs[i] - highs[i - 1]) / highs[i - 1]
        if lows[i - 1] > 0:
            features[i, 2] = (lows[i] - lows[i - 1]) / lows[i - 1]
        if opens[i - 1] > 0:
            features[i, 3] = (opens[i] - opens[i - 1]) / opens[i - 1]

    # Body and wick ratios
    for i in range(n):
        cr = highs[i] - lows[i]
        if cr > 0:
            features[i, 4] = (closes[i] - opens[i]) / cr
            upper_wick = highs[i] - max(closes[i], opens[i])
            lower_wick = min(closes[i], opens[i]) - lows[i]
            features[i, 5] = upper_wick / cr
            features[i, 6] = lower_wick / cr

    # Volume relative to 20-bar mean
    vol_cumsum = np.cumsum(volumes)
    for i in range(n):
        start = max(0, i - 20)
        count = i - start
        if count > 0:
            avg_vol = (vol_cumsum[i] - (vol_cumsum[start] if start > 0 else 0)) / count
            features[i, 7] = volumes[i] / (avg_vol + 1e-8)
        else:
            features[i, 7] = 1.0

    # RSI normalized
    features[:, 8] = rsi / 100.0

    # BB width
    for i in range(n):
        if closes[i] > 0:
            features[i, 9] = bb[i] / closes[i]

    # Trend strength
    for i in range(n):
        if closes[i] > 0:
            features[i, 10] = (closes[i] - ema_21[i]) / closes[i]

    # ATR relative (14-period)
    from src.labels.three_tap_labels import compute_atr
    atr = compute_atr(highs, lows, closes, period=14)
    for i in range(n):
        if closes[i] > 0:
            features[i, 11] = atr[i] / closes[i]

    # TF and asset
    features[:, 12] = tf_hours / 4.0
    features[:, 13] = asset_id

    # Clean NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def generate_per_bar_labels(highs, lows, closes, opens, tf_key="4h"):
    """Generate per-bar binary labels: 1 if inside an active tradeable range, else 0.

    Calls generate_labels() to get all_ranges, signal_map, quality, active_per_bar.
    Calls label_ranges() to get per-range quality.
    Per bar: label=1 if inside an active range with quality=1, else 0.

    Returns:
        labels: (N,) int array — 1=in good range, 0=not
        all_ranges: list of ZoneRange
        active_per_bar: list of lists of active ranges per bar
    """
    actions, quality, tp_labels, sl_labels, swept_levels, signal_map, all_ranges, active_per_bar = generate_labels(
        highs, lows, closes, opens, tf_key=tf_key,
    )

    # Label each range for quality
    range_labels = label_ranges(all_ranges, highs, lows, closes, signal_map, quality)

    # Build set of good range ids for fast lookup
    good_range_ids = set()
    for ri, r in enumerate(all_ranges):
        if range_labels[ri] == 1:
            good_range_ids.add(id(r))

    # Per-bar labels
    n = len(highs)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if not active_per_bar[i]:
            continue
        for r in active_per_bar[i]:
            if id(r) in good_range_ids:
                labels[i] = 1
                break

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    print(f"  [PerBar/{tf_key}] Labels: {n_pos} positive, {n_neg} negative "
          f"({n_pos / max(n_pos + n_neg, 1) * 100:.1f}% in range)")

    return labels, all_ranges, active_per_bar


def label_ranges(all_ranges, highs, lows, closes, signal_map, quality, eval_window=50):
    """Label ranges for quality classification.

    Two labeling sources (priority order):
    1. SFP-outcome: If range has SFP signals at boundaries via signal_map
       - label=1 if any SFP won, label=0 if all lost
    2. Boundary respect: proxy for ranges without SFPs
       - Look at next eval_window bars after confirmed
       - Count distinct boundary rejections (max 1 per bar)
       - Track if/when range broke (close outside boundaries)
       - label=1 if held (no break) AND rejections >= 3
       - label=0 if broke quickly (within eval_window/3)
       - label=0 if broke AND rejections < 3
       - label=1 if broke late (after eval_window/2) AND rejections >= 4
       - label=-1 (exclude) only if truly ambiguous

    Returns:
        labels: (N_ranges,) int array. 1=good, 0=bad, -1=exclude
    """
    n = len(highs)
    labels = np.full(len(all_ranges), -1, dtype=np.int64)

    # Build map: range object id -> list of (bar_idx, quality_value)
    range_signals = {}
    for bar_idx, sig in signal_map.items():
        r_id = id(sig.range_ref)
        if r_id not in range_signals:
            range_signals[r_id] = []
        if bar_idx < len(quality):
            range_signals[r_id].append((bar_idx, int(quality[bar_idx])))

    for ri, r in enumerate(all_ranges):
        r_id = id(r)

        # Source 1: SFP-outcome
        if r_id in range_signals:
            valid_sigs = [(bi, q) for bi, q in range_signals[r_id] if q in (0, 1)]
            if valid_sigs:
                any_won = any(q == 1 for _, q in valid_sigs)
                labels[ri] = 1 if any_won else 0
                continue

        # Source 2: Boundary respect proxy
        ci = r.confirmed
        if ci + eval_window >= n:
            continue

        future_highs = highs[ci + 1:ci + 1 + eval_window]
        future_lows = lows[ci + 1:ci + 1 + eval_window]
        future_closes = closes[ci + 1:ci + 1 + eval_window]

        # Count rejections: max 1 per bar (either resistance OR support, not both)
        rejections = 0
        for j in range(len(future_highs)):
            close_inside = (future_closes[j] >= r.support.bottom
                            and future_closes[j] <= r.resistance.top)
            if not close_inside:
                continue
            # Resistance rejection: wick reaches into resistance zone
            if future_highs[j] >= r.resistance.bottom:
                rejections += 1
            # Support rejection: wick reaches into support zone
            elif future_lows[j] <= r.support.top:
                rejections += 1

        # Find when/if range broke
        broke_bar = -1
        for j in range(len(future_closes)):
            if future_closes[j] > r.resistance.top or future_closes[j] < r.support.bottom:
                broke_bar = j
                break

        quick_window = eval_window // 3
        half_window = eval_window // 2
        broke_quickly = broke_bar >= 0 and broke_bar < quick_window
        broke_late = broke_bar >= half_window
        held = broke_bar < 0  # never broke within eval_window

        if held and rejections >= 3:
            labels[ri] = 1  # strong range: held with good rejections
        elif held and rejections < 2:
            labels[ri] = 0  # range "held" but no real interaction — weak/irrelevant
        elif broke_quickly:
            labels[ri] = 0  # broke fast — bad range
        elif broke_bar >= 0 and rejections < 3:
            labels[ri] = 0  # broke with few rejections — bad range
        elif broke_late and rejections >= 4:
            labels[ri] = 1  # held for a long time with strong rejections before breaking
        # else: ambiguous (held with moderate rejections, or broke moderately)

    return labels
