"""Liquidation level estimation from price action.

Estimates where liquidation clusters sit based on swing levels (likely entries)
and common leverage tiers. No exchange API needed — works with OHLCV only.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from src.labels.sfp_labels import detect_swings, build_swing_level_series
from src.labels.three_tap_labels import compute_atr


# Leverage tiers: (leverage, popularity_weight)
# Weights approximate retail usage distribution
LEVERAGE_TIERS = [
    (3, 0.05), (5, 0.15), (10, 0.30),
    (25, 0.25), (50, 0.15), (100, 0.10),
]

MM_RATE = 0.004  # Binance Tier 1 maintenance margin (0.4%)


@dataclass
class LiqLevel:
    price: float        # estimated liquidation price
    leverage: int       # leverage tier
    weight: float       # popularity weight
    swing_price: float  # entry price (the swing level)
    side: str           # "long" or "short" — what gets liquidated


def compute_liq_price(entry: float, leverage: int, side: str, mm_rate: float = MM_RATE) -> float:
    """Compute estimated liquidation price."""
    if side == "long":
        return entry * (1 - 1.0 / leverage + mm_rate)
    else:
        return entry * (1 + 1.0 / leverage - mm_rate)


def detect_liquidation_clusters(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray = None,
    n_swing: int = 5,
    max_age: int = 150,
    leverages: list = None,
    mm_rate: float = MM_RATE,
) -> dict:
    """Detect estimated liquidation clusters per bar.

    For each bar, computes:
    - All active liquidation levels above and below current price
    - Cluster density metrics (how many liq levels sit nearby)

    Args:
        highs, lows, closes: OHLC arrays
        atr: ATR array (computed if None)
        n_swing: swing detection window
        max_age: max bars a swing level stays active
        leverages: list of (leverage, weight) tuples
        mm_rate: maintenance margin rate

    Returns dict with:
        liq_above: list[list[LiqLevel]] — per-bar, short liquidation levels above price
        liq_below: list[list[LiqLevel]] — per-bar, long liquidation levels below price
        density_above: (N,) float32 — weighted liq density above current price within 1 ATR
        density_below: (N,) float32 — weighted liq density below current price within 1 ATR
        n_levels_above: (N,) int — count of liq levels above
        n_levels_below: (N,) int — count of liq levels below
        nearest_liq_above_pct: (N,) float32 — distance to nearest liq above as % of price
        nearest_liq_below_pct: (N,) float32 — distance to nearest liq below as % of price
    """
    if leverages is None:
        leverages = LEVERAGE_TIERS
    if atr is None:
        atr = compute_atr(highs, lows, closes, period=14)

    n = len(highs)

    # Detect swings and build active level series
    swing_highs, swing_lows = detect_swings(highs, lows, n_swing)
    active_sh, active_sl, active_sh_ages, active_sl_ages = build_swing_level_series(
        highs, lows, swing_highs, swing_lows, n_swing, max_age=max_age,
    )

    # Per-bar liquidation level lists
    liq_above = [[] for _ in range(n)]
    liq_below = [[] for _ in range(n)]

    # Summary arrays
    density_above = np.zeros(n, dtype=np.float32)
    density_below = np.zeros(n, dtype=np.float32)
    n_levels_above = np.zeros(n, dtype=np.int32)
    n_levels_below = np.zeros(n, dtype=np.int32)
    nearest_liq_above_pct = np.full(n, 999.0, dtype=np.float32)
    nearest_liq_below_pct = np.full(n, 999.0, dtype=np.float32)

    for i in range(n):
        price = closes[i]
        if price <= 0:
            continue
        local_atr = atr[i] if atr[i] > 0 else price * 0.01

        # Swing lows = likely long entries → liquidations sit BELOW
        for sl_price in active_sl[i]:
            for lev, weight in leverages:
                liq_p = compute_liq_price(sl_price, lev, "long", mm_rate)
                if liq_p < price:
                    ll = LiqLevel(price=liq_p, leverage=lev, weight=weight,
                                  swing_price=sl_price, side="long")
                    liq_below[i].append(ll)
                    dist = price - liq_p
                    dist_pct = dist / price
                    if dist_pct < nearest_liq_below_pct[i]:
                        nearest_liq_below_pct[i] = dist_pct
                    if dist <= local_atr:
                        density_below[i] += weight

        # Swing highs = likely short entries → liquidations sit ABOVE
        for sh_price in active_sh[i]:
            for lev, weight in leverages:
                liq_p = compute_liq_price(sh_price, lev, "short", mm_rate)
                if liq_p > price:
                    ll = LiqLevel(price=liq_p, leverage=lev, weight=weight,
                                  swing_price=sh_price, side="short")
                    liq_above[i].append(ll)
                    dist = liq_p - price
                    dist_pct = dist / price
                    if dist_pct < nearest_liq_above_pct[i]:
                        nearest_liq_above_pct[i] = dist_pct
                    if dist <= local_atr:
                        density_above[i] += weight

        n_levels_above[i] = len(liq_above[i])
        n_levels_below[i] = len(liq_below[i])

    # Cap the "no liq nearby" sentinel
    nearest_liq_above_pct[nearest_liq_above_pct > 1.0] = 1.0
    nearest_liq_below_pct[nearest_liq_below_pct > 1.0] = 1.0

    return {
        "liq_above": liq_above,
        "liq_below": liq_below,
        "density_above": density_above,
        "density_below": density_below,
        "n_levels_above": n_levels_above,
        "n_levels_below": n_levels_below,
        "nearest_liq_above_pct": nearest_liq_above_pct,
        "nearest_liq_below_pct": nearest_liq_below_pct,
    }
