"""Fetch OHLCV from Binance and compute technical indicators."""

import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd
from ta import volume, volatility, trend, momentum

from server.config import BAR_CACHE_DIR, BAR_CACHE_MAX

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Map tf_key to CSV filename suffix used in data/ directory
_TF_TO_CSV_SUFFIX = {"15m": "15min", "1h": "1h", "4h": "4h"}
_CACHE_COLS = ["timestamp", "Open", "High", "Low", "Close", "Volume"]


def _cache_path(asset_key: str, tf_key: str) -> Path:
    return Path(BAR_CACHE_DIR) / f"{asset_key}_{tf_key}.csv"


def load_bar_cache(asset_key: str, tf_key: str) -> pd.DataFrame:
    """Load cached bars from disk. Returns empty DataFrame if no cache."""
    path = _cache_path(asset_key, tf_key)
    if not path.exists():
        return pd.DataFrame(columns=_CACHE_COLS)
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        logger.warning(f"Failed to read bar cache {path}, starting fresh")
        return pd.DataFrame(columns=_CACHE_COLS)


def save_bar_cache(df: pd.DataFrame, asset_key: str, tf_key: str) -> None:
    """Save bar cache to disk."""
    path = _cache_path(asset_key, tf_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[_CACHE_COLS].to_csv(path, index=False)


def merge_bar_cache(cached_df: pd.DataFrame, fresh_df: pd.DataFrame, max_bars: int = BAR_CACHE_MAX) -> pd.DataFrame:
    """Merge cached and fresh bars, dedup by timestamp, keep latest max_bars."""
    if cached_df.empty:
        merged = fresh_df.copy()
    elif fresh_df.empty:
        merged = cached_df.copy()
    else:
        # Only keep OHLCV + timestamp from cache (indicators will be recomputed)
        cached_ohlcv = cached_df[_CACHE_COLS].copy()
        fresh_ohlcv = fresh_df[_CACHE_COLS].copy()
        merged = pd.concat([cached_ohlcv, fresh_ohlcv], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Trim to max_bars (keep most recent)
    if len(merged) > max_bars:
        merged = merged.iloc[-max_bars:].reset_index(drop=True)

    return merged


def seed_from_csv(asset_key: str, tf_key: str, max_seed: int = 3000) -> pd.DataFrame:
    """Seed cache from historical CSV data if available. Returns DataFrame or empty."""
    suffix = _TF_TO_CSV_SUFFIX.get(tf_key, tf_key)
    csv_path = Path(f"data/{asset_key}_{suffix}.csv")
    if not csv_path.exists():
        return pd.DataFrame(columns=_CACHE_COLS)

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Take tail (most recent bars)
        if len(df) > max_seed:
            df = df.iloc[-max_seed:]
        logger.info(f"Seeded {len(df)} bars from {csv_path}")
        return df[_CACHE_COLS].reset_index(drop=True)
    except Exception:
        logger.warning(f"Failed to seed from {csv_path}")
        return pd.DataFrame(columns=_CACHE_COLS)


def recompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute technical indicators on merged OHLCV data."""
    df = df.copy()
    df["obv"] = volume.on_balance_volume(close=df["Close"], volume=df["Volume"], fillna=True)
    df["bb"] = volatility.bollinger_wband(close=df["Close"], window=20, window_dev=2, fillna=True)
    df["ema_21"] = trend.ema_indicator(close=df["Close"], window=21, fillna=True)
    df["rsi"] = momentum.rsi(close=df["Close"], fillna=True)
    df = df.dropna().reset_index(drop=True)
    return df


async def fetch_candles(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance and return DataFrame with indicators.

    Retries up to 3 times on connection/timeout errors.
    Indicators match data/re_sample.py exactly.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                raw = resp.json()
            break
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"Binance fetch failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise

    # Binance kline format: [open_time, O, H, L, C, V, close_time, ...]
    rows = []
    for k in raw:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    df = pd.DataFrame(rows)

    # Compute indicators — must exactly match data/re_sample.py
    df["obv"] = volume.on_balance_volume(close=df["Close"], volume=df["Volume"], fillna=True)
    df["bb"] = volatility.bollinger_wband(close=df["Close"], window=20, window_dev=2, fillna=True)
    df["ema_21"] = trend.ema_indicator(close=df["Close"], window=21, fillna=True)
    df["rsi"] = momentum.rsi(close=df["Close"], fillna=True)
    df = df.dropna().reset_index(drop=True)

    return df
