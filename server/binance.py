"""Fetch OHLCV from Binance and compute technical indicators."""

import asyncio
import logging

import httpx
import pandas as pd
from ta import volume, volatility, trend, momentum

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


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
