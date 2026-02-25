"""Fetch OHLCV from Binance and compute technical indicators."""

import httpx
import pandas as pd
from ta import volume, volatility, trend, momentum


async def fetch_candles(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance and return DataFrame with indicators.

    Indicators match data/re_sample.py exactly.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        raw = resp.json()

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
