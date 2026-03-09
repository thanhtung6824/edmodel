"""Fetch OHLCV from Binance and compute technical indicators."""

import asyncio
import json
import logging
from collections import deque

import httpx
import pandas as pd
import websockets
from ta import volume, volatility, trend, momentum

from server.config import N_CANDLES, WS_RECONNECT_DELAY, WS_SPOT_URL, WS_FUTURES_URL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def fetch_candles(symbol: str, interval: str, limit: int = 500, futures: bool = False) -> pd.DataFrame:
    """Fetch klines from Binance and return DataFrame with indicators.

    Retries up to 3 times on connection/timeout errors.
    Indicators match data/re_sample.py exactly.
    Uses Futures API (/fapi) when futures=True (e.g. HYPEUSDT).
    """
    if futures:
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
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


async def fetch_raw_klines(symbol: str, interval: str, limit: int = 500, futures: bool = False) -> list[dict]:
    """Fetch raw klines from Binance as list of OHLCV dicts (no indicators).

    Used by BinanceStreamManager to populate the candle cache.
    """
    if futures:
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
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

    return [
        {
            "timestamp": int(k[0]),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        }
        for k in raw
    ]


class BinanceStreamManager:
    """Manages WebSocket kline streams with a local candle cache.

    Cache stores raw OHLCV (no indicators). build_df() computes indicators
    on the fly, producing the same DataFrame shape as fetch_candles().
    """

    def __init__(self, active_assets: dict, timeframes: dict):
        self.active_assets = active_assets
        self.timeframes = timeframes
        self.on_candle_close = None  # async callback(asset_key, tf_key, df)

        # Closed candle cache: {(asset_key, tf_key): deque[dict]}
        self.cache: dict[tuple, deque] = {}
        # Current forming candle: {(asset_key, tf_key): dict}
        self.current_candle: dict[tuple, dict] = {}

        # Reverse mappings for fast WS message routing
        self._symbol_to_asset: dict[str, str] = {}
        for key, cfg in active_assets.items():
            self._symbol_to_asset[cfg["symbol"]] = key

        self._interval_to_tf: dict[str, str] = {}
        for tf_key, tf_cfg in timeframes.items():
            self._interval_to_tf[tf_cfg["interval"]] = tf_key

        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Fetch initial history via REST, then start WS background tasks."""
        self._running = True
        await self._fetch_initial_history()

        # Group streams by spot vs futures
        spot_streams: list[str] = []
        futures_streams: list[str] = []
        for asset_key, cfg in self.active_assets.items():
            sym = cfg["symbol"].lower()
            is_futures = cfg.get("futures", False)
            for tf_cfg in self.timeframes.values():
                stream = f"{sym}@kline_{tf_cfg['interval']}"
                if is_futures:
                    futures_streams.append(stream)
                else:
                    spot_streams.append(stream)

        if spot_streams:
            self._tasks.append(
                asyncio.create_task(self._run_ws(WS_SPOT_URL, spot_streams, "spot"))
            )
            logger.info(f"Starting spot WebSocket ({len(spot_streams)} streams)")
        if futures_streams:
            self._tasks.append(
                asyncio.create_task(self._run_ws(WS_FUTURES_URL, futures_streams, "futures"))
            )
            logger.info(f"Starting futures WebSocket ({len(futures_streams)} streams)")

    async def stop(self):
        """Cancel all WS tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    # ── REST history ─────────────────────────────────────────────

    async def _fetch_initial_history(self):
        """Fetch N_CANDLES of raw OHLCV for every asset×TF pair (concurrent)."""
        sem = asyncio.Semaphore(10)

        async def _one(asset_key, asset_cfg, tf_key, tf_cfg):
            async with sem:
                rows = await fetch_raw_klines(
                    asset_cfg["symbol"],
                    tf_cfg["interval"],
                    limit=N_CANDLES,
                    futures=asset_cfg.get("futures", False),
                )
                self.cache[(asset_key, tf_key)] = deque(rows, maxlen=N_CANDLES)
                logger.info(f"Fetched {len(rows)} candles for {asset_key}/{tf_key}")

        tasks = [
            _one(ak, ac, tk, tc)
            for ak, ac in self.active_assets.items()
            for tk, tc in self.timeframes.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Initial history fetch error: {r}")

    # ── WebSocket connection ─────────────────────────────────────

    async def _run_ws(self, base_url: str, streams: list[str], label: str):
        """Connect to Binance combined WS with auto-reconnect + gap-fill."""
        url = f"{base_url}?streams={'/'.join(streams)}"
        delay = WS_RECONNECT_DELAY

        while self._running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    logger.info(f"WebSocket connected ({label}, {len(streams)} streams)")
                    delay = WS_RECONNECT_DELAY
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._handle_message(data)
                        except Exception:
                            logger.exception("Error handling WS message")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    f"WebSocket disconnected ({label}): {e}. "
                    f"Reconnecting in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                # Re-fetch to fill any candle gaps
                await self._fetch_initial_history()

    # ── Message handling ─────────────────────────────────────────

    async def _handle_message(self, data: dict):
        """Route a single combined-stream kline message."""
        k = data.get("data", {}).get("k")
        if not k:
            return

        symbol: str = k["s"]
        interval: str = k["i"]
        is_closed: bool = k["x"]

        asset_key = self._symbol_to_asset.get(symbol)
        tf_key = self._interval_to_tf.get(interval)
        if asset_key is None or tf_key is None:
            return

        candle = {
            "timestamp": int(k["t"]),
            "Open": float(k["o"]),
            "High": float(k["h"]),
            "Low": float(k["l"]),
            "Close": float(k["c"]),
            "Volume": float(k["v"]),
        }

        cache_key = (asset_key, tf_key)

        if is_closed:
            dq = self.cache.get(cache_key)
            if dq is not None:
                # Replace last entry if same timestamp (was forming), else append
                if dq and dq[-1]["timestamp"] == candle["timestamp"]:
                    dq[-1] = candle
                else:
                    dq.append(candle)
            self.current_candle.pop(cache_key, None)

            if self.on_candle_close:
                df = self.build_df(asset_key, tf_key)
                if df is not None:
                    await self.on_candle_close(asset_key, tf_key, df)
        else:
            self.current_candle[cache_key] = candle

    # ── DataFrame builder ────────────────────────────────────────

    def build_df(self, asset_key: str, tf_key: str) -> pd.DataFrame | None:
        """Build DataFrame with indicators from cache + current forming candle.

        Produces the same columns as fetch_candles():
        timestamp, Open, High, Low, Close, Volume, obv, bb, ema_21, rsi
        """
        cache_key = (asset_key, tf_key)
        dq = self.cache.get(cache_key)
        if not dq or len(dq) < 50:
            return None

        rows = list(dq)
        current = self.current_candle.get(cache_key)
        if current is not None:
            if rows[-1]["timestamp"] != current["timestamp"]:
                rows.append(current)
            else:
                rows[-1] = current

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Compute indicators — must exactly match fetch_candles / data/re_sample.py
        df["obv"] = volume.on_balance_volume(close=df["Close"], volume=df["Volume"], fillna=True)
        df["bb"] = volatility.bollinger_wband(close=df["Close"], window=20, window_dev=2, fillna=True)
        df["ema_21"] = trend.ema_indicator(close=df["Close"], window=21, fillna=True)
        df["rsi"] = momentum.rsi(close=df["Close"], fillna=True)
        df = df.dropna().reset_index(drop=True)

        return df
