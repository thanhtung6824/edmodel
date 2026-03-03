"""Server configuration constants."""

import os

ASSETS = {
    "btc": {"symbol": "BTCUSDT", "asset_id": 1.0, "active": True},
    "gold": {"symbol": "PAXGUSDT", "asset_id": 2.0, "active": True},
    "silver": {"symbol": "XAGUSDT", "asset_id": 3.0, "active": False},
    "sol": {"symbol": "SOLUSDT", "asset_id": 4.0, "active": True},
    "eth": {"symbol": "ETHUSDT", "asset_id": 5.0, "active": True},
}

N_CANDLES = 500
WINDOW = 30
RATIO_THRESHOLD = 1.4
SIGNAL_EXPIRY_BARS = 3
SIGNAL_HORIZON = 18  # bars to track outcome (matches backtest)
MODEL_PATH = "best_model_transformer.pth"
LIVE_SIGNALS_PATH = "signals_live.json"

# Past signal files (from backtest validation)
HISTORY_FILES = {
    "15m": "signals_15min.json",
    "1h": "signals_1h.json",
    "4h": "signals_4h.json",
}

# Timeframe configs: (binance interval, tf_hours, cron kwargs for APScheduler)
TIMEFRAMES = {
    "15m": {
        "interval": "15m",
        "tf_hours": 0.25,
        # Run at :01, :16, :31, :46 UTC (1 min after candle close)
        "cron": {"minute": "1,16,31,46"},
    },
    "1h": {
        "interval": "1h",
        "tf_hours": 1.0,
        # Run at :01 past each hour
        "cron": {"minute": "1"},
    },
    "4h": {
        "interval": "4h",
        "tf_hours": 4.0,
        # Run at 0:01, 4:01, 8:01, 12:01, 16:01, 20:01 UTC
        "cron": {"hour": "0,4,8,12,16,20", "minute": "1"},
    },
}

# Median TP/SL from training data — used to denormalize model predictions.
# Model outputs normalized values (1.0 = median). Multiply by these to get actual fractions.
# Key format: (asset, tf_key)
MEDIAN_TP_SL = {
    ("btc", "15m"): (0.00824, 0.00463),
    ("btc", "1h"):  (0.01682, 0.00987),
    ("btc", "4h"):  (0.03512, 0.02176),
    ("sol", "15m"): (0.01656, 0.01005),
    ("sol", "1h"):  (0.03388, 0.02258),
    ("sol", "4h"):  (0.06991, 0.04339),
    ("eth", "15m"): (0.01077, 0.00627),
    ("eth", "1h"):  (0.02289, 0.01373),
    ("eth", "4h"):  (0.04750, 0.02684),
    ("gold", "15m"): (0.00270, 0.00163),
    ("gold", "1h"):  (0.00585, 0.00356),
    ("gold", "4h"):  (0.01239, 0.00791),
}

# Telegram bot — set via environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
