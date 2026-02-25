"""Server configuration constants."""

import os

SYMBOL = "BTCUSDT"
N_CANDLES = 500
WINDOW = 30
RATIO_THRESHOLD = 1.4
SIGNAL_EXPIRY_BARS = 3
MODEL_PATH = "best_model_transformer.pth"

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

# Telegram bot — set via environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
