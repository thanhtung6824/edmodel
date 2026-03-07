"""Server configuration constants."""

import os

ASSETS = {
    "btc": {"symbol": "BTCUSDT", "asset_id": 1.0, "active": True},
    "gold": {"symbol": "PAXGUSDT", "asset_id": 2.0, "active": True},
    "silver": {"symbol": "XAGUSDT", "asset_id": 3.0, "active": False},
    "sol": {"symbol": "SOLUSDT", "asset_id": 4.0, "active": True},
    "eth": {"symbol": "ETHUSDT", "asset_id": 5.0, "active": True},
}

N_CANDLES = 1000  # max per Binance API request
BAR_CACHE_DIR = "cache/bars"
BAR_CACHE_MAX = 5000  # max bars to keep per asset/TF
WINDOW_BY_TF = {"15m": 120, "1h": 48, "4h": 30}
WINDOW = max(WINDOW_BY_TF.values())  # 120, used as fallback/max
SIGNAL_EXPIRY_BARS = 3
SIGNAL_HORIZON = 18  # bars to track outcome (matches backtest)

TTP_TRAILING = {
    "early": {"max_ttp": 0.3, "trail_pct": 0.003},   # 0.3% trail
    "mid":   {"max_ttp": 0.6, "trail_pct": 0.006},    # 0.6% trail
    "late":  {"trail_pct": 0.010},                      # 1.0% trail
}

# Liq+Range+SFP model
MODEL_PATH = "best_model_liq_range_sfp.pth"
SCALER_PATH = "liq_range_sfp_scaler.joblib"
MODEL_CONFIDENCE = 0.3  # P(win) threshold
N_FEATURES = 33

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
        "tf_key": "15m",
        # Run every 5min
        "cron": {"minute": "1,6,11,16,21,26,31,36,41,46,51,56"},
    },
    "1h": {
        "interval": "1h",
        "tf_hours": 1.0,
        "tf_key": "1h",
        # Run every 5min to catch partial-candle SFPs early
        "cron": {"minute": "1,6,11,16,21,26,31,36,41,46,51,56"},
    },
    "4h": {
        "interval": "4h",
        "tf_hours": 4.0,
        "tf_key": "4h",
        # Run every 5min to catch partial-candle SFPs early
        "cron": {"minute": "1,6,11,16,21,26,31,36,41,46,51,56"},
    },
}

# Telegram bot — set via environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "-1003745337450"
