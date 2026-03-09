"""Server configuration constants."""

import os

ASSETS = {
    "btc": {"symbol": "BTCUSDT", "asset_id": 1.0, "active": True},
    "gold": {"symbol": "PAXGUSDT", "asset_id": 2.0, "active": True},
    "silver": {"symbol": "XAGUSDT", "asset_id": 3.0, "active": False},
    "sol": {"symbol": "SOLUSDT", "asset_id": 4.0, "active": True},
    "eth": {"symbol": "ETHUSDT", "asset_id": 5.0, "active": True},
    # Altcoins — use SOL's asset_id (same FiLM conditioning, no retrain needed)
    "doge": {"symbol": "DOGEUSDT", "asset_id": 4.0, "active": True},
    "avax": {"symbol": "AVAXUSDT", "asset_id": 4.0, "active": True},
    "link": {"symbol": "LINKUSDT", "asset_id": 4.0, "active": True},
    "arb": {"symbol": "ARBUSDT", "asset_id": 4.0, "active": True},
    "sui": {"symbol": "SUIUSDT", "asset_id": 4.0, "active": True},
    "tao": {"symbol": "TAOUSDT", "asset_id": 4.0, "active": True},
    "ltc": {"symbol": "LTCUSDT", "asset_id": 4.0, "active": True},
    "tia": {"symbol": "TIAUSDT", "asset_id": 4.0, "active": True},
    "ondo": {"symbol": "ONDOUSDT", "asset_id": 4.0, "active": True},
    "aster": {"symbol": "ASTERUSDT", "asset_id": 4.0, "active": True},
    "sei": {"symbol": "SEIUSDT", "asset_id": 4.0, "active": True},
    "aave": {"symbol": "AAVEUSDT", "asset_id": 4.0, "active": True},
    "bnb": {"symbol": "BNBUSDT", "asset_id": 4.0, "active": True},
    "near": {"symbol": "NEARUSDT", "asset_id": 4.0, "active": True},
    "op": {"symbol": "OPUSDT", "asset_id": 4.0, "active": True},
    "hype": {"symbol": "HYPEUSDT", "asset_id": 4.0, "active": True, "futures": True},
    "pump": {"symbol": "PUMPUSDT", "asset_id": 4.0, "active": True, "futures": True},
    "bch": {"symbol": "BCHUSDT", "asset_id": 4.0, "active": True},
    "zro": {"symbol": "ZROUSDT", "asset_id": 4.0, "active": True},
    "zec": {"symbol": "ZECUSDT", "asset_id": 4.0, "active": True},
}

N_CANDLES = 1000  # max per Binance API request
WINDOW_BY_TF = {"15m": 120, "1h": 48, "4h": 30}
WINDOW = max(WINDOW_BY_TF.values())  # 120, used as fallback/max
SIGNAL_EXPIRY_BARS = 3
SIGNAL_ALERT_MAX_BARS = 2  # max bars_ago to still send Telegram alert
SIGNAL_HORIZON = 18  # bars to track outcome (matches backtest)
HORIZON_BY_TF = {"15m": 36, "1h": 18, "4h": 18}
HIDDEN_DIM = 48

TTP_TRAILING = {
    "early": {"max_ttp": 0.3, "trail_pct": 0.003},   # 0.3% trail
    "mid":   {"max_ttp": 0.6, "trail_pct": 0.006},    # 0.6% trail
    "late":  {"trail_pct": 0.010},                      # 1.0% trail
}

# Liq+Range+SFP model
MODEL_PATH = "best_model_liq_range_sfp.pth"
SCALER_PATH = "liq_range_sfp_scaler.joblib"
MODEL_CONFIDENCE = 0.3  # P(win) threshold
N_FEATURES = 27
ENSEMBLE_MODEL_PATTERN = "best_model_liq_range_sfp_*.pth"

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
