"""FastAPI signal server with APScheduler for live SFP detection."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from server.binance import fetch_candles
from server.config import (
    MODEL_PATH,
    N_CANDLES,
    RATIO_THRESHOLD,
    SIGNAL_EXPIRY_BARS,
    SYMBOL,
    TIMEFRAMES,
)
from server.inference import load_model, predict_latest
from server.pipeline import build_features, run_sfp_detection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# In-memory signal store
signals: list[dict] = []
model = None
scheduler = AsyncIOScheduler(timezone=timezone.utc)
last_run: dict[str, str] = {}


async def run_job(tf_key: str):
    """Scheduled job: fetch candles, run pipeline, store signal if hit."""
    cfg = TIMEFRAMES[tf_key]
    logger.info(f"[{tf_key}] Job started")

    try:
        df = await fetch_candles(SYMBOL, cfg["interval"], limit=N_CANDLES)
        logger.info(f"[{tf_key}] Fetched {len(df)} candles")

        actions, swept_levels = run_sfp_detection(df)
        feat_values, actions_trimmed = build_features(df, actions, cfg["tf_hours"])
        logger.info(f"[{tf_key}] Pipeline done — {int((actions_trimmed != 0).sum())} SFPs detected")

        result = predict_latest(model, feat_values)
        if result is None:
            logger.warning(f"[{tf_key}] Not enough data for prediction")
            last_run[tf_key] = datetime.now(timezone.utc).isoformat()
            return

        tp, sl, ratio = result
        last_action = int(actions_trimmed[-1])
        last_swept = float(swept_levels[len(swept_levels) - len(actions_trimmed) + len(actions_trimmed) - 1])

        logger.info(
            f"[{tf_key}] Latest bar — action={last_action}, "
            f"tp={tp:.4f}, sl={sl:.4f}, ratio={ratio:.2f}"
        )

        # Store signal if SFP detected and ratio passes threshold
        if last_action != 0 and ratio > RATIO_THRESHOLD:
            entry = last_swept
            direction = "LONG" if last_action == 1 else "SHORT"

            signal = {
                "timeframe": tf_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": SYMBOL,
                "direction": direction,
                "entry": round(entry, 2),
                "tp_pct": round(tp * 100, 2),
                "sl_pct": round(sl * 100, 2),
                "ratio": round(ratio, 4),
                "bars_remaining": SIGNAL_EXPIRY_BARS,
            }
            signals.append(signal)
            logger.info(f"[{tf_key}] SIGNAL: {direction} @ {entry:.2f} (ratio={ratio:.2f})")

        # Expire old signals for this timeframe
        _expire_signals(tf_key)

        last_run[tf_key] = datetime.now(timezone.utc).isoformat()

    except Exception:
        logger.exception(f"[{tf_key}] Job failed")


def _expire_signals(tf_key: str):
    """Decrement bars_remaining for signals of this timeframe, remove expired."""
    to_remove = []
    for sig in signals:
        if sig["timeframe"] == tf_key:
            sig["bars_remaining"] -= 1
            if sig["bars_remaining"] <= 0:
                to_remove.append(sig)
    for sig in to_remove:
        signals.remove(sig)
        logger.info(f"[{tf_key}] Expired signal: {sig['direction']} @ {sig['entry']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, start scheduler, run all jobs once."""
    global model

    logger.info("Loading model from %s", MODEL_PATH)
    model = load_model(MODEL_PATH)
    logger.info("Model loaded")

    # Schedule jobs
    for tf_key, cfg in TIMEFRAMES.items():
        trigger = CronTrigger(**cfg["cron"], timezone=timezone.utc)
        scheduler.add_job(run_job, trigger, args=[tf_key], id=tf_key, name=f"sfp_{tf_key}")
        logger.info(f"Scheduled {tf_key} job: {cfg['cron']}")

    scheduler.start()
    logger.info("Scheduler started")

    # Run all jobs once immediately on startup
    for tf_key in TIMEFRAMES:
        await run_job(tf_key)

    yield

    scheduler.shutdown()
    logger.info("Scheduler shut down")


app = FastAPI(title="SFP Signal Server", lifespan=lifespan)


@app.get("/signals")
async def get_signals():
    """Return all active signals."""
    return {"signals": signals, "count": len(signals)}


@app.get("/health")
async def health():
    """Server health check."""
    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        jobs.append({"id": job.id, "name": job.name, "next_run": next_run})

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scheduled_jobs": jobs,
        "active_signals": len(signals),
        "last_run": last_run,
    }
