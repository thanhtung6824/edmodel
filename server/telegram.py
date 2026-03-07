"""Telegram bot notifications for trading signals."""

import logging

import httpx

from server.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


async def send_signal_alert(signal: dict):
    """Send a Telegram message when a new signal fires."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping notification")
        return

    strategy = signal.get("strategy", "SFP")
    direction = signal["direction"]
    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    entry = signal["entry"]
    sl_price = signal["sl_price"]
    sl_pct = signal["sl_pct"]

    # Multi-TP format
    tp1_price = signal.get("tp1_price")
    if tp1_price is not None:
        tp2_price = signal["tp2_price"]
        tp1_pct = signal["tp1_pct"]
        tp2_pct = signal["tp2_pct"]
        confidence = signal.get("confidence", 0)
        best_tp = signal.get("best_tp", "TP1")
        ratio = signal.get("ratio", 0)

        tp_lines = (
            f"TP1: <code>${tp1_price:,.2f}</code> (+{tp1_pct}%)\n"
            f"TP2: <code>${tp2_price:,.2f}</code> (+{tp2_pct}%)"
        )
        quality_line = f"P(win): <b>{confidence:.0%}</b> | R:R: <b>{ratio:.1f}:1</b> ({best_tp})"
    else:
        # Legacy single-TP format
        tp_price = signal["tp_price"]
        tp_pct = signal["tp_pct"]
        tp_lines = f"TP: <code>${tp_price:,.2f}</code> (+{tp_pct}%)"
        quality_line = f"Ratio: <b>{signal.get('ratio', 0)}</b>"

    text = (
        f"{emoji} <b>{strategy} Signal — {signal['timeframe']} {signal['symbol']}</b>\n"
        f"\n"
        f"Direction: <b>{direction}</b>\n"
        f"Entry: <code>${entry:,.2f}</code>\n"
        f"{tp_lines}\n"
        f"SL: <code>${sl_price:,.2f}</code> (-{sl_pct}%)\n"
        f"{quality_line}\n"
        f"\n"
        f"Bars remaining: {signal['bars_remaining']}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        logger.info("Telegram notification sent")
    except Exception:
        logger.exception("Failed to send Telegram notification")
