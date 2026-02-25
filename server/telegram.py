"""Telegram bot notifications for SFP signals."""

import logging

import httpx

from server.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


async def send_signal_alert(signal: dict):
    """Send a Telegram message when a new signal fires."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping notification")
        return

    direction = signal["direction"]
    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    entry = signal["entry"]
    tp_pct = signal["tp_pct"]
    sl_pct = signal["sl_pct"]

    is_long = direction == "LONG"
    tp_price = entry * (1 + tp_pct / 100) if is_long else entry * (1 - tp_pct / 100)
    sl_price = entry * (1 - sl_pct / 100) if is_long else entry * (1 + sl_pct / 100)

    text = (
        f"{emoji} <b>SFP Signal — {signal['timeframe']} {signal['symbol']}</b>\n"
        f"\n"
        f"Direction: <b>{direction}</b>\n"
        f"Entry: <code>${entry:,.2f}</code>\n"
        f"TP: <code>${tp_price:,.2f}</code> (+{tp_pct}%)\n"
        f"SL: <code>${sl_price:,.2f}</code> (-{sl_pct}%)\n"
        f"Ratio: <b>{signal['ratio']}</b>\n"
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
