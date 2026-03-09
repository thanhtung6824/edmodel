"""Telegram bot notifications for trading signals."""

import logging

import httpx

from server.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def _fmt(price: float) -> str:
    """Format price with enough decimals for sub-penny coins."""
    if price >= 1.0:
        return f"${price:,.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


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
            f"TP1: <code>{_fmt(tp1_price)}</code> (+{tp1_pct}%)\n"
            f"TP2: <code>{_fmt(tp2_price)}</code> (+{tp2_pct}%)"
        )
        quality_line = f"P(win): <b>{confidence:.0%}</b> | R:R: <b>{ratio:.1f}:1</b> ({best_tp})"
    else:
        # Legacy single-TP format
        tp_price = signal["tp_price"]
        tp_pct = signal["tp_pct"]
        tp_lines = f"TP: <code>{_fmt(tp_price)}</code> (+{tp_pct}%)"
        quality_line = f"Ratio: <b>{signal.get('ratio', 0)}</b>"

    text = (
        f"{emoji} <b>{strategy} Signal — {signal['timeframe']} {signal['symbol']}</b>\n"
        f"\n"
        f"Direction: <b>{direction}</b>\n"
        f"Entry: <code>{_fmt(entry)}</code>\n"
        f"{tp_lines}\n"
        f"SL: <code>{_fmt(sl_price)}</code> (-{sl_pct}%)\n"
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


async def send_trade_update(signal: dict, event_type: str):
    """Send a Telegram notification for trade status changes."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping trade update")
        return

    direction = signal["direction"]
    symbol = signal["symbol"]
    tf = signal["timeframe"]
    entry = signal["entry"]
    header = f"{tf} {symbol} {direction}"

    if event_type == "tp1_hit":
        trail_pct = signal.get("trail_pct", 0.6)
        tp1_price = signal.get("tp1_price", signal.get("tp_price"))
        tp2_price = signal.get("tp2_price", signal.get("tp_price"))
        tp1_pct = abs(tp1_price - entry) / (entry + 1e-8) * 100
        text = (
            f"\U0001f7e1 <b>TP1 Hit — {header}</b>\n"
            f"\n"
            f"Entry: <code>{_fmt(entry)}</code>\n"
            f"TP1: <code>{_fmt(tp1_price)}</code> (+{tp1_pct:.1f}%) \u2705\n"
            f"\u2192 Move SL to breakeven: <code>{_fmt(entry)}</code>\n"
            f"Trailing stop active ({trail_pct*100:.1f}%)\n"
            f"Remaining target: TP2 <code>{_fmt(tp2_price)}</code>"
        )
    elif event_type == "sl_hit":
        sl_price = signal["sl_price"]
        actual_r = signal.get("actual_r", -1.0)
        text = (
            f"\U0001f534 <b>Stop Loss — {header}</b>\n"
            f"\n"
            f"Entry: <code>{_fmt(entry)}</code>\n"
            f"SL: <code>{_fmt(sl_price)}</code>\n"
            f"Result: {actual_r:+.1f}R \u274c"
        )
    elif event_type == "tp2_hit":
        tp2_price = signal.get("tp2_price", signal.get("tp_price"))
        actual_r = signal.get("actual_r", 0)
        text = (
            f"\U0001f7e2 <b>TP2 Hit — {header}</b>\n"
            f"\n"
            f"Entry: <code>{_fmt(entry)}</code>\n"
            f"TP2: <code>{_fmt(tp2_price)}</code> \u2705\n"
            f"Result: {actual_r:+.1f}R \U0001f3c6"
        )
    elif event_type == "trail_stop":
        exit_price = signal.get("exit_price", 0)
        actual_r = signal.get("actual_r", 0)
        text = (
            f"\U0001f7e1 <b>Trail Stop — {header}</b>\n"
            f"\n"
            f"Entry: <code>{_fmt(entry)}</code>\n"
            f"Trailed at: <code>{_fmt(exit_price)}</code>\n"
            f"Result: {actual_r:+.1f}R \u2705"
        )
    elif event_type == "horizon_expired":
        exit_price = signal.get("exit_price", 0)
        actual_r = signal.get("actual_r", 0)
        text = (
            f"\u23f0 <b>Expired — {header}</b>\n"
            f"\n"
            f"Entry: <code>{_fmt(entry)}</code>\n"
            f"Closed at: <code>{_fmt(exit_price)}</code>\n"
            f"Result: {actual_r:+.1f}R"
        )
    else:
        logger.warning(f"Unknown trade event type: {event_type}")
        return

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
        logger.info(f"Telegram trade update sent: {event_type}")
    except Exception:
        logger.exception(f"Failed to send Telegram trade update: {event_type}")
