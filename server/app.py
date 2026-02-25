"""FastAPI signal server with APScheduler for live SFP detection."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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
from server.telegram import send_signal_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# In-memory stores
signals: list[dict] = []
candle_store: dict[str, list[dict]] = {}  # tf_key -> [{time, open, high, low, close}, ...]
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

        # Store candle data for chart endpoint
        candle_store[tf_key] = [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
            }
            for _, row in df.iterrows()
        ]

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

            # Send Telegram notification
            await send_signal_alert(signal)

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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live signal dashboard with chart."""
    return DASHBOARD_HTML


@app.get("/signals")
async def get_signals():
    """Return all active signals."""
    return {"signals": signals, "count": len(signals)}


@app.get("/candles/{tf}")
async def get_candles(tf: str):
    """Return cached candle data for a timeframe."""
    if tf not in candle_store:
        return {"candles": [], "error": f"No data for {tf}"}
    return {"candles": candle_store[tf]}


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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFP Signal Server</title>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
  .header { padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #21262d; }
  .header h1 { color: #58a6ff; font-size: 18px; font-weight: 600; }
  .status { color: #8b949e; font-size: 12px; }
  .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-right: 4px; vertical-align: middle; }
  .main { display: flex; height: calc(100vh - 57px); }
  .chart-panel { flex: 1; display: flex; flex-direction: column; }
  .tf-bar { display: flex; gap: 4px; padding: 8px 16px; background: #161b22; border-bottom: 1px solid #21262d; }
  .tf-btn { padding: 6px 16px; border: 1px solid #30363d; border-radius: 6px; background: transparent; color: #8b949e; cursor: pointer; font-size: 13px; font-weight: 600; transition: all .15s; }
  .tf-btn:hover { border-color: #58a6ff; color: #c9d1d9; }
  .tf-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  #chart-container { flex: 1; }
  .sidebar { width: 340px; background: #161b22; border-left: 1px solid #21262d; overflow-y: auto; }
  .section { padding: 16px; border-bottom: 1px solid #21262d; }
  .section h2 { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; font-weight: 600; }
  .signal-card { background: #1c2333; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 3px solid #30363d; }
  .signal-card.long { border-left-color: #3fb950; }
  .signal-card.short { border-left-color: #f85149; }
  .sig-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .sig-dir { font-weight: 700; font-size: 14px; }
  .sig-dir.long { color: #3fb950; }
  .sig-dir.short { color: #f85149; }
  .sig-badge { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; background: #1f2937; color: #58a6ff; }
  .sig-meta { color: #484f58; font-size: 11px; margin-bottom: 8px; }
  .sig-levels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .sig-levels .lbl { color: #8b949e; font-size: 10px; }
  .sig-levels .val { font-size: 13px; font-weight: 600; font-family: 'SF Mono', monospace; }
  .sig-levels .val.entry { color: #58a6ff; }
  .sig-levels .val.tp { color: #3fb950; }
  .sig-levels .val.sl { color: #f85149; }
  .sig-ratio { margin-top: 6px; font-size: 11px; color: #8b949e; }
  .empty { color: #484f58; font-size: 13px; padding: 24px 0; text-align: center; }
  .job-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 12px; border-bottom: 1px solid #21262d; }
  .job-row:last-child { border-bottom: none; }
  .job-tf { color: #c9d1d9; font-weight: 600; }
  .job-next { color: #8b949e; }
</style>
</head>
<body>
<div class="header">
  <h1>SFP Signal Server</h1>
  <div class="status" id="status"><span class="dot"></span>Loading...</div>
</div>
<div class="main">
  <div class="chart-panel">
    <div class="tf-bar">
      <button class="tf-btn active" data-tf="15m">15m</button>
      <button class="tf-btn" data-tf="1h">1h</button>
      <button class="tf-btn" data-tf="4h">4h</button>
    </div>
    <div id="chart-container"></div>
  </div>
  <div class="sidebar">
    <div class="section">
      <h2>Active Signals</h2>
      <div id="signals"><div class="empty">Loading...</div></div>
    </div>
    <div class="section">
      <h2>Scheduled Jobs</h2>
      <div id="jobs"><div class="empty">Loading...</div></div>
    </div>
  </div>
</div>
<script>
// ─── Chart setup ───
const container = document.getElementById('chart-container');
const chart = LightweightCharts.createChart(container, {
  layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
  grid: { vertLines: { color: '#1b2028' }, horzLines: { color: '#1b2028' } },
  crosshair: { mode: 0 },
  rightPriceScale: { borderColor: '#21262d' },
  timeScale: { borderColor: '#21262d', timeVisible: true, secondsVisible: false },
});

const candleSeries = chart.addCandlestickSeries({
  upColor: '#3fb950', downColor: '#f85149',
  borderUpColor: '#3fb950', borderDownColor: '#f85149',
  wickUpColor: '#3fb950', wickDownColor: '#f85149',
});

// Signal lines stored here for cleanup
let signalLines = [];

function clearSignalLines() {
  signalLines.forEach(l => candleSeries.removePriceLine(l));
  signalLines = [];
}

function drawSignalLines(signals, currentTf) {
  clearSignalLines();
  signals.forEach(s => {
    if (s.timeframe !== currentTf) return;
    const isLong = s.direction === 'LONG';
    const entry = s.entry;
    const tpPrice = isLong ? entry * (1 + s.tp_pct / 100) : entry * (1 - s.tp_pct / 100);
    const slPrice = isLong ? entry * (1 - s.sl_pct / 100) : entry * (1 + s.sl_pct / 100);

    signalLines.push(candleSeries.createPriceLine({
      price: entry, color: '#2962ff', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true, title: s.direction + ' Entry',
    }));
    signalLines.push(candleSeries.createPriceLine({
      price: tpPrice, color: '#3fb950', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'TP +' + s.tp_pct + '%',
    }));
    signalLines.push(candleSeries.createPriceLine({
      price: slPrice, color: '#f85149', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'SL -' + s.sl_pct + '%',
    }));
  });
}

// ─── State ───
let currentTf = '15m';
let cachedSignals = [];

// ─── Timeframe buttons ───
document.querySelectorAll('.tf-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTf = btn.dataset.tf;
    loadCandles(currentTf);
  });
});

async function loadCandles(tf) {
  try {
    const res = await fetch('/candles/' + tf);
    const data = await res.json();
    if (data.candles && data.candles.length > 0) {
      candleSeries.setData(data.candles);
      chart.timeScale().fitContent();
      drawSignalLines(cachedSignals, tf);
    }
  } catch(e) { console.error('Failed to load candles:', e); }
}

async function refreshSignals() {
  try {
    const [sigRes, hpRes] = await Promise.all([
      fetch('/signals'), fetch('/health')
    ]);
    const sigData = await sigRes.json();
    const hpData = await hpRes.json();
    cachedSignals = sigData.signals;

    // Draw signal lines on chart
    drawSignalLines(cachedSignals, currentTf);

    // Render signal cards
    const el = document.getElementById('signals');
    if (cachedSignals.length === 0) {
      el.innerHTML = '<div class="empty">No active signals</div>';
    } else {
      el.innerHTML = cachedSignals.map(s => {
        const cls = s.direction.toLowerCase();
        const isLong = s.direction === 'LONG';
        const tp = isLong ? s.entry * (1 + s.tp_pct / 100) : s.entry * (1 - s.tp_pct / 100);
        const sl = isLong ? s.entry * (1 - s.sl_pct / 100) : s.entry * (1 + s.sl_pct / 100);
        return '<div class="signal-card ' + cls + '">' +
          '<div class="sig-header">' +
            '<span class="sig-dir ' + cls + '">' + s.direction + ' · ' + s.timeframe + '</span>' +
            '<span class="sig-badge">' + s.bars_remaining + ' bars</span>' +
          '</div>' +
          '<div class="sig-meta">' + s.timestamp.replace('T', ' ').split('.')[0] + ' UTC</div>' +
          '<div class="sig-levels">' +
            '<div><div class="lbl">Entry</div><div class="val entry">$' + s.entry.toLocaleString() + '</div></div>' +
            '<div><div class="lbl">TP</div><div class="val tp">$' + tp.toFixed(2) + '</div></div>' +
            '<div><div class="lbl">SL</div><div class="val sl">$' + sl.toFixed(2) + '</div></div>' +
          '</div>' +
          '<div class="sig-ratio">Ratio: ' + s.ratio + ' · TP +' + s.tp_pct + '% · SL -' + s.sl_pct + '%</div>' +
        '</div>';
      }).join('');
    }

    // Render jobs
    const jEl = document.getElementById('jobs');
    jEl.innerHTML = hpData.scheduled_jobs.map(j => {
      const next = j.next_run ? new Date(j.next_run).toUTCString().slice(5, -4) : 'N/A';
      return '<div class="job-row"><span class="job-tf">' + j.id + '</span><span class="job-next">next: ' + next + '</span></div>';
    }).join('');

    document.getElementById('status').innerHTML =
      '<span class="dot"></span>' + hpData.active_signals + ' signal(s) · ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('status').innerHTML = '<span class="dot" style="background:#f85149"></span>Connection error';
  }
}

// ─── Resize handler ───
new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight }))
  .observe(container);

// ─── Init ───
loadCandles(currentTf);
refreshSignals();
setInterval(refreshSignals, 15000);
setInterval(() => loadCandles(currentTf), 60000);
</script>
</body>
</html>"""
