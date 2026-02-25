"""FastAPI signal server with APScheduler for live SFP detection."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from server.binance import fetch_candles
from server.config import (
    HISTORY_FILES,
    LIVE_SIGNALS_PATH,
    MODEL_PATH,
    N_CANDLES,
    RATIO_THRESHOLD,
    SIGNAL_EXPIRY_BARS,
    SIGNAL_HORIZON,
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
active_signals: list[dict] = []  # signals with bars_remaining > 0
live_signals: list[dict] = []  # all live signals (persisted to JSON)
candle_store: dict[str, list[dict]] = {}  # tf_key -> [{time, open, high, low, close}, ...]
model = None
scheduler = AsyncIOScheduler(timezone=timezone.utc)
last_run: dict[str, str] = {}


def _load_live_signals():
    """Load persisted live signals from JSON on startup."""
    path = Path(LIVE_SIGNALS_PATH)
    if path.exists():
        data = json.loads(path.read_text())
        live_signals.extend(data)
        logger.info(f"Loaded {len(data)} live signals from {LIVE_SIGNALS_PATH}")


def _save_live_signals():
    """Persist live signals to JSON."""
    Path(LIVE_SIGNALS_PATH).write_text(json.dumps(live_signals, indent=2))


def _resolve_open_signals(tf_key: str, candles: list[dict]):
    """Check open live signals against candle data to determine win/loss."""
    changed = False
    for sig in live_signals:
        if sig["timeframe"] != tf_key or sig["status"] != "open":
            continue

        entry = sig["entry"]
        tp_price = sig["tp_price"]
        sl_price = sig["sl_price"]
        sig_time = sig["time"]
        is_long = sig["direction"] == "LONG"

        # Count bars after signal and check TP/SL hit
        bars_after = 0
        for c in candles:
            if c["time"] <= sig_time:
                continue
            bars_after += 1

            if is_long:
                if c["high"] >= tp_price:
                    sig["status"] = "win"
                    sig["actual_r"] = round((tp_price - entry) / (entry - sl_price + 1e-8), 2)
                    sig["resolved_at"] = c["time"]
                    changed = True
                    break
                if c["low"] <= sl_price:
                    sig["status"] = "loss"
                    sig["actual_r"] = -1.0
                    sig["resolved_at"] = c["time"]
                    changed = True
                    break
            else:
                if c["low"] <= tp_price:
                    sig["status"] = "win"
                    sig["actual_r"] = round((entry - tp_price) / (sl_price - entry + 1e-8), 2)
                    sig["resolved_at"] = c["time"]
                    changed = True
                    break
                if c["high"] >= sl_price:
                    sig["status"] = "loss"
                    sig["actual_r"] = -1.0
                    sig["resolved_at"] = c["time"]
                    changed = True
                    break

            if bars_after >= SIGNAL_HORIZON:
                # Expired without hitting TP or SL — mark by close P&L
                last_close = c["close"]
                if is_long:
                    pnl = (last_close - entry) / (entry - sl_price + 1e-8)
                else:
                    pnl = (entry - last_close) / (sl_price - entry + 1e-8)
                sig["status"] = "win" if pnl > 0 else "loss"
                sig["actual_r"] = round(pnl, 2)
                sig["resolved_at"] = c["time"]
                changed = True
                break

    if changed:
        _save_live_signals()


async def run_job(tf_key: str):
    """Scheduled job: fetch candles, run pipeline, store signal if hit."""
    cfg = TIMEFRAMES[tf_key]
    logger.info(f"[{tf_key}] Job started")

    try:
        df = await fetch_candles(SYMBOL, cfg["interval"], limit=N_CANDLES)
        logger.info(f"[{tf_key}] Fetched {len(df)} candles")

        # Store candle data for chart endpoint
        candles = [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
            }
            for _, row in df.iterrows()
        ]
        candle_store[tf_key] = candles

        # Resolve any open live signals
        _resolve_open_signals(tf_key, candles)

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
            is_long = last_action == 1
            tp_price = entry * (1 + tp) if is_long else entry * (1 - tp)
            sl_price = entry * (1 - sl) if is_long else entry * (1 + sl)

            signal = {
                "timeframe": tf_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "time": candles[-1]["time"],
                "symbol": SYMBOL,
                "direction": direction,
                "entry": round(entry, 2),
                "tp_pct": round(tp * 100, 2),
                "sl_pct": round(sl * 100, 2),
                "tp_price": round(tp_price, 2),
                "sl_price": round(sl_price, 2),
                "ratio": round(ratio, 4),
                "bars_remaining": SIGNAL_EXPIRY_BARS,
                "status": "open",
                "actual_r": None,
                "resolved_at": None,
            }
            active_signals.append(signal)
            live_signals.append(signal)
            _save_live_signals()
            logger.info(f"[{tf_key}] SIGNAL: {direction} @ {entry:.2f} (ratio={ratio:.2f})")

            await send_signal_alert(signal)

        # Expire old active signals for this timeframe
        _expire_active_signals(tf_key)

        last_run[tf_key] = datetime.now(timezone.utc).isoformat()

    except Exception:
        logger.exception(f"[{tf_key}] Job failed")


def _expire_active_signals(tf_key: str):
    """Decrement bars_remaining for active signals, remove expired from active list."""
    to_remove = []
    for sig in active_signals:
        if sig["timeframe"] == tf_key:
            sig["bars_remaining"] -= 1
            if sig["bars_remaining"] <= 0:
                to_remove.append(sig)
    for sig in to_remove:
        active_signals.remove(sig)
        logger.info(f"[{tf_key}] Expired active signal: {sig['direction']} @ {sig['entry']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, start scheduler, run all jobs once."""
    global model

    _load_live_signals()

    logger.info("Loading model from %s", MODEL_PATH)
    model = load_model(MODEL_PATH)
    logger.info("Model loaded")

    for tf_key, cfg in TIMEFRAMES.items():
        trigger = CronTrigger(**cfg["cron"], timezone=timezone.utc)
        scheduler.add_job(run_job, trigger, args=[tf_key], id=tf_key, name=f"sfp_{tf_key}")
        logger.info(f"Scheduled {tf_key} job: {cfg['cron']}")

    scheduler.start()
    logger.info("Scheduler started")

    for tf_key in TIMEFRAMES:
        await run_job(tf_key)

    yield

    scheduler.shutdown()
    logger.info("Scheduler shut down")


app = FastAPI(title="SFP Signal Server", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/signals")
async def get_signals():
    """Return active signals (bars_remaining > 0)."""
    return {"signals": active_signals, "count": len(active_signals)}


@app.get("/signals/live")
async def get_live_signals():
    """Return all live signals with outcomes and stats."""
    resolved = [s for s in live_signals if s["status"] != "open"]
    wins = [s for s in resolved if s["status"] == "win"]
    losses = [s for s in resolved if s["status"] == "loss"]
    open_count = sum(1 for s in live_signals if s["status"] == "open")
    total_r = sum(s["actual_r"] for s in resolved if s["actual_r"] is not None)

    return {
        "signals": live_signals,
        "stats": {
            "total": len(live_signals),
            "open": open_count,
            "resolved": len(resolved),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
            "total_r": round(total_r, 2),
        },
    }


@app.get("/signals/history/{tf}")
async def get_history(tf: str):
    """Return past backtest signals for chart markers."""
    filename = HISTORY_FILES.get(tf)
    if not filename:
        return {"signals": []}
    path = Path(filename)
    if not path.exists():
        return {"signals": []}
    data = json.loads(path.read_text())
    return {"signals": data.get("signals", [])}


@app.get("/candles/{tf}")
async def get_candles(tf: str):
    if tf not in candle_store:
        return {"candles": [], "error": f"No data for {tf}"}
    return {"candles": candle_store[tf]}


@app.get("/health")
async def health():
    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        jobs.append({"id": job.id, "name": job.name, "next_run": next_run})

    resolved = [s for s in live_signals if s["status"] != "open"]
    wins = sum(1 for s in resolved if s["status"] == "win")

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scheduled_jobs": jobs,
        "active_signals": len(active_signals),
        "live_stats": {
            "total": len(live_signals),
            "resolved": len(resolved),
            "wins": wins,
            "win_rate": round(wins / len(resolved) * 100, 1) if resolved else 0,
        },
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
  .header { padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #21262d; }
  .header h1 { color: #58a6ff; font-size: 18px; font-weight: 600; }
  .status { color: #8b949e; font-size: 12px; }
  .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-right: 4px; vertical-align: middle; }
  .main { display: flex; height: calc(100vh - 49px); }
  .chart-panel { flex: 1; display: flex; flex-direction: column; }
  .tf-bar { display: flex; gap: 4px; padding: 8px 16px; background: #161b22; border-bottom: 1px solid #21262d; align-items: center; }
  .tf-btn { padding: 5px 14px; border: 1px solid #30363d; border-radius: 6px; background: transparent; color: #8b949e; cursor: pointer; font-size: 13px; font-weight: 600; transition: all .15s; }
  .tf-btn:hover { border-color: #58a6ff; color: #c9d1d9; }
  .tf-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .tf-toggle { margin-left: 16px; display: flex; gap: 4px; align-items: center; }
  .tf-toggle label { font-size: 12px; color: #8b949e; cursor: pointer; user-select: none; }
  .tf-toggle input { cursor: pointer; }
  #chart-container { flex: 1; }
  .sidebar { width: 360px; background: #161b22; border-left: 1px solid #21262d; overflow-y: auto; }
  .section { padding: 16px; border-bottom: 1px solid #21262d; }
  .section h2 { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600; }
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
  .stat-box { background: #1c2333; border-radius: 6px; padding: 10px; text-align: center; }
  .stat-box .val { font-size: 20px; font-weight: 700; }
  .stat-box .val.green { color: #3fb950; }
  .stat-box .val.red { color: #f85149; }
  .stat-box .val.blue { color: #58a6ff; }
  .stat-box .lbl { font-size: 10px; color: #8b949e; margin-top: 2px; }
  .signal-card { background: #1c2333; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 3px solid #30363d; }
  .signal-card.long { border-left-color: #3fb950; }
  .signal-card.short { border-left-color: #f85149; }
  .sig-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .sig-dir { font-weight: 700; font-size: 14px; }
  .sig-dir.long { color: #3fb950; }
  .sig-dir.short { color: #f85149; }
  .sig-badge { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }
  .sig-badge.open { background: #1f2937; color: #58a6ff; }
  .sig-badge.win { background: #0d2818; color: #3fb950; }
  .sig-badge.loss { background: #2d1215; color: #f85149; }
  .sig-meta { color: #484f58; font-size: 11px; margin-bottom: 6px; }
  .sig-levels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .sig-levels .lbl { color: #8b949e; font-size: 10px; }
  .sig-levels .val { font-size: 13px; font-weight: 600; font-family: 'SF Mono', monospace; }
  .sig-levels .val.entry { color: #58a6ff; }
  .sig-levels .val.tp { color: #3fb950; }
  .sig-levels .val.sl { color: #f85149; }
  .sig-ratio { margin-top: 4px; font-size: 11px; color: #8b949e; }
  .empty { color: #484f58; font-size: 13px; padding: 20px 0; text-align: center; }
  .job-row { display: flex; justify-content: space-between; padding: 5px 0; font-size: 12px; border-bottom: 1px solid #21262d; }
  .job-row:last-child { border-bottom: none; }
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
      <div class="tf-toggle">
        <input type="checkbox" id="showHistory" checked>
        <label for="showHistory">Show past signals</label>
      </div>
    </div>
    <div id="chart-container"></div>
  </div>
  <div class="sidebar">
    <div class="section">
      <h2>Live Performance</h2>
      <div class="stats-grid" id="stats">
        <div class="stat-box"><div class="val blue" id="st-total">-</div><div class="lbl">Signals</div></div>
        <div class="stat-box"><div class="val green" id="st-wr">-</div><div class="lbl">Win Rate</div></div>
        <div class="stat-box"><div class="val" id="st-r">-</div><div class="lbl">Total R</div></div>
      </div>
    </div>
    <div class="section">
      <h2>Active Signals</h2>
      <div id="active-signals"><div class="empty">Loading...</div></div>
    </div>
    <div class="section">
      <h2>Signal History (Live)</h2>
      <div id="live-signals"><div class="empty">Loading...</div></div>
    </div>
    <div class="section">
      <h2>Scheduled Jobs</h2>
      <div id="jobs"><div class="empty">Loading...</div></div>
    </div>
  </div>
</div>
<script>
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

let signalLines = [];
let currentTf = '15m';
let cachedActive = [];
let cachedHistory = {};
let cachedLive = [];

function clearSignalLines() {
  signalLines.forEach(l => candleSeries.removePriceLine(l));
  signalLines = [];
}

function drawActiveSignalLines(sigs, tf) {
  clearSignalLines();
  sigs.forEach(s => {
    if (s.timeframe !== tf) return;
    const isLong = s.direction === 'LONG';
    signalLines.push(candleSeries.createPriceLine({
      price: s.entry, color: '#2962ff', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true, title: s.direction + ' Entry',
    }));
    signalLines.push(candleSeries.createPriceLine({
      price: s.tp_price, color: '#3fb950', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'TP',
    }));
    signalLines.push(candleSeries.createPriceLine({
      price: s.sl_price, color: '#f85149', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'SL',
    }));
  });
}

function buildMarkers(tf) {
  const markers = [];
  const showHist = document.getElementById('showHistory').checked;
  // Only show markers within candle range
  const candles = candleSeries.data ? candleSeries.data() : [];
  const minTime = candles.length > 0 ? candles[0].time : 0;

  // Past backtest signals
  if (showHist && cachedHistory[tf]) {
    cachedHistory[tf].forEach(s => {
      const t = Math.floor(s.time_ms / 1000);
      if (t < minTime) return;
      const isLong = s.dir === 1;
      markers.push({
        time: t,
        position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#3fb950' : '#f85149',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: isLong ? 'L' : 'S',
      });
    });
  }

  // Live signals
  cachedLive.forEach(s => {
    if (s.timeframe !== tf || !s.time || s.time < minTime) return;
    const isLong = s.direction === 'LONG';
    markers.push({
      time: s.time,
      position: isLong ? 'belowBar' : 'aboveBar',
      color: isLong ? '#3fb950' : '#f85149',
      shape: isLong ? 'arrowUp' : 'arrowDown',
      text: isLong ? 'L' : 'S',
    });
  });

  markers.sort((a, b) => a.time - b.time);
  return markers;
}

// TF buttons
document.querySelectorAll('.tf-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTf = btn.dataset.tf;
    loadChart(currentTf);
  });
});
document.getElementById('showHistory').addEventListener('change', () => {
  candleSeries.setMarkers(buildMarkers(currentTf));
});

async function loadChart(tf) {
  try {
    const [candleRes, histRes] = await Promise.all([
      fetch('/candles/' + tf),
      cachedHistory[tf] ? Promise.resolve(null) : fetch('/signals/history/' + tf),
    ]);
    const candleData = await candleRes.json();
    if (histRes) {
      const histData = await histRes.json();
      cachedHistory[tf] = histData.signals || [];
    }
    if (candleData.candles && candleData.candles.length > 0) {
      candleSeries.setData(candleData.candles);
      candleSeries.setMarkers(buildMarkers(tf));
      chart.timeScale().fitContent();
      drawActiveSignalLines(cachedActive, tf);
    }
  } catch(e) { console.error('Failed to load chart:', e); }
}

function renderSignalCard(s, showStatus) {
  const cls = s.direction.toLowerCase();
  const statusCls = s.status || 'open';
  const statusText = s.status === 'open' ? s.bars_remaining + ' bars'
    : s.status === 'win' ? 'WIN +' + (s.actual_r || 0) + 'R'
    : 'LOSS ' + (s.actual_r || 0) + 'R';
  return '<div class="signal-card ' + cls + '">' +
    '<div class="sig-header">' +
      '<span class="sig-dir ' + cls + '">' + s.direction + ' · ' + s.timeframe + '</span>' +
      '<span class="sig-badge ' + statusCls + '">' + statusText + '</span>' +
    '</div>' +
    '<div class="sig-meta">' + (s.timestamp || '').replace('T', ' ').split('.')[0] + ' UTC</div>' +
    '<div class="sig-levels">' +
      '<div><div class="lbl">Entry</div><div class="val entry">$' + (s.entry || 0).toLocaleString() + '</div></div>' +
      '<div><div class="lbl">TP</div><div class="val tp">$' + (s.tp_price || 0).toLocaleString() + '</div></div>' +
      '<div><div class="lbl">SL</div><div class="val sl">$' + (s.sl_price || 0).toLocaleString() + '</div></div>' +
    '</div>' +
    '<div class="sig-ratio">Ratio: ' + s.ratio + '</div>' +
  '</div>';
}

async function refreshData() {
  try {
    const [actRes, liveRes, hpRes] = await Promise.all([
      fetch('/signals'), fetch('/signals/live'), fetch('/health')
    ]);
    const actData = await actRes.json();
    const liveData = await liveRes.json();
    const hpData = await hpRes.json();
    cachedActive = actData.signals;
    cachedLive = liveData.signals;
    const stats = liveData.stats;

    drawActiveSignalLines(cachedActive, currentTf);
    candleSeries.setMarkers(buildMarkers(currentTf));

    // Stats
    document.getElementById('st-total').textContent = stats.resolved + ' / ' + stats.total;
    document.getElementById('st-wr').textContent = stats.resolved > 0 ? stats.win_rate + '%' : '-';
    const rEl = document.getElementById('st-r');
    rEl.textContent = stats.total_r >= 0 ? '+' + stats.total_r + 'R' : stats.total_r + 'R';
    rEl.className = 'val ' + (stats.total_r >= 0 ? 'green' : 'red');

    // Active signals
    const aEl = document.getElementById('active-signals');
    aEl.innerHTML = cachedActive.length === 0
      ? '<div class="empty">No active signals</div>'
      : cachedActive.map(s => renderSignalCard(s, false)).join('');

    // Live signal history (most recent first)
    const lEl = document.getElementById('live-signals');
    const reversed = [...cachedLive].reverse().slice(0, 20);
    lEl.innerHTML = reversed.length === 0
      ? '<div class="empty">No live signals yet</div>'
      : reversed.map(s => renderSignalCard(s, true)).join('');

    // Jobs
    const jEl = document.getElementById('jobs');
    jEl.innerHTML = hpData.scheduled_jobs.map(j => {
      const next = j.next_run ? new Date(j.next_run).toUTCString().slice(5, -4) : 'N/A';
      return '<div class="job-row"><span>' + j.id + '</span><span style="color:#8b949e">next: ' + next + '</span></div>';
    }).join('');

    document.getElementById('status').innerHTML =
      '<span class="dot"></span>' + stats.total + ' live signal(s) · ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('status').innerHTML = '<span class="dot" style="background:#f85149"></span>Error';
  }
}

new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight })).observe(container);

loadChart(currentTf);
refreshData();
setInterval(refreshData, 15000);
setInterval(() => loadChart(currentTf), 60000);
</script>
</body>
</html>"""
