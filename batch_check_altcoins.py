"""Fetch Binance data for all altcoins × all TFs, run check_signal, summarize."""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.parse

ALTCOINS = {
    "doge": {"symbol": "DOGEUSDT"},
    "avax": {"symbol": "AVAXUSDT"},
    "link": {"symbol": "LINKUSDT"},
    "arb": {"symbol": "ARBUSDT"},
    "sui": {"symbol": "SUIUSDT"},
    "tao": {"symbol": "TAOUSDT"},
    "ltc": {"symbol": "LTCUSDT"},
    "tia": {"symbol": "TIAUSDT"},
    "ondo": {"symbol": "ONDOUSDT"},
    "aster": {"symbol": "ASTERUSDT"},
    "sei": {"symbol": "SEIUSDT"},
    "aave": {"symbol": "AAVEUSDT"},
    "bnb": {"symbol": "BNBUSDT"},
    "near": {"symbol": "NEARUSDT"},
    "op": {"symbol": "OPUSDT"},
    "hype": {"symbol": "HYPEUSDT", "futures": True},
    "bch": {"symbol": "BCHUSDT"},
    "zro": {"symbol": "ZROUSDT"},
    "zec": {"symbol": "ZECUSDT"},
}

TFS = ["15m", "1h", "4h"]
N_CANDLES = 1000
SPOT_URL = "https://api.binance.com/api/v3/klines"
FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_klines(symbol: str, interval: str, limit: int = N_CANDLES, futures: bool = False):
    """Fetch raw klines from Binance (sync). Uses Futures API when futures=True."""
    base_url = FUTURES_URL if futures else SPOT_URL
    params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    url = f"{base_url}?{params}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(2)
            else:
                raise


def run_check_signal(asset: str, tf: str) -> str:
    """Run check_signal.py and return stdout."""
    result = subprocess.run(
        [sys.executable, "check_signal.py", "--asset", asset, "--tf", tf],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout + result.stderr


def parse_summary_line(output: str, prefix: str) -> str:
    """Extract a summary line from check_signal output."""
    for line in output.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


def main():
    results = []
    failed = []

    total = len(ALTCOINS) * len(TFS)
    done = 0

    for asset, acfg in ALTCOINS.items():
        symbol = acfg["symbol"]
        use_futures = acfg.get("futures", False)
        for tf in TFS:
            done += 1
            src = "futures" if use_futures else "spot"
            print(f"[{done}/{total}] {asset.upper()} {tf} ({symbol}, {src})...", end=" ", flush=True)

            # Fetch data
            try:
                raw = fetch_klines(symbol, tf, futures=use_futures)
            except Exception as e:
                msg = f"FETCH FAILED: {e}"
                print(msg)
                failed.append(f"{asset.upper()} {tf}: {msg}")
                continue

            # Save to recent_data.json
            with open("recent_data.json", "w") as f:
                json.dump(raw, f)

            # Run check_signal
            try:
                output = run_check_signal(asset, tf)
            except subprocess.TimeoutExpired:
                msg = "TIMEOUT"
                print(msg)
                failed.append(f"{asset.upper()} {tf}: {msg}")
                continue
            except Exception as e:
                msg = f"ERROR: {e}"
                print(msg)
                failed.append(f"{asset.upper()} {tf}: {msg}")
                continue

            # Parse summary lines
            all_line = parse_summary_line(output, "All signals:")
            passed_line = parse_summary_line(output, "Passed only:")
            signals_line = parse_summary_line(output, "Signals after dedup:")

            print(f"{all_line or 'No signals'}")

            results.append({
                "asset": asset.upper(),
                "tf": tf,
                "symbol": symbol,
                "all_line": all_line,
                "passed_line": passed_line,
                "signals_line": signals_line,
                "full_output": output,
            })

            # Small delay to avoid rate limiting
            time.sleep(0.5)

    # Write summary
    with open("altcoin_check_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ALTCOIN SIGNAL CHECK RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Assets: {len(ALTCOINS)} | Timeframes: {', '.join(TFS)}\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write(f"{'Asset':<8} {'TF':<5} | {'All Signals':<45} | {'Passed Only':<45}\n")
        f.write("-" * 110 + "\n")

        for r in results:
            f.write(f"{r['asset']:<8} {r['tf']:<5} | {r['all_line']:<45} | {r['passed_line']:<45}\n")

        if failed:
            f.write("\n\nFAILED:\n")
            for fl in failed:
                f.write(f"  {fl}\n")

        # Full outputs
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETAILED OUTPUTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"--- {r['asset']} {r['tf']} ({r['symbol']}) ---\n")
            f.write(r["full_output"])
            f.write("\n\n")

    print(f"\nResults written to altcoin_check_results.txt")
    if failed:
        print(f"Failed: {len(failed)} combos")


if __name__ == "__main__":
    main()
