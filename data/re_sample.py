import sys
import pandas as pd
from ta import volume, volatility, trend, momentum

# Usage: python data/re_sample.py [asset] [timeframes...]
# Examples:
#   python data/re_sample.py btc 4h 1h 15min
#   python data/re_sample.py gold 4h 1h
#   python data/re_sample.py           # defaults: btc, 4h

args = sys.argv[1:]
asset = args[0] if args else "btc"
timeframes = args[1:] if len(args) > 1 else ["4h"]

input_file = f"data/{asset}_data.csv"
print(f"Loading {input_file}...")

# Auto-detect format: semicolon-delimited (MT4/5 export) vs comma-delimited (Binance)
with open(input_file) as f:
    first_line = f.readline()

if ";" in first_line:
    # MT4/MT5 format: Date;Open;High;Low;Close;Volume (Date = "YYYY.MM.DD HH:MM")
    df = pd.read_csv(input_file, sep=";")
    df["timestamp"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
else:
    # Binance format: Timestamp (unix seconds)
    df = pd.read_csv(input_file)
    df["timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

df = df.set_index("timestamp")

for tf in timeframes:
    print(f"Resampling 1m → {tf}...")
    df_rs = (
        df.resample(tf)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )

    df_rs["obv"] = volume.on_balance_volume(close=df_rs["Close"], volume=df_rs["Volume"], fillna=True)
    df_rs["bb"] = volatility.bollinger_wband(close=df_rs["Close"], window=20, window_dev=2, fillna=True)
    df_rs["ema_21"] = trend.ema_indicator(close=df_rs["Close"], window=21, fillna=True)
    df_rs["rsi"] = momentum.rsi(close=df_rs["Close"], fillna=True)
    df_rs = df_rs.dropna()

    out = f"data/{asset}_{tf}.csv"
    df_rs.to_csv(out)
    print(f"  {len(df_rs)} bars → {out}")
