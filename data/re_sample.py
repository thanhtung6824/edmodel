import sys
import pandas as pd
from ta import volume, volatility, trend, momentum

df = pd.read_csv("data/btc_data.csv")
df["timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
df = df.set_index("timestamp")

timeframes = sys.argv[1:] if len(sys.argv) > 1 else ["4h"]

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

    out = f"data/btc_{tf}.csv"
    df_rs.to_csv(out)
    print(f"  {len(df_rs)} bars → {out}")
