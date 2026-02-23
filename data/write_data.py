import pandas as pd
from ta import volume, volatility, trend, momentum
from ta.utils import dropna
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "../btcusd_1-min_data.csv")

df = pd.read_csv(filename, sep=",")

df = dropna(df)

df["obv"] = volume.on_balance_volume(close=df["Close"], volume=df["Volume"], fillna=True)

df["bb"] = volatility.bollinger_wband(close=df["Close"], window=20, window_dev=2, fillna=True)

df["ema_21"] = trend.ema_indicator(close=df["Close"], window=21, fillna=True)

df["rsi"] = momentum.rsi(close=df["Close"], fillna=True)


def write_data():
    data = pd.DataFrame(df)
    data.to_csv("data/btc_data.csv", index=False)


write_data()
