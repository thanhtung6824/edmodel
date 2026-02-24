import pandas as pd
from ta import volume, volatility, trend, momentum

df = pd.read_csv('data/btc_data.csv')
df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('timestamp')

# Resample 1m → 4h
df_4h = df.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

df_4h["obv"] = volume.on_balance_volume(close=df_4h["Close"], volume=df_4h["Volume"], fillna=True)

df_4h["bb"] = volatility.bollinger_wband(close=df_4h["Close"], window=20, window_dev=2, fillna=True)

df_4h["ema_21"] = trend.ema_indicator(close=df_4h["Close"], window=21, fillna=True)

df_4h["rsi"] = momentum.rsi(close=df_4h["Close"], fillna=True)

df_4h = df_4h.dropna()

df_4h.to_csv('data/btc_4h.csv')
