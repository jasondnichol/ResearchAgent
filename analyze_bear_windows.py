"""Analyze bear market windows to understand tradeable opportunities for shorts."""
import pandas as pd
import numpy as np
from backtest_donchian_daily import fetch_all_coins, calculate_indicators
from backtest_shorts import compute_btc_bear_filter, SHORT_DEFAULT_PARAMS

coin_data = fetch_all_coins(coins=['BTC-USD', 'ETH-USD', 'SOL-USD'], years=4)
btc = coin_data['BTC-USD']

df = btc.copy()
df['sma_200'] = df['close'].rolling(200).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['ema_50'] = df['close'].ewm(span=50).mean()
df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df = df.dropna(subset=['sma_200'])

print(f"Total trading days (with SMA200): {len(df)}")

# Bear conditions
below_200 = df[df['close'] < df['sma_200']]
death_cross = df[(df['close'] < df['sma_200']) & (df['sma_50'] < df['sma_200'])]
below_ema50 = df[df['close'] < df['ema_50']]
macd_bearish = df[df['macd'] < df['macd_signal']]

print(f"Below SMA200:       {len(below_200):>4d} ({len(below_200)/len(df)*100:.1f}%)")
print(f"Death cross:        {len(death_cross):>4d} ({len(death_cross)/len(df)*100:.1f}%)")
print(f"Below EMA50:        {len(below_ema50):>4d} ({len(below_ema50)/len(df)*100:.1f}%)")
print(f"MACD bearish:       {len(macd_bearish):>4d} ({len(macd_bearish)/len(df)*100:.1f}%)")

# Bear windows
df['below_200'] = df['close'] < df['sma_200']
df['regime_change'] = df['below_200'] != df['below_200'].shift(1)
df['regime_id'] = df['regime_change'].cumsum()

bear_regimes = df[df['below_200']].groupby('regime_id').agg(
    start=('time', 'first'),
    end=('time', 'last'),
    days=('time', 'count'),
    min_price=('close', 'min'),
    max_price=('close', 'max'),
).reset_index()

print("\nBear Windows (BTC below SMA200):")
for _, r in bear_regimes.iterrows():
    start_str = str(r['start'])[:10]
    end_str = str(r['end'])[:10]
    print(f"  {start_str} to {end_str}: {r['days']} days, "
          f"price ${r['min_price']:,.0f}-${r['max_price']:,.0f}")

# Now check: during bear windows, how many Donchian breakdowns occur per coin?
print("\n" + "="*80)
print("DONCHIAN BREAKDOWN OPPORTUNITIES DURING BEAR WINDOWS")
print("="*80)

bear_simple = compute_btc_bear_filter(btc, require_death_cross=False)
bear_strict = compute_btc_bear_filter(btc, require_death_cross=True)

coins = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD']

for filter_name, bear_filter in [('SMA200 only', bear_simple), ('Death cross', bear_strict)]:
    print(f"\n--- {filter_name} ---")
    total_signals = 0
    for symbol in coins:
        if symbol not in coin_data:
            continue
        df_c = calculate_indicators(coin_data[symbol], SHORT_DEFAULT_PARAMS)
        df_c['exit_high'] = df_c['high'].rolling(window=10).max()
        df_c = df_c.dropna(subset=['donchian_low', 'ema_21', 'volume_sma', 'atr'])

        signals = 0
        for _, row in df_c.iterrows():
            date = row['time'].date()
            if not bear_filter.get(date, False):
                continue
            # Check previous day's donchian_low
            idx = df_c.index.get_loc(row.name)
            if idx == 0:
                continue
            prev = df_c.iloc[idx - 1]
            if pd.isna(prev['donchian_low']):
                continue

            breakdown = row['close'] < prev['donchian_low']
            vol_ok = row['volume'] > 1.5 * row['volume_sma']
            trend_ok = row['close'] < row['ema_21']
            if breakdown and vol_ok and trend_ok:
                signals += 1

        total_signals += signals
        print(f"  {symbol:<10s}: {signals} entry signals")
    print(f"  TOTAL:      {total_signals} signals across all coins")

# Now count alternative signal types during bear windows
print("\n" + "="*80)
print("ALTERNATIVE SIGNAL OPPORTUNITIES (SMA200 bear filter)")
print("="*80)

for symbol in coins:
    if symbol not in coin_data:
        continue
    df_c = coin_data[symbol].copy()
    df_c['sma_200'] = df_c['close'].rolling(200).mean()
    df_c['ema_21'] = df_c['close'].ewm(span=21).mean()
    df_c['rsi'] = 100 - (100 / (1 + df_c['close'].diff().clip(lower=0).rolling(14).mean() /
                                  df_c['close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df_c['macd'] = df_c['close'].ewm(span=12).mean() - df_c['close'].ewm(span=26).mean()
    df_c['macd_signal'] = df_c['macd'].ewm(span=9).mean()
    df_c['vol_sma'] = df_c['volume'].rolling(20).mean()
    df_c = df_c.dropna()

    # Count signal types
    macd_crosses = 0
    bearish_engulf = 0
    rsi_overbought = 0
    vol_exhaustion = 0

    for i in range(1, len(df_c)):
        row = df_c.iloc[i]
        prev = df_c.iloc[i-1]
        date = row['time'].date()
        if not bear_simple.get(date, False):
            continue

        # MACD crosses below signal
        if prev['macd'] >= prev['macd_signal'] and row['macd'] < row['macd_signal']:
            macd_crosses += 1

        # Bearish engulfing
        if (row['close'] < row['open'] and  # red candle
            prev['close'] > prev['open'] and  # prev green
            row['open'] > prev['close'] and  # opens above prev close
            row['close'] < prev['open']):  # closes below prev open
            bearish_engulf += 1

        # RSI exits overbought
        if prev['rsi'] >= 60 and row['rsi'] < 60 and row['close'] < float(row['sma_200']):
            rsi_overbought += 1

        # Volume exhaustion (3-day volume declining while price rose)
        if i >= 3:
            p3 = df_c.iloc[i-3:i+1]
            price_rose = p3['close'].iloc[-1] > p3['close'].iloc[0]
            vol_declining = p3['volume'].iloc[-1] < p3['volume'].iloc[0] * 0.7
            bearish_close = row['close'] < row['open']
            if price_rose and vol_declining and bearish_close:
                vol_exhaustion += 1

    print(f"\n{symbol}:")
    print(f"  MACD cross below signal:  {macd_crosses}")
    print(f"  Bearish engulfing:        {bearish_engulf}")
    print(f"  RSI exits 60+ zone:       {rsi_overbought}")
    print(f"  Volume exhaustion:        {vol_exhaustion}")
