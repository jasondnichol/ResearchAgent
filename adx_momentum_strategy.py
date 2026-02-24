"""ADX Momentum Thrust Strategy Implementation - TRENDING Markets"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ADXMomentumStrategy:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
        self.timeframe = '1H'  # 1 hour candles
        self.lookback = 100    # Need 100 candles for indicators
        self.entry_price = None  # Track entry for trailing stop
        self.high_watermark = None  # Track highest price since entry

    def fetch_candles(self, symbol='BTC-USD', granularity=3600, limit=100):
        """Fetch recent 1H candles"""
        url = f"{self.coinbase_api}/products/{symbol}/candles"
        params = {'granularity': granularity}

        response = requests.get(url, params=params)
        data = response.json()

        # Take most recent 'limit' candles
        df = pd.DataFrame(data[:limit], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)

        return df

    def calculate_indicators(self, df, adx_period=14, rsi_period=14, atr_period=14, sma_period=50):
        """Calculate ADX, +DI, -DI, RSI, ATR, SMA(50)"""
        df = df.copy()

        # --- True Range ---
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)

        # --- +DM / -DM ---
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = pd.Series(
            np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
            index=df.index
        )

        # --- Smoothed TR, +DM, -DM (Wilder's smoothing) ---
        atr = tr.ewm(alpha=1/adx_period, min_periods=adx_period, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(alpha=1/adx_period, min_periods=adx_period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/adx_period, min_periods=adx_period, adjust=False).mean()

        # --- +DI / -DI ---
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr

        # --- DX and ADX ---
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/adx_period, min_periods=adx_period, adjust=False).mean()

        # --- RSI ---
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # --- ATR (for stop loss) ---
        atr_stop = tr.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean()

        # --- SMA(50) ---
        sma_50 = df['close'].rolling(window=sma_period).mean()

        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['rsi'] = rsi
        df['atr'] = atr_stop
        df['sma_50'] = sma_50

        return df

    def generate_signal(self, symbol='BTC-USD'):
        """Generate BUY/SELL/HOLD signal"""

        # Fetch data
        df = self.fetch_candles(symbol, granularity=3600, limit=100)

        if len(df) < 60:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Need at least 4 bars of valid ADX for "rising over 3 bars" check
        if df['adx'].dropna().shape[0] < 4:
            return {'signal': 'HOLD', 'reason': 'Indicators not ready'}

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check for NaN
        for col in ['adx', 'plus_di', 'minus_di', 'rsi', 'atr', 'sma_50']:
            if pd.isna(current[col]):
                return {'signal': 'HOLD', 'reason': 'Indicators not ready'}

        # --- BUY CONDITIONS ---
        # 1. ADX > 20 and rising over last 5 bars
        adx_rising = (df['adx'].iloc[-1] > df['adx'].iloc[-6])
        adx_above_20 = current['adx'] > 20

        # 2. +DI > -DI
        plus_di_above = current['plus_di'] > current['minus_di']

        # 3. +DI crossed above -DI within last 15 bars
        di_cross_recent = False
        for offset in range(0, 15):
            idx = len(df) - 1 - offset
            if idx >= 1:
                bar = df.iloc[idx]
                bar_prev = df.iloc[idx - 1]
                if (not pd.isna(bar['plus_di']) and not pd.isna(bar['minus_di'])
                        and not pd.isna(bar_prev['plus_di']) and not pd.isna(bar_prev['minus_di'])):
                    if bar['plus_di'] > bar['minus_di'] and bar_prev['plus_di'] <= bar_prev['minus_di']:
                        di_cross_recent = True
                        break

        # 4. RSI between 35-78
        rsi_ok = 35 <= current['rsi'] <= 78

        # 5. Price above SMA(50)
        price_above_sma = current['close'] > current['sma_50']

        if adx_above_20 and adx_rising and plus_di_above and di_cross_recent and rsi_ok and price_above_sma:
            self.entry_price = float(current['close'])
            self.high_watermark = float(current['close'])
            return {
                'signal': 'BUY',
                'reason': 'ADX momentum thrust: ADX rising, +DI crossover, RSI confirming uptrend',
                'price': float(current['close']),
                'adx': float(current['adx']),
                'plus_di': float(current['plus_di']),
                'minus_di': float(current['minus_di']),
                'rsi': float(current['rsi']),
                'atr': float(current['atr']),
                'sma_50': float(current['sma_50']),
                'time': str(current['time'])
            }

        # --- SELL CONDITIONS (any one triggers) ---
        sell_reason = None

        # Update trailing stop high watermark
        if self.high_watermark is not None:
            self.high_watermark = max(self.high_watermark, float(current['close']))

        # 1. ADX drops below 20 (trend exhausted)
        if current['adx'] < 20:
            sell_reason = 'ADX dropped below 20 (trend weakening)'

        # 2. RSI > 75 (overbought — take profit)
        elif current['rsi'] > 75:
            sell_reason = 'RSI > 75 (overbought)'

        # 3. Trailing stop: price drops 1.5x ATR below high watermark
        elif self.high_watermark is not None:
            trailing_stop = self.high_watermark - (1.5 * current['atr'])
            if current['close'] <= trailing_stop:
                sell_reason = f'Trailing stop hit (1.5x ATR from high ${self.high_watermark:,.2f})'

        # 4. -DI sustained above +DI for 3 bars (confirmed reversal, not noise)
        elif len(df) >= 4:
            di_bearish_count = 0
            for j in range(3):
                idx = len(df) - 1 - j
                if idx >= 0 and df.iloc[idx]['minus_di'] > df.iloc[idx]['plus_di']:
                    di_bearish_count += 1
            if di_bearish_count == 3:
                sell_reason = '-DI above +DI for 3 bars (confirmed bearish reversal)'

        if sell_reason:
            self.entry_price = None
            self.high_watermark = None
            return {
                'signal': 'SELL',
                'reason': sell_reason,
                'price': float(current['close']),
                'adx': float(current['adx']),
                'plus_di': float(current['plus_di']),
                'minus_di': float(current['minus_di']),
                'rsi': float(current['rsi']),
                'atr': float(current['atr']),
                'sma_50': float(current['sma_50']),
                'time': str(current['time'])
            }

        # Otherwise hold
        return {
            'signal': 'HOLD',
            'reason': 'No entry/exit conditions met',
            'price': float(current['close']),
            'adx': float(current['adx']),
            'plus_di': float(current['plus_di']),
            'minus_di': float(current['minus_di']),
            'rsi': float(current['rsi']),
            'atr': float(current['atr']),
            'sma_50': float(current['sma_50']),
            'time': str(current['time'])
        }


def main():
    """Test the strategy"""
    print("=" * 80)
    print("TESTING ADX MOMENTUM THRUST STRATEGY")
    print("=" * 80 + "\n")

    strategy = ADXMomentumStrategy()

    print("Fetching BTC-USD 1H candles...")
    signal = strategy.generate_signal('BTC-USD')

    print("\n" + "=" * 80)
    print(f"SIGNAL: {signal['signal']}")
    print("=" * 80)
    print(f"Reason: {signal['reason']}")
    print(f"Price: ${signal.get('price', 0):,.2f}")
    print(f"ADX: {signal.get('adx', 0):.2f}")
    print(f"+DI: {signal.get('plus_di', 0):.2f}")
    print(f"-DI: {signal.get('minus_di', 0):.2f}")
    print(f"RSI: {signal.get('rsi', 0):.2f}")
    print(f"ATR: ${signal.get('atr', 0):,.2f}")
    print(f"SMA 50: ${signal.get('sma_50', 0):,.2f}")
    print(f"Time: {signal.get('time', 'N/A')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
