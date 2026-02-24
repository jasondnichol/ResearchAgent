"""Bollinger Band Mean Reversion Strategy Implementation - VOLATILE Markets"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class BBReversionStrategy:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
        self.timeframe = '1H'  # 1 hour candles
        self.lookback = 100    # Need 100 candles for indicators
        self.entry_price = None  # Track entry for stop loss

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

    def calculate_indicators(self, df, bb_period=20, bb_sigma=2.0, rsi_period=14, atr_period=14):
        """Calculate Bollinger Bands (20, 2sigma), RSI(14), ATR(14)"""
        df = df.copy()

        # --- Bollinger Bands ---
        df['sma_20'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['sma_20'] + bb_sigma * df['bb_std']
        df['bb_lower'] = df['sma_20'] - bb_sigma * df['bb_std']

        # --- RSI (Wilder's smoothing) ---
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # --- ATR (Wilder's smoothing, for stop loss) ---
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean()

        return df

    def generate_signal(self, symbol='BTC-USD'):
        """Generate BUY/SELL/HOLD signal"""

        # Fetch data
        df = self.fetch_candles(symbol, granularity=3600, limit=100)

        if len(df) < 30:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}

        # Calculate indicators
        df = self.calculate_indicators(df)

        current = df.iloc[-1]

        # Check for NaN
        for col in ['sma_20', 'bb_lower', 'bb_upper', 'rsi', 'atr']:
            if pd.isna(current[col]):
                return {'signal': 'HOLD', 'reason': 'Indicators not ready'}

        # --- BUY CONDITIONS ---
        # 1. Close below lower Bollinger Band
        close_below_lower_bb = current['close'] < current['bb_lower']

        # 2. RSI < 35 (oversold)
        rsi_oversold = current['rsi'] < 35

        if close_below_lower_bb and rsi_oversold:
            self.entry_price = float(current['close'])
            return {
                'signal': 'BUY',
                'reason': 'BB mean reversion: price below lower BB with RSI oversold',
                'price': float(current['close']),
                'rsi': float(current['rsi']),
                'bb_lower': float(current['bb_lower']),
                'bb_upper': float(current['bb_upper']),
                'sma_20': float(current['sma_20']),
                'atr': float(current['atr']),
                'time': str(current['time'])
            }

        # --- SELL CONDITIONS (any one triggers) ---
        sell_reason = None

        # 1. Price reaches middle BB (SMA 20) — profit target
        if current['close'] >= current['sma_20']:
            sell_reason = 'Price reached middle BB (SMA 20)'

        # 2. RSI > 70 (overbought)
        elif current['rsi'] > 70:
            sell_reason = 'RSI > 70 (overbought)'

        # 3. Stop loss: price drops 1.5x ATR below entry
        elif self.entry_price is not None:
            stop_level = self.entry_price - (1.5 * current['atr'])
            if current['close'] <= stop_level:
                sell_reason = f'Stop loss hit (1.5x ATR below entry ${self.entry_price:,.2f})'

        if sell_reason:
            self.entry_price = None
            return {
                'signal': 'SELL',
                'reason': sell_reason,
                'price': float(current['close']),
                'rsi': float(current['rsi']),
                'bb_lower': float(current['bb_lower']),
                'bb_upper': float(current['bb_upper']),
                'sma_20': float(current['sma_20']),
                'atr': float(current['atr']),
                'time': str(current['time'])
            }

        # Otherwise hold
        return {
            'signal': 'HOLD',
            'reason': 'No entry/exit conditions met',
            'price': float(current['close']),
            'rsi': float(current['rsi']),
            'bb_lower': float(current['bb_lower']),
            'bb_upper': float(current['bb_upper']),
            'sma_20': float(current['sma_20']),
            'atr': float(current['atr']),
            'time': str(current['time'])
        }


def main():
    """Test the strategy"""
    print("=" * 80)
    print("TESTING BOLLINGER BAND MEAN REVERSION STRATEGY")
    print("=" * 80 + "\n")

    strategy = BBReversionStrategy()

    print("Fetching BTC-USD 1H candles...")
    signal = strategy.generate_signal('BTC-USD')

    print("\n" + "=" * 80)
    print(f"SIGNAL: {signal['signal']}")
    print("=" * 80)
    print(f"Reason: {signal['reason']}")
    print(f"Price: ${signal.get('price', 0):,.2f}")
    print(f"RSI: {signal.get('rsi', 0):.2f}")
    print(f"BB Lower: ${signal.get('bb_lower', 0):,.2f}")
    print(f"BB Upper: ${signal.get('bb_upper', 0):,.2f}")
    print(f"SMA 20: ${signal.get('sma_20', 0):,.2f}")
    print(f"ATR: ${signal.get('atr', 0):,.2f}")
    print(f"Time: {signal.get('time', 'N/A')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
