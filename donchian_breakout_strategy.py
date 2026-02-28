"""Donchian Channel Breakout Strategy — Daily Candles, Multi-Coin

Designed for bull cycle crypto trading with realistic transaction costs.
Targets 5-15% gains per trade with holds lasting days to weeks.

Entry: Price breaks above 20-day high + volume > 1.5x 20-day avg + weekly EMA(21) filter
Exit: 3x ATR(14) trailing stop from high watermark, with partial profit taking
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DonchianBreakoutStrategy:
    def __init__(self, params=None):
        self.coinbase_api = "https://api.exchange.coinbase.com"

        # Configurable parameters (with research-backed defaults)
        p = params or {}
        self.donchian_period = p.get('donchian_period', 20)      # 20-day channel
        self.exit_period = p.get('exit_period', 10)               # 10-day low for exit
        self.atr_period = p.get('atr_period', 14)
        self.atr_mult = p.get('atr_mult', 3.0)                   # 3x ATR trailing stop
        self.volume_mult = p.get('volume_mult', 1.5)              # volume > 1.5x avg
        self.ema_period = p.get('ema_period', 21)                 # weekly EMA(21) trend filter
        self.rsi_blowoff = p.get('rsi_blowoff', 80)              # tighten stop if RSI > 80
        self.volume_blowoff = p.get('volume_blowoff', 3.0)       # tighten stop if vol > 3x
        self.atr_mult_tight = p.get('atr_mult_tight', 1.5)       # tightened stop multiplier
        self.tp1_pct = p.get('tp1_pct', 10.0)                    # first partial exit at +10%
        self.tp2_pct = p.get('tp2_pct', 20.0)                    # second partial exit at +20%
        self.tp1_fraction = p.get('tp1_fraction', 0.25)           # sell 25% at TP1
        self.tp2_fraction = p.get('tp2_fraction', 0.25)           # sell 25% at TP2
        self.lookback = p.get('lookback', 60)                     # candles needed for indicators

        # State tracking (for live trading)
        self.entry_price = None
        self.high_watermark = None
        self.partials_taken = 0  # 0, 1, or 2

    def fetch_daily_candles(self, symbol='BTC-USD', limit=60):
        """Fetch recent daily candles from Coinbase"""
        url = f"{self.coinbase_api}/products/{symbol}/candles"
        params = {'granularity': 86400}  # daily

        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        df = pd.DataFrame(data[:limit], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def calculate_indicators(self, df):
        """Calculate all indicators needed for the strategy"""
        df = df.copy()

        # Donchian Channel (20-day)
        df['donchian_high'] = df['high'].rolling(window=self.donchian_period).max()
        df['donchian_low'] = df['low'].rolling(window=self.donchian_period).min()

        # 10-day low (for exit channel)
        df['exit_low'] = df['low'].rolling(window=self.exit_period).min()

        # ATR(14) using Wilder's EWM
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/self.atr_period, min_periods=self.atr_period, adjust=False).mean()

        # Volume SMA(20)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # EMA(21) — trend filter
        df['ema_21'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()

        # RSI(14) — for blow-off detection
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signal(self, symbol='BTC-USD'):
        """Generate BUY/SELL/HOLD signal for live trading"""
        df = self.fetch_daily_candles(symbol, limit=self.lookback)

        if len(df) < self.lookback - 5:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}

        df = self.calculate_indicators(df)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check for NaN in key indicators
        for col in ['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi']:
            if pd.isna(current[col]):
                return {'signal': 'HOLD', 'reason': 'Indicators not ready'}

        # --- BUY CONDITIONS ---
        # 1. Price breaks above previous bar's Donchian high (use prev to avoid look-ahead)
        breakout = current['close'] > prev['donchian_high']

        # 2. Volume confirmation: today's volume > 1.5x 20-day average
        volume_ok = current['volume'] > self.volume_mult * current['volume_sma']

        # 3. Trend filter: price above EMA(21)
        trend_ok = current['close'] > current['ema_21']

        if breakout and volume_ok and trend_ok:
            self.entry_price = float(current['close'])
            self.high_watermark = float(current['close'])
            self.partials_taken = 0
            return {
                'signal': 'BUY',
                'reason': f'Donchian breakout above ${prev["donchian_high"]:,.2f} with volume confirmation',
                'price': float(current['close']),
                'donchian_high': float(prev['donchian_high']),
                'atr': float(current['atr']),
                'volume_ratio': float(current['volume'] / current['volume_sma']),
                'ema_21': float(current['ema_21']),
                'rsi': float(current['rsi']),
                'time': str(current['time'])
            }

        # --- SELL CONDITIONS (if in position) ---
        if self.entry_price is not None:
            self.high_watermark = max(self.high_watermark, float(current['high']))

            sell_reason = None

            # Blow-off detection: tighten stop if volume > 3x AND RSI > 80
            is_blowoff = (current['volume'] > self.volume_blowoff * current['volume_sma']
                          and current['rsi'] > self.rsi_blowoff)
            stop_mult = self.atr_mult_tight if is_blowoff else self.atr_mult

            # Trailing stop: high watermark - Nx ATR
            trailing_stop = self.high_watermark - (stop_mult * current['atr'])
            if current['close'] <= trailing_stop:
                sell_reason = f'Trailing stop hit ({stop_mult}x ATR from ${self.high_watermark:,.2f})'

            # Donchian exit channel: price breaks below 10-day low
            if not sell_reason and current['close'] < prev['exit_low']:
                sell_reason = f'Donchian exit: broke below 10-day low ${prev["exit_low"]:,.2f}'

            if sell_reason:
                self.entry_price = None
                self.high_watermark = None
                self.partials_taken = 0
                return {
                    'signal': 'SELL',
                    'reason': sell_reason,
                    'price': float(current['close']),
                    'atr': float(current['atr']),
                    'rsi': float(current['rsi']),
                    'time': str(current['time'])
                }

        # HOLD
        return {
            'signal': 'HOLD',
            'reason': 'No entry/exit conditions met',
            'price': float(current['close']),
            'donchian_high': float(current['donchian_high']) if pd.notna(current['donchian_high']) else 0,
            'atr': float(current['atr']) if pd.notna(current['atr']) else 0,
            'ema_21': float(current['ema_21']) if pd.notna(current['ema_21']) else 0,
            'rsi': float(current['rsi']) if pd.notna(current['rsi']) else 0,
            'time': str(current['time'])
        }

    def calculate_short_indicators(self, df):
        """Calculate indicators for short-side trading.

        Adds exit_high (N-day rolling HIGH for short cover exit)
        in addition to all standard indicators.
        """
        df = self.calculate_indicators(df)
        df['exit_high'] = df['high'].rolling(window=self.exit_period).max()
        return df

    def generate_short_signal(self, symbol='BTC-USD'):
        """Generate SHORT_SELL/HOLD signal for short trading.

        Entry: close < previous Donchian LOW + volume confirmation + below EMA(21)
        Uses inverted logic compared to generate_signal().
        """
        df = self.fetch_daily_candles(symbol, limit=self.lookback)

        if len(df) < self.lookback - 5:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}

        df = self.calculate_short_indicators(df)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check for NaN in key indicators
        for col in ['donchian_low', 'atr', 'volume_sma', 'ema_21', 'rsi']:
            if pd.isna(current[col]):
                return {'signal': 'HOLD', 'reason': 'Indicators not ready'}

        # --- SHORT ENTRY CONDITIONS ---
        # 1. Price breaks below previous bar's Donchian low (breakdown)
        breakdown = current['close'] < prev['donchian_low']

        # 2. Volume confirmation: today's volume > Nx 20-day average
        volume_ok = current['volume'] > self.volume_mult * current['volume_sma']

        # 3. Trend filter: price below EMA(21) (in downtrend)
        trend_ok = current['close'] < current['ema_21']

        if breakdown and volume_ok and trend_ok:
            return {
                'signal': 'SHORT_SELL',
                'reason': f'Donchian breakdown below ${prev["donchian_low"]:,.2f} with volume confirmation',
                'price': float(current['close']),
                'donchian_low': float(prev['donchian_low']),
                'atr': float(current['atr']),
                'volume_ratio': float(current['volume'] / current['volume_sma']),
                'ema_21': float(current['ema_21']),
                'rsi': float(current['rsi']),
                'time': str(current['time'])
            }

        # HOLD — build reason
        reasons = []
        if not breakdown:
            reasons.append(f'no breakdown (close ${current["close"]:,.2f} vs low ${prev["donchian_low"]:,.2f})')
        if not volume_ok:
            vol_ratio = current['volume'] / current['volume_sma'] if current['volume_sma'] > 0 else 0
            reasons.append(f'low volume ({vol_ratio:.1f}x, need {self.volume_mult}x)')
        if not trend_ok:
            reasons.append(f'above EMA21 (${current["close"]:,.2f} vs ${current["ema_21"]:,.2f})')

        return {
            'signal': 'HOLD',
            'reason': '; '.join(reasons) if reasons else 'No short conditions met',
            'price': float(current['close']),
            'donchian_low': float(current['donchian_low']) if pd.notna(current['donchian_low']) else 0,
            'atr': float(current['atr']) if pd.notna(current['atr']) else 0,
            'ema_21': float(current['ema_21']) if pd.notna(current['ema_21']) else 0,
            'rsi': float(current['rsi']) if pd.notna(current['rsi']) else 0,
            'time': str(current['time'])
        }


def main():
    """Test the strategy"""
    print("=" * 80)
    print("TESTING DONCHIAN CHANNEL BREAKOUT STRATEGY (Daily)")
    print("=" * 80 + "\n")

    strategy = DonchianBreakoutStrategy()

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']:
        print(f"\n--- {symbol} ---")
        try:
            signal = strategy.generate_signal(symbol)
            print(f"Signal: {signal['signal']}")
            print(f"Reason: {signal['reason']}")
            print(f"Price: ${signal.get('price', 0):,.2f}")
            if signal.get('donchian_high'):
                print(f"Donchian High: ${signal['donchian_high']:,.2f}")
            print(f"ATR: ${signal.get('atr', 0):,.2f}")
            print(f"RSI: {signal.get('rsi', 0):.1f}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
