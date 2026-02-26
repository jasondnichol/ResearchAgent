"""Detect current market regime for strategy selection"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RegimeClassifier:
    """Single source of truth for market regime classification.

    Every backtester and the production bot use this class so that
    regime labels are always consistent.

    Timeframe-aware: ATR/close volatility scales with candle size, so the
    VOLATILE threshold differs between daily (3.0%) and hourly (0.5%) data.
    ADX > 25 works the same on both timeframes.
    """

    # Volatility thresholds by timeframe (ATR/close %)
    # Calibrated so that each timeframe produces ~45% VOLATILE bars on 4yr BTC data
    VOLATILITY_THRESHOLDS = {
        'daily': 3.0,
        'hourly': 0.5,
    }
    ADX_THRESHOLD = 25  # Same for all timeframes

    @staticmethod
    def detect_timeframe(df):
        """Auto-detect whether a DataFrame contains daily or hourly candles.

        Uses the median time delta between consecutive bars.
        Returns 'hourly' or 'daily'.
        """
        if 'time' not in df.columns or len(df) < 3:
            return 'daily'  # safe default

        times = pd.to_datetime(df['time'])
        deltas = times.diff().dropna()
        median_delta = deltas.median()

        # Hourly: median delta ~1 hour (allow some slack for gaps)
        if median_delta <= pd.Timedelta(hours=2):
            return 'hourly'
        return 'daily'

    @staticmethod
    def compute_indicators(df):
        """Compute SMA(20), SMA(50), ATR(14), ADX(14) using Wilder's EWM,
        volatility_pct, and trend_direction on a full DataFrame.

        Returns a copy of df with the new columns added.
        """
        df = df.copy()
        period = 14

        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # True Range
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)

        # Wilder's EWM smoothing for ATR
        df['atr'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        df['volatility_pct'] = (df['atr'] / df['close']) * 100

        # +DM / -DM
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

        # Wilder's smoothing for DM
        smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        plus_di = 100 * smooth_plus_dm / df['atr']
        minus_di = 100 * smooth_minus_dm / df['atr']

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Trend direction
        df['trend_direction'] = 'SIDEWAYS'
        df.loc[
            (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']),
            'trend_direction'
        ] = 'UPTREND'
        df.loc[
            (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']),
            'trend_direction'
        ] = 'DOWNTREND'

        return df

    @classmethod
    def classify_bar(cls, adx, volatility_pct, trend_direction, timeframe='daily'):
        """The ONE canonical decision function.

        Args:
            adx: ADX value
            volatility_pct: ATR/close as percentage
            trend_direction: 'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
            timeframe: 'daily' or 'hourly' — determines volatility threshold

        Returns: 'TRENDING', 'VOLATILE', or 'RANGING'
        """
        vol_threshold = cls.VOLATILITY_THRESHOLDS.get(timeframe, 3.0)

        if adx > cls.ADX_THRESHOLD and trend_direction in ('UPTREND', 'DOWNTREND'):
            return 'TRENDING'
        if volatility_pct > vol_threshold:
            return 'VOLATILE'
        return 'RANGING'

    @classmethod
    def classify_dataframe(cls, df, min_warmup=50, timeframe=None):
        """Convenience: compute indicators + classify every bar.

        Args:
            df: DataFrame with OHLCV + time columns
            min_warmup: bars before this index are labelled 'UNKNOWN'
            timeframe: 'daily', 'hourly', or None (auto-detect)

        Returns a copy of df with 'regime' column added.
        """
        if timeframe is None:
            timeframe = cls.detect_timeframe(df)

        df = cls.compute_indicators(df)
        df['regime'] = 'UNKNOWN'

        for i in range(min_warmup, len(df)):
            row = df.iloc[i]
            if pd.isna(row['adx']) or pd.isna(row['sma_20']):
                continue
            df.iloc[i, df.columns.get_loc('regime')] = cls.classify_bar(
                row['adx'], row['volatility_pct'], row['trend_direction'],
                timeframe=timeframe
            )

        return df


class MarketRegimeDetector:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
    
    def fetch_recent_data(self, symbol='BTC-USD', days=30):
        """Fetch recent market data"""
        print(f"📊 Fetching {days} days of recent {symbol} data...")
        
        url = f"{self.coinbase_api}/products/{symbol}/candles"
        params = {'granularity': 3600}  # 1 hour candles
        
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        
        # Limit to requested days
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['time'] >= cutoff]
        
        print(f"✅ Loaded {len(df)} candles")
        return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (volatility measure)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, df, period=14):
        """Calculate ADX (trend strength indicator)"""
        # +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # True Range
        tr = self.calculate_atr(df, 1)
        
        # Smoothed DM and TR
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def detect_regime(self, symbol='BTC-USD'):
        """Detect current market regime"""
        print("\n🔍 Analyzing Market Regime...")

        df = self.fetch_recent_data(symbol, days=30)

        if len(df) < 50:
            print("⚠️  Insufficient data for regime detection")
            return None

        # Use unified RegimeClassifier for indicators + classification
        df = RegimeClassifier.compute_indicators(df)

        # Get latest values
        latest = df.iloc[-1]

        current_price = latest['close']
        adx = latest['adx']
        volatility_pct = latest['volatility_pct']
        trend_direction = latest['trend_direction']

        # Price change over 30 days
        price_change_30d = ((current_price - df.iloc[0]['close']) / df.iloc[0]['close']) * 100

        # Classify using the canonical function (hourly candles)
        regime_type = RegimeClassifier.classify_bar(
            float(adx) if pd.notna(adx) else 0,
            float(volatility_pct),
            trend_direction,
            timeframe='hourly'
        )

        # Build regime dict
        regime = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'current_price': float(current_price),
            'price_change_30d': float(price_change_30d),
            'volatility_pct': float(volatility_pct),
            'adx': float(adx) if not pd.isna(adx) else 0,
            'trend_direction': trend_direction,
            'trend_strength': None,
            'regime_type': regime_type,
            'recommended_strategy_types': []
        }

        # Trend strength (using ADX)
        if pd.notna(adx):
            if adx > 25:
                regime['trend_strength'] = 'STRONG'
            elif adx > 20:
                regime['trend_strength'] = 'MODERATE'
            else:
                regime['trend_strength'] = 'WEAK'
        else:
            regime['trend_strength'] = 'UNKNOWN'

        # Recommended strategies based on regime type
        if regime_type == 'TRENDING':
            regime['recommended_strategy_types'] = ['Trend Following', 'Momentum', 'Breakout']
        elif regime_type == 'VOLATILE':
            regime['recommended_strategy_types'] = ['Volatility Breakout', 'Momentum', 'Mean Reversion']
        else:
            regime['recommended_strategy_types'] = ['Mean Reversion', 'Oscillator-based', 'Range Trading']

        return regime
    
    def display_regime(self, regime):
        """Display regime analysis"""
        print("\n" + "="*80)
        print("🌍 MARKET REGIME ANALYSIS")
        print("="*80)
        print(f"Symbol: {regime['symbol']}")
        print(f"Current Price: ${regime['current_price']:,.2f}")
        print(f"30-Day Change: {regime['price_change_30d']:+.2f}%")
        print(f"Volatility: {regime['volatility_pct']:.2f}%")
        print(f"ADX (Trend Strength): {regime['adx']:.1f}")
        print()
        print(f"📊 Trend Direction: {regime['trend_direction']}")
        print(f"💪 Trend Strength: {regime['trend_strength']}")
        print(f"🏷️  Regime Type: {regime['regime_type']}")
        print()
        print("✅ Recommended Strategy Types:")
        for strategy_type in regime['recommended_strategy_types']:
            print(f"   • {strategy_type}")
        print("="*80)

def main():
    detector = MarketRegimeDetector()
    regime = detector.detect_regime('BTC-USD')
    
    if regime:
        detector.display_regime(regime)
        
        # Save regime data
        import json
        with open('current_regime.json', 'w') as f:
            json.dump(regime, f, indent=2)
        print("\n💾 Regime data saved to: current_regime.json")

if __name__ == "__main__":
    main()