"""Backtest strategies only on matching market regimes"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from market_regime import RegimeClassifier

class RegimeBacktester:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
    
    def fetch_historical_data(self, symbol='BTC-USD', days=90, granularity=3600):
        """Fetch historical candles from Coinbase"""
        print(f"📊 Fetching {days} days of {symbol} data...")
        
        all_data = []
        chunk_size = 300
        
        end_time = datetime.now()
        
        for i in range(days // 12):
            chunk_end = end_time - timedelta(hours=i * chunk_size)
            chunk_start = chunk_end - timedelta(hours=chunk_size)
            
            url = f"{self.coinbase_api}/products/{symbol}/candles"
            params = {
                'start': chunk_start.isoformat(),
                'end': chunk_end.isoformat(),
                'granularity': granularity
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        all_data.extend(data)
            except Exception as e:
                pass
        
        df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} total candles")
        return df
    
    def calculate_williams_r(self, df, period=21):
        """Calculate Williams %R indicator"""
        high_roll = df['high'].rolling(window=period).max()
        low_roll = df['low'].rolling(window=period).min()
        
        wr = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
        return wr
    
    def calculate_sma(self, df, period=21):
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()
    
    def backtest_williams_r_ranging(self, df):
        """Backtest Williams %R strategy ONLY on RANGING periods"""
        print("\n🧪 Backtesting Williams %R on RANGING periods only...")
        
        # Classify regimes using unified classifier
        print("🔍 Classifying historical regimes...")
        df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

        # Calculate strategy-specific indicators
        df['williams_r'] = self.calculate_williams_r(df, 14)
        df['sma_21'] = self.calculate_sma(df, 21)
        
        # Count regimes
        regime_counts = df['regime'].value_counts()
        print(f"   RANGING periods: {regime_counts.get('RANGING', 0)}")
        print(f"   TRENDING periods: {regime_counts.get('TRENDING', 0)}")
        print(f"   UNKNOWN periods: {regime_counts.get('UNKNOWN', 0)}")
        
        # Remove NaN and UNKNOWN
        df = df.dropna()
        
        # Trading logic - ONLY on RANGING periods
        position = None
        trades = []
        entry_price = 0
        entry_time = None
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # CRITICAL: Only trade in RANGING regime
            if current['regime'] != 'RANGING':
                # Exit any open position if regime changes
                if position == 'LONG':
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl),
                        'exit_reason': 'Regime changed to ' + current['regime'],
                        'win': bool(pnl > 0)
                    })
                    position = None
                continue
            
            # Entry conditions (Williams %R crosses above -80, price below SMA)
            if position is None:
                if (current['williams_r'] > -80 and 
                    prev['williams_r'] <= -80 and
                    current['close'] < current['sma_21']):
                    
                    position = 'LONG'
                    entry_price = float(current['close'])
                    entry_time = current['time']
            
            # Exit conditions
            elif position == 'LONG':
                pnl_pct = ((current['close'] - entry_price) / entry_price) * 100
                
                exit_reason = None
                if current['williams_r'] < -20:
                    exit_reason = "Williams %R below -20 (overbought)"
                elif current['close'] >= current['sma_21'] * 1.015:
                    exit_reason = "Price reached SMA + 1.5%"
                
                if exit_reason:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl_pct),
                        'exit_reason': exit_reason,
                        'win': bool(pnl_pct > 0)
                    })
                    
                    position = None
        
        # Calculate metrics
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['win'])
            losses = len(trades) - wins
            win_rate = wins / len(trades)
            
            avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0
            
            total_pnl = sum(t['pnl_pct'] for t in trades)
            
            total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
            total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
            profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
            
            return {
                'strategy': 'Williams %R Mean Reversion',
                'regime_tested': 'RANGING',
                'total_periods': len(df),
                'ranging_periods': int(regime_counts.get('RANGING', 0)),
                'trending_periods': int(regime_counts.get('TRENDING', 0)),
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor,
                'trades': trades
            }
        else:
            return None

    def calculate_adx_indicators(self, df, period=14, sma_period=50):
        """Calculate ADX, +DI, -DI, RSI, ATR, SMA(50) for ADX Momentum strategy"""
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
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # --- +DI / -DI ---
        df['plus_di'] = 100 * smooth_plus_dm / atr
        df['minus_di'] = 100 * smooth_minus_dm / atr

        # --- DX and ADX ---
        dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # --- RSI ---
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # --- ATR ---
        df['atr'] = atr

        # --- SMA(50) ---
        df['sma_50'] = df['close'].rolling(window=sma_period).mean()

        return df

    def backtest_adx_momentum_trending(self, df):
        """Backtest ADX Momentum Thrust strategy ONLY on TRENDING periods"""
        print("\n🧪 Backtesting ADX Momentum Thrust on TRENDING periods only...")

        # Classify regimes using unified classifier
        print("🔍 Classifying historical regimes...")
        df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

        # Calculate strategy-specific indicators (overwrites adx with strategy ADX)
        df = self.calculate_adx_indicators(df)

        # Count regimes
        regime_counts = df['regime'].value_counts()
        print(f"   TRENDING periods: {regime_counts.get('TRENDING', 0)}")
        print(f"   RANGING periods: {regime_counts.get('RANGING', 0)}")
        print(f"   UNKNOWN periods: {regime_counts.get('UNKNOWN', 0)}")

        # Remove NaN and UNKNOWN
        df = df.dropna()

        # Trading logic - ONLY on TRENDING periods
        position = None
        trades = []
        entry_price = 0
        entry_time = None
        high_watermark = 0
        non_trending_count = 0  # Track consecutive non-TRENDING bars

        for i in range(5, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Track regime stability
            if current['regime'] != 'TRENDING':
                non_trending_count += 1
            else:
                non_trending_count = 0

            # Only force-exit after 3 consecutive non-TRENDING bars (regime stability buffer)
            if current['regime'] != 'TRENDING':
                if position == 'LONG' and non_trending_count >= 3:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    pnl = ((exit_price - entry_price) / entry_price) * 100

                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl),
                        'exit_reason': 'Regime changed to ' + current['regime'] + ' (3-bar confirmed)',
                        'win': bool(pnl > 0)
                    })
                    position = None
                # Don't enter new positions while non-TRENDING
                if position is None:
                    continue
                # If in position but not yet 3 bars, let strategy exits handle it below

            if position is None and current['regime'] == 'TRENDING':
                # --- BUY CONDITIONS ---
                # 1. ADX > 20 and rising over last 5 bars
                adx_rising = df.iloc[i]['adx'] > df.iloc[i - 5]['adx']
                adx_above_20 = current['adx'] > 20

                # 2. +DI > -DI
                plus_di_above = current['plus_di'] > current['minus_di']

                # 3. +DI crossed above -DI within last 15 bars
                di_cross_recent = False
                for offset in range(0, 15):
                    idx = i - offset
                    if idx >= 1:
                        bar = df.iloc[idx]
                        bar_prev = df.iloc[idx - 1]
                        if bar['plus_di'] > bar['minus_di'] and bar_prev['plus_di'] <= bar_prev['minus_di']:
                            di_cross_recent = True
                            break

                # 4. RSI between 35-78
                rsi_ok = 35 <= current['rsi'] <= 78

                # 5. Price above SMA(50)
                price_above_sma = current['close'] > current['sma_50']

                if adx_above_20 and adx_rising and plus_di_above and di_cross_recent and rsi_ok and price_above_sma:
                    position = 'LONG'
                    entry_price = float(current['close'])
                    entry_time = current['time']
                    high_watermark = float(current['close'])

            # --- SELL CONDITIONS (any one triggers) ---
            elif position == 'LONG':
                # Update trailing stop high watermark
                high_watermark = max(high_watermark, float(current['close']))

                exit_reason = None

                # 1. ADX drops below 20 (trend exhausted)
                if current['adx'] < 20:
                    exit_reason = 'ADX dropped below 20'

                # 2. RSI > 75 (overbought — take profit)
                elif current['rsi'] > 75:
                    exit_reason = 'RSI > 75 (overbought)'

                # 3. Trailing stop: price drops 1.5x ATR below high watermark
                elif current['close'] <= high_watermark - (1.5 * current['atr']):
                    exit_reason = 'Trailing stop hit (1.5x ATR from high)'

                # 4. -DI sustained above +DI for 3 bars (confirmed reversal)
                elif i >= 3:
                    di_bearish_count = 0
                    for j in range(3):
                        if df.iloc[i - j]['minus_di'] > df.iloc[i - j]['plus_di']:
                            di_bearish_count += 1
                    if di_bearish_count == 3:
                        exit_reason = '-DI above +DI for 3 bars (confirmed reversal)'

                if exit_reason:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    pnl = ((exit_price - entry_price) / entry_price) * 100

                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl),
                        'exit_reason': exit_reason,
                        'win': bool(pnl > 0)
                    })
                    position = None

        # Calculate metrics
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['win'])
            losses = len(trades) - wins
            win_rate = wins / len(trades)

            avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0

            total_pnl = sum(t['pnl_pct'] for t in trades)

            total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
            total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
            profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

            return {
                'strategy': 'ADX Momentum Thrust',
                'regime_tested': 'TRENDING',
                'total_periods': len(df),
                'trending_periods': int(regime_counts.get('TRENDING', 0)),
                'ranging_periods': int(regime_counts.get('RANGING', 0)),
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor,
                'trades': trades
            }
        else:
            return None

    def calculate_bb_indicators(self, df, bb_period=20, bb_sigma=2.0, rsi_period=14, atr_period=14):
        """Calculate Bollinger Bands, RSI, ATR for BB Mean Reversion strategy"""
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

    def backtest_bb_reversion_volatile(self, df):
        """Backtest Bollinger Band Mean Reversion strategy ONLY on VOLATILE periods"""
        print("\n🧪 Backtesting BB Mean Reversion on VOLATILE periods only...")

        # Classify regimes using unified classifier
        print("🔍 Classifying historical regimes...")
        df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

        # Calculate strategy-specific indicators
        df = self.calculate_bb_indicators(df)

        # Count regimes
        regime_counts = df['regime'].value_counts()
        print(f"   VOLATILE periods: {regime_counts.get('VOLATILE', 0)}")
        print(f"   TRENDING periods: {regime_counts.get('TRENDING', 0)}")
        print(f"   RANGING periods: {regime_counts.get('RANGING', 0)}")

        # Remove NaN and UNKNOWN
        df = df.dropna()

        # Trading logic - ONLY on VOLATILE periods
        position = None
        trades = []
        entry_price = 0
        entry_time = None
        non_volatile_count = 0  # 3-bar stability buffer

        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Track regime stability
            if current['regime'] != 'VOLATILE':
                non_volatile_count += 1
            else:
                non_volatile_count = 0

            # Force-exit on confirmed regime change (3 bars non-VOLATILE)
            if current['regime'] != 'VOLATILE':
                if position == 'LONG' and non_volatile_count >= 3:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    pnl = ((exit_price - entry_price) / entry_price) * 100

                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl),
                        'exit_reason': 'Regime changed to ' + current['regime'] + ' (3-bar confirmed)',
                        'win': bool(pnl > 0)
                    })
                    position = None
                if position is None:
                    continue

            # --- BUY CONDITIONS ---
            if position is None and current['regime'] == 'VOLATILE':
                # 1. Close below lower Bollinger Band
                close_below_lower_bb = current['close'] < current['bb_lower']

                # 2. RSI < 35 (oversold)
                rsi_oversold = current['rsi'] < 35

                if close_below_lower_bb and rsi_oversold:
                    position = 'LONG'
                    entry_price = float(current['close'])
                    entry_time = current['time']

            # --- SELL CONDITIONS ---
            elif position == 'LONG':
                exit_reason = None

                # 1. Price reaches middle BB (SMA 20) — profit target
                if current['close'] >= current['sma_20']:
                    exit_reason = 'Price reached middle BB (SMA 20)'

                # 2. RSI > 70 (overbought)
                elif current['rsi'] > 70:
                    exit_reason = 'RSI > 70 (overbought)'

                # 3. Stop loss: price drops 1.5x ATR below entry
                elif current['close'] <= entry_price - (1.5 * current['atr']):
                    exit_reason = 'Stop loss hit (1.5x ATR below entry)'

                if exit_reason:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    pnl = ((exit_price - entry_price) / entry_price) * 100

                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl_pct': float(pnl),
                        'exit_reason': exit_reason,
                        'win': bool(pnl > 0)
                    })
                    position = None

        # Calculate metrics
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['win'])
            losses = len(trades) - wins
            win_rate = wins / len(trades)

            avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0

            total_pnl = sum(t['pnl_pct'] for t in trades)

            total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
            total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
            profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

            return {
                'strategy': 'Bollinger Band Mean Reversion',
                'regime_tested': 'VOLATILE',
                'total_periods': len(df),
                'volatile_periods': int(regime_counts.get('VOLATILE', 0)),
                'trending_periods': int(regime_counts.get('TRENDING', 0)),
                'ranging_periods': int(regime_counts.get('RANGING', 0)),
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor,
                'trades': trades
            }
        else:
            return None

def main():
    print("="*80)
    print("🧪 REGIME-AWARE BACKTESTER")
    print("="*80 + "\n")
    
    backtester = RegimeBacktester()
    
    # Fetch data
    df = backtester.fetch_historical_data('BTC-USD', days=90, granularity=3600)
    
    # Run regime-specific backtest
    results = backtester.backtest_williams_r_ranging(df)
    
    if results:
        print("\n" + "="*80)
        print("📊 REGIME-AWARE BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: {results['strategy']}")
        print(f"Tested On Regime: {results['regime_tested']}")
        print(f"\nHistorical Period Analysis:")
        print(f"  Total Periods: {results['total_periods']}")
        print(f"  RANGING Periods: {results['ranging_periods']} ({results['ranging_periods']/results['total_periods']*100:.1f}%)")
        print(f"  TRENDING Periods: {results['trending_periods']} ({results['trending_periods']/results['total_periods']*100:.1f}%)")
        print(f"\nTrading Results (RANGING periods only):")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Win Rate: {results['win_rate']*100:.1f}%")
        print(f"  Average Win: {results['avg_win']:.2f}%")
        print(f"  Average Loss: {results['avg_loss']:.2f}%")
        print(f"  Total P&L: {results['total_pnl']:.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print("="*80)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_ranging_{timestamp}.json"
        
        results_copy = results.copy()
        for trade in results_copy['trades']:
            trade['entry_time'] = str(trade['entry_time'])
            trade['exit_time'] = str(trade['exit_time'])
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n💾 Full results saved to: {filename}")
        
        # Decision — two-path approval
        print("\n🎯 RECOMMENDATION:")
        wr = results['win_rate']
        pf = results['profit_factor']
        avg_w = results.get('avg_win', 0)
        avg_l = abs(results.get('avg_loss', 0)) or 1
        n_trades = results['total_trades']
        path_a = wr >= 0.55 and pf >= 1.5
        path_b = pf >= 1.8 and (avg_w / avg_l) >= 1.5 and n_trades >= 10
        if path_a or path_b:
            path = "A (win-rate)" if path_a else "B (trend-following)"
            print(f"✅ STRATEGY APPROVED for {results['regime_tested']} markets via path {path}!")
            print(f"   Win Rate: {wr*100:.1f}% | PF: {pf:.2f}")
        else:
            print(f"❌ Strategy needs improvement for {results['regime_tested']} markets")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades generated in RANGING periods")

    # --- ADX Momentum Thrust on TRENDING periods (needs more data) ---
    print("\n\n" + "="*80)
    print("🧪 ADX MOMENTUM THRUST - TRENDING REGIME BACKTEST")
    print("="*80 + "\n")

    print("📊 Fetching 365 days of hourly data for ADX backtest...")
    df_adx = backtester.fetch_historical_data('BTC-USD', days=365, granularity=3600)
    adx_results = backtester.backtest_adx_momentum_trending(df_adx)

    if adx_results:
        print("\n" + "="*80)
        print("📊 ADX MOMENTUM THRUST BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: {adx_results['strategy']}")
        print(f"Tested On Regime: {adx_results['regime_tested']}")
        print(f"\nHistorical Period Analysis:")
        print(f"  Total Periods: {adx_results['total_periods']}")
        print(f"  TRENDING Periods: {adx_results['trending_periods']} ({adx_results['trending_periods']/adx_results['total_periods']*100:.1f}%)")
        print(f"  RANGING Periods: {adx_results['ranging_periods']} ({adx_results['ranging_periods']/adx_results['total_periods']*100:.1f}%)")
        print(f"\nTrading Results (TRENDING periods only):")
        print(f"  Total Trades: {adx_results['total_trades']}")
        print(f"  Wins: {adx_results['wins']}")
        print(f"  Losses: {adx_results['losses']}")
        print(f"  Win Rate: {adx_results['win_rate']*100:.1f}%")
        print(f"  Average Win: {adx_results['avg_win']:.2f}%")
        print(f"  Average Loss: {adx_results['avg_loss']:.2f}%")
        print(f"  Total P&L: {adx_results['total_pnl']:.2f}%")
        print(f"  Profit Factor: {adx_results['profit_factor']:.2f}")
        print("="*80)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_trending_{timestamp}.json"

        adx_copy = adx_results.copy()
        for trade in adx_copy['trades']:
            trade['entry_time'] = str(trade['entry_time'])
            trade['exit_time'] = str(trade['exit_time'])

        with open(filename, 'w') as f:
            json.dump(adx_copy, f, indent=2)

        print(f"\n💾 Full results saved to: {filename}")

        # Print individual trades
        print("\n📋 TRADE LOG:")
        for idx, trade in enumerate(adx_results['trades'], 1):
            win_marker = "✅" if trade['win'] else "❌"
            print(f"  {idx}. {win_marker} Entry: ${trade['entry_price']:,.2f} → Exit: ${trade['exit_price']:,.2f} | P&L: {trade['pnl_pct']:+.2f}% | {trade['exit_reason']}")

        # Decision — two-path approval
        print("\n🎯 RECOMMENDATION:")
        wr = adx_results['win_rate']
        pf = adx_results['profit_factor']
        avg_w = adx_results.get('avg_win', 0)
        avg_l = abs(adx_results.get('avg_loss', 0)) or 1
        n_trades = adx_results['total_trades']
        path_a = wr >= 0.55 and pf >= 1.5
        path_b = pf >= 1.8 and (avg_w / avg_l) >= 1.5 and n_trades >= 10
        if path_a or path_b:
            path = "A (win-rate)" if path_a else "B (trend-following)"
            print(f"✅ STRATEGY APPROVED for {adx_results['regime_tested']} markets via path {path}!")
            print(f"   Win Rate: {wr*100:.1f}% | PF: {pf:.2f}")
        else:
            print(f"❌ Strategy needs improvement for {adx_results['regime_tested']} markets")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades generated in TRENDING periods")

    # --- BB Mean Reversion on VOLATILE periods ---
    print("\n\n" + "="*80)
    print("🧪 BB MEAN REVERSION - VOLATILE REGIME BACKTEST")
    print("="*80 + "\n")

    print("📊 Fetching 365 days of hourly data for BB backtest...")
    df_bb = backtester.fetch_historical_data('BTC-USD', days=365, granularity=3600)
    bb_results = backtester.backtest_bb_reversion_volatile(df_bb)

    if bb_results:
        print("\n" + "="*80)
        print("📊 BB MEAN REVERSION BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: {bb_results['strategy']}")
        print(f"Tested On Regime: {bb_results['regime_tested']}")
        print(f"\nHistorical Period Analysis:")
        print(f"  Total Periods: {bb_results['total_periods']}")
        print(f"  VOLATILE Periods: {bb_results['volatile_periods']} ({bb_results['volatile_periods']/bb_results['total_periods']*100:.1f}%)")
        print(f"  TRENDING Periods: {bb_results['trending_periods']} ({bb_results['trending_periods']/bb_results['total_periods']*100:.1f}%)")
        print(f"  RANGING Periods: {bb_results['ranging_periods']} ({bb_results['ranging_periods']/bb_results['total_periods']*100:.1f}%)")
        print(f"\nTrading Results (VOLATILE periods only):")
        print(f"  Total Trades: {bb_results['total_trades']}")
        print(f"  Wins: {bb_results['wins']}")
        print(f"  Losses: {bb_results['losses']}")
        print(f"  Win Rate: {bb_results['win_rate']*100:.1f}%")
        print(f"  Average Win: {bb_results['avg_win']:.2f}%")
        print(f"  Average Loss: {bb_results['avg_loss']:.2f}%")
        print(f"  Total P&L: {bb_results['total_pnl']:.2f}%")
        print(f"  Profit Factor: {bb_results['profit_factor']:.2f}")
        print("="*80)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_volatile_{timestamp}.json"

        bb_copy = bb_results.copy()
        for trade in bb_copy['trades']:
            trade['entry_time'] = str(trade['entry_time'])
            trade['exit_time'] = str(trade['exit_time'])

        with open(filename, 'w') as f:
            json.dump(bb_copy, f, indent=2)

        print(f"\n💾 Full results saved to: {filename}")

        # Print individual trades
        print("\n📋 TRADE LOG:")
        for idx, trade in enumerate(bb_results['trades'], 1):
            win_marker = "✅" if trade['win'] else "❌"
            print(f"  {idx}. {win_marker} Entry: ${trade['entry_price']:,.2f} → Exit: ${trade['exit_price']:,.2f} | P&L: {trade['pnl_pct']:+.2f}% | {trade['exit_reason']}")

        # Decision — two-path approval
        print("\n🎯 RECOMMENDATION:")
        wr = bb_results['win_rate']
        pf = bb_results['profit_factor']
        avg_w = bb_results.get('avg_win', 0)
        avg_l = abs(bb_results.get('avg_loss', 0)) or 1
        n_trades = bb_results['total_trades']
        path_a = wr >= 0.55 and pf >= 1.5
        path_b = pf >= 1.8 and (avg_w / avg_l) >= 1.5 and n_trades >= 10
        if path_a or path_b:
            path = "A (win-rate)" if path_a else "B (trend-following)"
            print(f"✅ STRATEGY APPROVED for {bb_results['regime_tested']} markets via path {path}!")
            print(f"   Win Rate: {wr*100:.1f}% | PF: {pf:.2f}")
        else:
            print(f"❌ Strategy needs improvement for {bb_results['regime_tested']} markets")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades generated in VOLATILE periods")

if __name__ == "__main__":
    main()