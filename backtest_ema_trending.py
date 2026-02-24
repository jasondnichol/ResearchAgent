"""Backtest EMA Crossover strategy on TRENDING periods only"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from market_regime import RegimeClassifier

class EMATrendingBacktester:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
    
    def fetch_historical_data(self, symbol='BTC-USD', days=90, granularity=3600):
        """Fetch historical candles"""
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
            except:
                pass
        
        df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} candles")
        return df
    
    def calculate_ema(self, df, period):
        """Calculate EMA"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def backtest_ema_crossover(self, df):
        """Backtest EMA Crossover on TRENDING periods"""
        print("\n🧪 Backtesting EMA Crossover on TRENDING periods...")
        
        # Classify regimes using unified classifier
        print("🔍 Classifying historical regimes...")
        df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

        # Calculate strategy-specific indicators
        df['ema_12'] = self.calculate_ema(df, 12)
        df['ema_26'] = self.calculate_ema(df, 26)
        df['rsi'] = self.calculate_rsi(df, 14)
        
        regime_counts = df['regime'].value_counts()
        print(f"   RANGING periods: {regime_counts.get('RANGING', 0)}")
        print(f"   TRENDING periods: {regime_counts.get('TRENDING', 0)}")
        
        df = df.dropna()
        
        # Trading logic - ONLY on TRENDING periods
        position = None
        position_type = None  # 'LONG' or 'SHORT'
        trades = []
        entry_price = 0
        entry_time = None
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Only trade in TRENDING regime
            if current['regime'] != 'TRENDING':
                # Exit any position if regime changes
                if position:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    
                    if position_type == 'LONG':
                        pnl = ((exit_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        pnl = ((entry_price - exit_price) / entry_price) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'position_type': position_type,
                        'pnl_pct': float(pnl),
                        'exit_reason': f'Regime changed to {current["regime"]}',
                        'win': bool(pnl > 0)
                    })
                    position = None
                    position_type = None
                continue
            
            # Entry conditions
            if position is None:
                # LONG entry: EMA 12 crosses above EMA 26 AND RSI > 50
                if (prev['ema_12'] <= prev['ema_26'] and 
                    current['ema_12'] > current['ema_26'] and
                    current['rsi'] > 50):
                    
                    position = 'ACTIVE'
                    position_type = 'LONG'
                    entry_price = float(current['close'])
                    entry_time = current['time']
                
                # SHORT entry: EMA 12 crosses below EMA 26 AND RSI < 50
                elif (prev['ema_12'] >= prev['ema_26'] and 
                      current['ema_12'] < current['ema_26'] and
                      current['rsi'] < 50):
                    
                    position = 'ACTIVE'
                    position_type = 'SHORT'
                    entry_price = float(current['close'])
                    entry_time = current['time']
            
            # Exit conditions
            elif position:
                exit_reason = None
                
                if position_type == 'LONG':
                    pnl = ((current['close'] - entry_price) / entry_price) * 100
                    
                    # Exit if EMA crosses back OR RSI overbought
                    if current['ema_12'] < current['ema_26']:
                        exit_reason = "EMA crossover reversal"
                    elif current['rsi'] > 80:
                        exit_reason = "RSI overbought (>80)"
                
                else:  # SHORT
                    pnl = ((entry_price - current['close']) / entry_price) * 100
                    
                    # Exit if EMA crosses back OR RSI oversold
                    if current['ema_12'] > current['ema_26']:
                        exit_reason = "EMA crossover reversal"
                    elif current['rsi'] < 20:
                        exit_reason = "RSI oversold (<20)"
                
                if exit_reason:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'position_type': position_type,
                        'pnl_pct': float(pnl),
                        'exit_reason': exit_reason,
                        'win': bool(pnl > 0)
                    })
                    
                    position = None
                    position_type = None
        
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
            
            # Count long vs short
            long_trades = [t for t in trades if t['position_type'] == 'LONG']
            short_trades = [t for t in trades if t['position_type'] == 'SHORT']
            
            return {
                'strategy': 'EMA Crossover Momentum',
                'regime_tested': 'TRENDING',
                'total_periods': len(df),
                'ranging_periods': int(regime_counts.get('RANGING', 0)),
                'trending_periods': int(regime_counts.get('TRENDING', 0)),
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
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
    print("🧪 EMA CROSSOVER TRENDING BACKTESTER")
    print("="*80 + "\n")
    
    backtester = EMATrendingBacktester()
    df = backtester.fetch_historical_data('BTC-USD', days=90, granularity=3600)
    results = backtester.backtest_ema_crossover(df)
    
    if results:
        print("\n" + "="*80)
        print("📊 TRENDING BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: {results['strategy']}")
        print(f"Tested On: {results['regime_tested']} periods")
        print(f"\nHistorical Analysis:")
        print(f"  RANGING: {results['ranging_periods']} ({results['ranging_periods']/results['total_periods']*100:.1f}%)")
        print(f"  TRENDING: {results['trending_periods']} ({results['trending_periods']/results['total_periods']*100:.1f}%)")
        print(f"\nTrading Results (TRENDING only):")
        print(f"  Total Trades: {results['total_trades']} (Long: {results['long_trades']}, Short: {results['short_trades']})")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Win Rate: {results['win_rate']*100:.1f}%")
        print(f"  Average Win: {results['avg_win']:.2f}%")
        print(f"  Average Loss: {results['avg_loss']:.2f}%")
        print(f"  Total P&L: {results['total_pnl']:.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print("="*80)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_trending_{timestamp}.json"
        
        results_copy = results.copy()
        for trade in results_copy['trades']:
            trade['entry_time'] = str(trade['entry_time'])
            trade['exit_time'] = str(trade['exit_time'])
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n💾 Saved to: {filename}")
        
        # Recommendation — two-path approval
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
            print(f"❌ Strategy needs improvement")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades in TRENDING periods")

if __name__ == "__main__":
    main()