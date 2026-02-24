"""Backtest trading strategies on historical data"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class Backtester:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
    
    def fetch_historical_data(self, symbol='BTC-USD', days=90, granularity=3600):
        """Fetch historical candles from Coinbase"""
        print(f"📊 Fetching {days} days of {symbol} data...")
        
        # Coinbase limits to 300 candles per request
        # So we'll fetch in chunks
        all_data = []
        chunk_size = 300
        
        end_time = datetime.now()
        
        for i in range(days // 12):  # 12 days per chunk at 1hr granularity
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
                        print(f"  Loaded chunk {i+1}: {len(data)} candles")
            except Exception as e:
                print(f"  Error fetching chunk {i+1}: {e}")
        
        if not all_data:
            print("❌ No data fetched. Using fallback method...")
            # Fallback: just get recent data without date range
            url = f"{self.coinbase_api}/products/{symbol}/candles"
            params = {'granularity': granularity}
            response = requests.get(url, params=params)
            all_data = response.json() if response.status_code == 200 else []
        
        # Convert to DataFrame
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
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def calculate_std_bands(self, df, period=20, std_dev=1.5):
        """Calculate Standard Deviation Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def backtest_williams_r_strategy(self, df):
        """Backtest the Williams %R Mean Reversion strategy"""
        print("\n🧪 Backtesting Williams %R Mean Reversion Strategy...")
        
        # Calculate indicators
        df['williams_r'] = self.calculate_williams_r(df, 21)
        df['vwap'] = self.calculate_vwap(df)
        df['upper_band'], df['lower_band'] = self.calculate_std_bands(df, 20, 1.5)
        
        # Remove NaN rows
        df = df.dropna()
        
        # Trading logic
        position = None
        trades = []
        entry_price = 0
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Entry conditions
            if position is None:
                if (current['williams_r'] < -80 and  # Oversold
                    current['close'] <= current['lower_band'] and  # Touch lower band
                    current['close'] > current['vwap'] and  # Above VWAP
                    current['williams_r'] > prev['williams_r']):  # Starting to rise
                    
                    position = 'LONG'
                    entry_price = current['close']
                    entry_time = current['time']
            
            # Exit conditions
            elif position == 'LONG':
                pnl_pct = ((current['close'] - entry_price) / entry_price) * 100
                
                exit_reason = None
                if current['williams_r'] > -50:
                    exit_reason = "Williams %R > -50"
                elif current['close'] >= current['upper_band']:
                    exit_reason = "Upper band reached"
                elif pnl_pct >= 2.0:
                    exit_reason = "Profit target (2%)"
                elif current['close'] < current['vwap']:
                    exit_reason = "Below VWAP"
                
                if exit_reason:
                    exit_price = current['close']
                    exit_time = current['time']
                    pnl = pnl_pct
                    
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_time,
                        'exit_price': float(exit_price),
                        'pnl_pct': float(pnl),
                        'exit_reason': exit_reason,
                        'win': bool(pnl > 0)  # <- Fixed
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
            
            profit_factor = abs(sum(t['pnl_pct'] for t in trades if t['win']) / 
                               sum(t['pnl_pct'] for t in trades if not t['win'])) if losses > 0 else float('inf')
            
            return {
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
    print("🧪 STRATEGY BACKTESTER")
    print("="*80 + "\n")
    
    backtester = Backtester()
    
    # Fetch data
    df = backtester.fetch_historical_data('BTC-USD', days=90, granularity=3600)
    
    # Run backtest
    results = backtester.backtest_williams_r_strategy(df)
    
    if results:
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Wins: {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Average Win: {results['avg_win']:.2f}%")
        print(f"Average Loss: {results['avg_loss']:.2f}%")
        print(f"Total P&L: {results['total_pnl']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print("="*80)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{timestamp}.json"
        with open(filename, 'w') as f:
            # Convert trades to serializable format
            results_copy = results.copy()
            for trade in results_copy['trades']:
                trade['entry_time'] = str(trade['entry_time'])
                trade['exit_time'] = str(trade['exit_time'])
            json.dump(results_copy, f, indent=2)
        
        print(f"\n💾 Full results saved to: {filename}")
        
        # Decision — two-path approval
        print("\n🎯 RECOMMENDATION:")
        wr = results['win_rate']
        pf = results['profit_factor']
        avg_w = results.get('avg_win', 0)
        avg_l = abs(results.get('avg_loss', 0)) or 1
        n_trades = results['total_trades']
        path_a = wr >= 0.55 and pf >= 1.5                          # mean-reversion path
        path_b = pf >= 1.8 and (avg_w / avg_l) >= 1.5 and n_trades >= 10  # trend-following path
        if path_a or path_b:
            path = "A (win-rate)" if path_a else "B (trend-following)"
            print(f"✅ STRATEGY PASSES via path {path}! Ready for deployment consideration.")
        else:
            print("❌ Strategy needs improvement or more testing.")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades generated - strategy too conservative or needs tuning")

if __name__ == "__main__":
    main()