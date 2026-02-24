"""Backtest Adaptive Breakout on 4-year STRONG trending periods"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

class BreakoutBacktester:
    def __init__(self):
        self.classified_data_file = 'classified_regimes.json'
    
    def load_classified_data(self):
        """Load pre-classified 4-year data"""
        print("📂 Loading 4-year classified data...")
        
        df = pd.read_json(self.classified_data_file)
        df['time'] = pd.to_datetime(df['time'])
        
        print(f"✅ Loaded {len(df)} days")
        print(f"   Range: {df['time'].min().date()} to {df['time'].max().date()}")
        
        return df
    
    def calculate_donchian(self, df, period=20):
        """Calculate Donchian Channels"""
        df['donchian_upper'] = df['high'].rolling(window=period).max()
        df['donchian_lower'] = df['low'].rolling(window=period).min()
        df['donchian_mid'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        return df
    
    def calculate_roc(self, df, period=10):
        """Calculate Rate of Change"""
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return df
    
    def backtest_adaptive_breakout(self, df):
        """Backtest on TRENDING periods only"""
        print("\n🧪 Backtesting Adaptive Breakout on TRENDING periods...")
        
        # Filter to TRENDING periods only
        trending_df = df[df['regime'] == 'TRENDING'].copy()
        
        print(f"   Testing on {len(trending_df)} TRENDING days (out of {len(df)} total)")
        
        if len(trending_df) < 100:
            print("❌ Not enough trending data")
            return None
        
        # Calculate indicators
        trending_df = self.calculate_donchian(trending_df, 20)
        trending_df = self.calculate_roc(trending_df, 10)
        
        # ADX already in data
        # Volume oscillator (simplified)
        trending_df['volume_sma'] = trending_df['volume'].rolling(20).mean()
        trending_df['volume_osc'] = ((trending_df['volume'] - trending_df['volume_sma']) / trending_df['volume_sma']) * 100
        
        trending_df = trending_df.dropna()
        
        # Trading logic
        position = None
        position_type = None
        trades = []
        entry_price = 0
        entry_time = None
        highest_price = 0  # For trailing stop
        lowest_price = 0
        
        for i in range(1, len(trending_df)):
            current = trending_df.iloc[i]
            prev = trending_df.iloc[i-1]
            
            # Entry conditions
            if position is None:
                # LONG: Break Donchian upper + ADX > 25 + ROC > 5% + Volume positive
                if (current['high'] > prev['donchian_upper'] and
                    current['adx'] > 25 and
                    current['roc'] > 5.0 and
                    current['volume_osc'] > 0):
                    
                    position = 'ACTIVE'
                    position_type = 'LONG'
                    entry_price = float(current['close'])
                    entry_time = current['time']
                    highest_price = entry_price
                
                # SHORT: Break Donchian lower + ADX > 25 + ROC < -5% + Volume positive
                elif (current['low'] < prev['donchian_lower'] and
                      current['adx'] > 25 and
                      current['roc'] < -5.0 and
                      current['volume_osc'] > 0):
                    
                    position = 'ACTIVE'
                    position_type = 'SHORT'
                    entry_price = float(current['close'])
                    entry_time = current['time']
                    lowest_price = entry_price
            
            # Exit conditions
            elif position:
                exit_reason = None
                
                if position_type == 'LONG':
                    # Update highest price for trailing stop
                    if current['high'] > highest_price:
                        highest_price = float(current['high'])
                    
                    pnl = ((current['close'] - entry_price) / entry_price) * 100
                    
                    # Trailing stop (4%)
                    trailing_stop = highest_price * 0.96
                    
                    # Exit conditions
                    if current['close'] < trailing_stop:
                        exit_reason = "4% trailing stop hit"
                    elif current['adx'] < prev['adx'] and current['adx'] < 25:
                        exit_reason = "ADX declining below 25"
                    elif abs(current['roc']) < 2:  # ROC weakening
                        exit_reason = "ROC momentum weakening"
                    elif current['close'] < current['donchian_mid']:
                        exit_reason = "Price re-entered Donchian channel"
                
                else:  # SHORT
                    if current['low'] < lowest_price:
                        lowest_price = float(current['low'])
                    
                    pnl = ((entry_price - current['close']) / entry_price) * 100
                    
                    trailing_stop = lowest_price * 1.04
                    
                    if current['close'] > trailing_stop:
                        exit_reason = "4% trailing stop hit"
                    elif current['adx'] < prev['adx'] and current['adx'] < 25:
                        exit_reason = "ADX declining"
                    elif abs(current['roc']) < 2:
                        exit_reason = "ROC weakening"
                    elif current['close'] > current['donchian_mid']:
                        exit_reason = "Re-entered channel"
                
                if exit_reason:
                    exit_price = float(current['close'])
                    exit_time = current['time']
                    
                    trades.append({
                        'entry_time': str(entry_time),
                        'entry_price': float(entry_price),
                        'exit_time': str(exit_time),
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
            
            long_trades = [t for t in trades if t['position_type'] == 'LONG']
            short_trades = [t for t in trades if t['position_type'] == 'SHORT']
            
            return {
                'strategy': 'Adaptive Breakout with Momentum Filter',
                'regime_tested': 'TRENDING',
                'data_period': f"{trending_df['time'].min().date()} to {trending_df['time'].max().date()}",
                'trending_days_tested': len(trending_df),
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
    print("🧪 ADAPTIVE BREAKOUT BACKTESTER (4-Year Data)")
    print("="*80 + "\n")
    
    backtester = BreakoutBacktester()
    df = backtester.load_classified_data()
    results = backtester.backtest_adaptive_breakout(df)
    
    if results:
        print("\n" + "="*80)
        print("📊 4-YEAR BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: {results['strategy']}")
        print(f"Regime: {results['regime_tested']}")
        print(f"Period: {results['data_period']}")
        print(f"Trending Days Tested: {results['trending_days_tested']}")
        print(f"\nTrading Results:")
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
        filename = f"backtest_breakout_4year_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Saved to: {filename}")
        
        # Recommendation
        # Two-path approval
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
            print(f"✅ STRATEGY APPROVED via path {path}!")
            print(f"   Win Rate: {wr*100:.1f}% | PF: {pf:.2f}")
            print(f"   Ready to add to strategy library!")
        else:
            print(f"❌ Needs improvement")
            if wr < 0.55:
                print(f"   Win Rate: {wr*100:.1f}% (path A needs 55%+)")
            if pf < 1.8:
                print(f"   Profit Factor: {pf:.2f} (path B needs 1.8+)")
            if n_trades < 10:
                print(f"   Trades: {n_trades} (path B needs 10+)")
    else:
        print("❌ No trades generated")

if __name__ == "__main__":
    main()