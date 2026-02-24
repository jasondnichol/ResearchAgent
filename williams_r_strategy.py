"""Williams %R Mean Reversion Strategy Implementation"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WilliamsRStrategy:
    def __init__(self):
        self.coinbase_api = "https://api.exchange.coinbase.com"
        self.timeframe = '1H'  # 1 hour candles
        self.lookback = 100    # Need 100 candles for indicators
    
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
    
    def calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        high_roll = df['high'].rolling(window=period).max()
        low_roll = df['low'].rolling(window=period).min()
        
        wr = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
        return wr
    
    def calculate_sma(self, df, period=21):
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()
    
    def generate_signal(self, symbol='BTC-USD'):
        """Generate BUY/SELL/HOLD signal"""
        
        # Fetch data
        df = self.fetch_candles(symbol, granularity=3600, limit=100)
        
        if len(df) < 50:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}
        
        # Calculate indicators
        df['williams_r'] = self.calculate_williams_r(df, 14)
        df['sma_21'] = self.calculate_sma(df, 21)
        
        # Get current and previous values
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN
        if pd.isna(current['williams_r']) or pd.isna(current['sma_21']):
            return {'signal': 'HOLD', 'reason': 'Indicators not ready'}
        
        # Entry: Williams %R crosses above -80 AND price below SMA
        if (prev['williams_r'] <= -80 and 
            current['williams_r'] > -80 and
            current['close'] < current['sma_21']):
            
            return {
                'signal': 'BUY',
                'reason': 'Williams %R crossed above -80, price below SMA',
                'price': float(current['close']),
                'williams_r': float(current['williams_r']),
                'sma_21': float(current['sma_21']),
                'time': str(current['time'])
            }
        
        # Exit: Williams %R crosses below -20 OR price reaches SMA + 1.5%
        if (current['williams_r'] < -20 or
            current['close'] >= current['sma_21'] * 1.015):
            
            return {
                'signal': 'SELL',
                'reason': 'Williams %R overbought or target reached',
                'price': float(current['close']),
                'williams_r': float(current['williams_r']),
                'sma_21': float(current['sma_21']),
                'time': str(current['time'])
            }
        
        # Otherwise hold
        return {
            'signal': 'HOLD',
            'reason': 'No entry/exit conditions met',
            'price': float(current['close']),
            'williams_r': float(current['williams_r']),
            'sma_21': float(current['sma_21']),
            'time': str(current['time'])
        }

def main():
    """Test the strategy"""
    print("="*80)
    print("🧪 TESTING WILLIAMS %R STRATEGY")
    print("="*80 + "\n")
    
    strategy = WilliamsRStrategy()
    
    print("📊 Fetching BTC-USD 1H candles...")
    signal = strategy.generate_signal('BTC-USD')
    
    print("\n" + "="*80)
    print(f"📈 SIGNAL: {signal['signal']}")
    print("="*80)
    print(f"Reason: {signal['reason']}")
    print(f"Price: ${signal.get('price', 0):,.2f}")
    print(f"Williams %R: {signal.get('williams_r', 0):.2f}")
    print(f"SMA 21: ${signal.get('sma_21', 0):,.2f}")
    print(f"Time: {signal.get('time', 'N/A')}")
    print("="*80)

if __name__ == "__main__":
    main()