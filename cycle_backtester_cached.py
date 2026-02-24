"""Multi-year cycle backtester WITH CACHING"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
from market_regime import RegimeClassifier

class CachedCycleBacktester:
    def __init__(self, cache_file='btc_4year_cache.json'):
        self.coinbase_api = "https://api.exchange.coinbase.com"
        self.cache_file = cache_file
        self.cache_max_age_days = 7  # Re-fetch after 7 days
    
    def is_cache_valid(self):
        """Check if cache exists and is fresh"""
        if not os.path.exists(self.cache_file):
            return False
        
        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
        age_days = (datetime.now() - file_time).days
        
        if age_days > self.cache_max_age_days:
            print(f"⚠️  Cache is {age_days} days old (max: {self.cache_max_age_days})")
            return False
        
        print(f"✅ Cache found ({age_days} days old)")
        return True
    
    def load_from_cache(self):
        """Load data from cache"""
        print(f"📂 Loading from cache: {self.cache_file}")
        
        with open(self.cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Convert back to DataFrame
        df = pd.DataFrame(cache_data['data'])
        df['time'] = pd.to_datetime(df['time'])
        
        print(f"✅ Loaded {len(df)} days from cache")
        print(f"   Range: {cache_data['metadata']['start_date']} to {cache_data['metadata']['end_date']}")
        
        return df
    
    def save_to_cache(self, df):
        """Save data to cache"""
        print(f"\n💾 Saving to cache: {self.cache_file}")
        
        # Prepare cache data
        df_export = df.copy()
        df_export['time'] = df_export['time'].astype(str)
        
        cache_data = {
            'metadata': {
                'cached_at': datetime.now().isoformat(),
                'start_date': str(df['time'].min().date()),
                'end_date': str(df['time'].max().date()),
                'total_days': len(df),
                'symbol': 'BTC-USD'
            },
            'data': df_export.to_dict('records')
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"✅ Cached {len(df)} days")
    
    def fetch_multi_year_data(self, symbol='BTC-USD', years=4, granularity=86400):
        """Fetch multiple years of data"""
        print(f"\n📊 Fetching {years} years of {symbol} data from Coinbase...")
        print("⏳ This will take 2-3 minutes...")
        
        all_data = []
        end_time = datetime.now()
        total_days = years * 365
        chunks = (total_days // 300) + 1
        
        for i in range(chunks):
            chunk_end = end_time - timedelta(days=i * 300)
            chunk_start = chunk_end - timedelta(days=300)
            
            if chunk_start < (datetime.now() - timedelta(days=total_days)):
                chunk_start = datetime.now() - timedelta(days=total_days)
            
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
                        print(f"  ✓ Chunk {i+1}/{chunks}: {len(data)} candles")
                        time.sleep(0.5)
            except Exception as e:
                print(f"  ✗ Chunk {i+1}/{chunks}: Error")
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop_duplicates(subset=['time'])
        
        print(f"\n✅ Fetched {len(df)} days")
        print(f"   Range: {df['time'].min().date()} to {df['time'].max().date()}")
        
        return df
    
    def get_data(self, years=4, force_refresh=False):
        """Get data (from cache or fresh fetch)"""
        
        # Check cache first
        if not force_refresh and self.is_cache_valid():
            return self.load_from_cache()
        
        # Fetch fresh data
        df = self.fetch_multi_year_data(years=years)
        
        if df is not None:
            # Save to cache
            self.save_to_cache(df)
        
        return df
    
    def classify_all_regimes(self, df):
        """Classify every period using the unified RegimeClassifier"""
        print("\n🔍 Classifying all historical regimes...")

        df = RegimeClassifier.classify_dataframe(df, min_warmup=50)

        # Compute regime_strength for save_classified_data() compatibility
        df['regime_strength'] = 0.0
        df.loc[df['regime'] == 'TRENDING', 'regime_strength'] = df['adx']
        df.loc[df['regime'] == 'VOLATILE', 'regime_strength'] = df['volatility_pct']
        df.loc[df['regime'] == 'RANGING', 'regime_strength'] = 100 - df['adx']

        df = df[df['regime'] != 'UNKNOWN']

        regime_counts = df['regime'].value_counts()
        print(f"   RANGING: {regime_counts.get('RANGING', 0)} days ({regime_counts.get('RANGING', 0)/len(df)*100:.1f}%)")
        print(f"   TRENDING: {regime_counts.get('TRENDING', 0)} days ({regime_counts.get('TRENDING', 0)/len(df)*100:.1f}%)")
        print(f"   VOLATILE: {regime_counts.get('VOLATILE', 0)} days ({regime_counts.get('VOLATILE', 0)/len(df)*100:.1f}%)")

        return df
    
    def save_classified_data(self, df):
        """Save classified dataset for backtesting"""
        filename = 'classified_regimes.json'
        
        df_export = df[['time', 'open', 'high', 'low', 'close', 'volume', 
                        'regime', 'regime_strength', 'adx', 'volatility_pct']].copy()
        df_export['time'] = df_export['time'].astype(str)
        
        df_export.to_json(filename, orient='records', indent=2)
        
        print(f"\n💾 Saved classified data: {filename}")
        print(f"   {len(df)} days classified by regime")
        print(f"   Backtests can now use this instantly!")

def main():
    print("="*80)
    print("🔄 CACHED FULL CYCLE BACKTESTER")
    print("="*80 + "\n")
    
    backtester = CachedCycleBacktester()
    
    # Get data (from cache or fresh)
    print("📥 Getting 4 years of BTC data...")
    df = backtester.get_data(years=4, force_refresh=False)
    
    if df is None:
        print("❌ Failed to get data")
        return
    
    # Classify all regimes
    df = backtester.classify_all_regimes(df)
    
    # Save classified data for backtesting
    backtester.save_classified_data(df)
    
    # Stats
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total Days: {len(df)}")
    print(f"Date Range: {df['time'].min().date()} to {df['time'].max().date()}")
    print(f"\nRegime Distribution:")
    for regime, count in df['regime'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")
    
    print("\n✅ Setup complete!")
    print("\n💡 Next time you run this:")
    print("   - Will use cache (instant)")
    print("   - Cache auto-refreshes after 7 days")
    print("   - Use force_refresh=True to fetch now")

if __name__ == "__main__":
    main()