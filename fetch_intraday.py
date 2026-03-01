"""
Intraday Data Fetcher for ResearchAgent
========================================
Fetches and caches 1h and 15m candles from Coinbase for backtesting
intraday strategies. Reuses the same API pattern as daily fetcher.

Usage:
    from fetch_intraday import fetch_all_intraday
    data_1h = fetch_all_intraday(timeframe='1h', months=6)
    data_15m = fetch_all_intraday(timeframe='15m', months=3)
"""

import os
import json
import math
import time as time_module
import requests
import pandas as pd
from datetime import datetime, timedelta

# Top liquid coins for intraday (high volume = tighter spreads, better fills)
INTRADAY_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
    'NEAR-USD', 'AVAX-USD',
]

# Coinbase granularity values (seconds)
GRANULARITY_MAP = {
    '1h':  3600,
    '15m': 900,
    '5m':  300,
    '1m':  60,
}

# Candles per chunk (Coinbase max is 300)
CHUNK_SIZE = 300


def fetch_and_cache_intraday(symbol='BTC-USD', timeframe='1h', months=6,
                              cache_dir=None, force_refresh=False):
    """Fetch and cache intraday candles for a single coin.

    Args:
        symbol: Coinbase product ID (e.g., 'BTC-USD')
        timeframe: '1h', '15m', '5m', or '1m'
        months: How many months of history to fetch
        cache_dir: Override cache directory
        force_refresh: Ignore cache and re-fetch

    Returns:
        pandas DataFrame with columns [time, low, high, open, close, volume]
    """
    if timeframe not in GRANULARITY_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(GRANULARITY_MAP.keys())}")

    granularity = GRANULARITY_MAP[timeframe]
    cache_dir = cache_dir or f'cache_{timeframe}'
    os.makedirs(cache_dir, exist_ok=True)

    safe_symbol = symbol.replace('-', '_').lower()
    cache_file = os.path.join(cache_dir, f'{safe_symbol}_{months}mo_{timeframe}.json')

    # Check cache (1 day expiry for intraday — we want fresh data)
    if not force_refresh and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        if age_hours <= 24:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame(cache_data['data'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            print(f"  {symbol}: loaded {len(df):,} {timeframe} candles from cache")
            return df

    # Calculate chunks needed
    coinbase_api = "https://api.exchange.coinbase.com"
    end_time = datetime.utcnow()
    total_seconds = months * 30 * 24 * 3600  # approx
    candles_per_chunk = CHUNK_SIZE
    seconds_per_chunk = candles_per_chunk * granularity
    num_chunks = math.ceil(total_seconds / seconds_per_chunk)

    expected_candles = total_seconds // granularity
    print(f"  {symbol}: fetching {months}mo {timeframe} data (~{expected_candles:,} candles, {num_chunks} chunks)...",
          end='', flush=True)

    all_data = []
    errors = 0

    for i in range(num_chunks):
        chunk_end = end_time - timedelta(seconds=i * seconds_per_chunk)
        chunk_start = chunk_end - timedelta(seconds=seconds_per_chunk)

        # Don't go past our total lookback
        earliest = end_time - timedelta(seconds=total_seconds)
        if chunk_start < earliest:
            chunk_start = earliest

        url = f"{coinbase_api}/products/{symbol}/candles"
        params = {
            'start': chunk_start.isoformat(),
            'end': chunk_end.isoformat(),
            'granularity': granularity,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
            elif response.status_code == 429:
                # Rate limited — back off
                print(f" [rate-limited]", end='')
                time_module.sleep(2)
                # Retry this chunk
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        all_data.extend(data)
            else:
                errors += 1

            # Rate limit: be polite (more aggressive for more chunks)
            time_module.sleep(0.25)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f" [err:{e}]", end='')
            time_module.sleep(1)

    if not all_data:
        print(f" FAILED (no data)")
        return None

    # Build DataFrame
    df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'])

    # Convert numeric columns
    for col in ['low', 'high', 'open', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cache it
    df_export = df.copy()
    df_export['time'] = df_export['time'].astype(str)
    cache_data = {
        'metadata': {
            'cached_at': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'start_date': str(df['time'].min()),
            'end_date': str(df['time'].max()),
        },
        'data': df_export.to_dict('records')
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    if errors > 0:
        print(f" {len(df):,} candles ({errors} errors) ({df['time'].min()} to {df['time'].max()})")
    else:
        print(f" {len(df):,} candles ({df['time'].min()} to {df['time'].max()})")

    return df


def fetch_all_intraday(coins=None, timeframe='1h', months=6, force_refresh=False):
    """Fetch intraday data for all coins.

    Args:
        coins: List of Coinbase product IDs (defaults to INTRADAY_COINS)
        timeframe: '1h', '15m', '5m'
        months: How many months of history
        force_refresh: Ignore cache

    Returns:
        dict: {symbol: DataFrame}
    """
    coins = coins or INTRADAY_COINS
    print(f"\n{'='*80}")
    print(f"FETCHING {timeframe.upper()} DATA FOR {len(coins)} COINS ({months} months)")
    print(f"{'='*80}")

    coin_data = {}
    for symbol in coins:
        df = fetch_and_cache_intraday(symbol, timeframe=timeframe, months=months,
                                       force_refresh=force_refresh)
        if df is not None and len(df) > 100:
            coin_data[symbol] = df
        else:
            print(f"  {symbol}: SKIPPED (insufficient data)")

    total_candles = sum(len(df) for df in coin_data.values())
    print(f"\nLoaded {len(coin_data)} coins, {total_candles:,} total {timeframe} candles")
    return coin_data


def summarize_data(coin_data, timeframe='1h'):
    """Print a summary of the fetched data — date range, gaps, volatility stats."""
    print(f"\n{'='*80}")
    print(f"DATA SUMMARY ({timeframe})")
    print(f"{'='*80}")

    for symbol, df in sorted(coin_data.items()):
        # Date range
        start = df['time'].min()
        end = df['time'].max()
        days = (end - start).days

        # Compute hourly returns
        df_calc = df.copy()
        df_calc['return'] = df_calc['close'].pct_change()

        # Volatility stats
        avg_abs_return = df_calc['return'].abs().mean() * 100
        max_up = df_calc['return'].max() * 100
        max_down = df_calc['return'].min() * 100

        # Average daily range (high-low as % of close)
        df_calc['range_pct'] = (df_calc['high'] - df_calc['low']) / df_calc['close'] * 100
        avg_range = df_calc['range_pct'].mean()

        # Volume
        avg_vol = df_calc['volume'].mean()

        print(f"\n  {symbol}:")
        print(f"    Range: {start.date()} to {end.date()} ({days} days, {len(df):,} candles)")
        print(f"    Avg |return|: {avg_abs_return:.3f}%  |  Max up: +{max_up:.2f}%  |  Max down: {max_down:.2f}%")
        print(f"    Avg candle range: {avg_range:.3f}%  |  Avg volume: {avg_vol:,.1f}")


# ── Main: fetch data when run directly ──────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Default: fetch 1h candles for 6 months
    timeframe = sys.argv[1] if len(sys.argv) > 1 else '1h'
    months = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"Fetching {timeframe} candles for {months} months...")
    print(f"Coins: {', '.join(INTRADAY_COINS)}")

    coin_data = fetch_all_intraday(timeframe=timeframe, months=months)

    if coin_data:
        summarize_data(coin_data, timeframe=timeframe)

        print(f"\n{'='*80}")
        print(f"DONE — cached in cache_{timeframe}/")
        print(f"{'='*80}")
