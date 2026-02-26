"""Multi-Coin Donchian Channel Breakout Backtest — 4-Year Daily Candles

Backtests the Donchian breakout strategy on up to 10 coins with:
  - Realistic Coinbase transaction costs (taker fees by volume tier + slippage)
  - Portfolio-level position sizing (2% risk per trade, max 4 concurrent)
  - 3x ATR trailing stop with blow-off detection
  - Partial profit taking at +10% and +20%
  - Per-coin and aggregate statistics

Coins (by Coinbase liquidity):
  Tier 1: BTC, ETH, SOL, XRP
  Tier 2: SUI, LINK, ADA
  Tier 3: AVAX, NEAR, HBAR
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time as time_module
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

# Coinbase pairs to backtest
COIN_UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD',
    'AVAX-USD', 'NEAR-USD', 'HBAR-USD',
]

# Strategy parameters
DEFAULT_PARAMS = {
    'label': 'Donchian Breakout (Daily)',
    'donchian_period': 20,        # 20-day high for entry
    'exit_period': 10,            # 10-day low for exit channel
    'atr_period': 14,
    'atr_mult': 3.0,              # 3x ATR trailing stop
    'volume_mult': 1.5,           # volume > 1.5x 20-day avg
    'ema_period': 21,             # EMA(21) trend filter
    'rsi_blowoff': 80,            # tighten stop if RSI > 80
    'volume_blowoff': 3.0,        # tighten stop if vol > 3x avg
    'atr_mult_tight': 1.5,        # tightened stop for blow-off
    'tp1_pct': 10.0,              # partial exit at +10%
    'tp2_pct': 20.0,              # partial exit at +20%
    'tp1_fraction': 0.25,         # sell 25% at TP1
    'tp2_fraction': 0.25,         # sell 25% at TP2
    # Transaction costs (Coinbase $10K-$50K tier: 0.25% maker / 0.40% taker)
    'fee_pct': 0.40,              # taker fee (market orders)
    'slippage_pct': 0.05,         # estimated slippage
    # Portfolio management
    'risk_per_trade_pct': 2.0,    # risk 2% of portfolio per trade
    'max_positions': 4,           # max concurrent positions
    'starting_capital': 10000.0,  # $10K starting portfolio
}

# No volume filter variant
NO_VOLUME_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Donchian (No Volume Filter)',
    'volume_mult': 0.0,           # disabled — enter on any breakout
}

# Conservative variant (wider stops, stricter filters)
CONSERVATIVE_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Donchian (Conservative)',
    'atr_mult': 4.0,              # 4x ATR trailing stop
    'volume_mult': 2.0,           # volume > 2x avg
    'tp1_pct': 15.0,              # partial at +15%
    'tp2_pct': 30.0,              # partial at +30%
}


# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

def fetch_and_cache_daily_data(symbol='BTC-USD', cache_dir='cache_daily', years=4, force_refresh=False):
    """Fetch and cache 4 years of daily candles for a single coin"""
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.replace('-', '_').lower()
    cache_file = os.path.join(cache_dir, f'{safe_symbol}_{years}yr_daily.json')

    if not force_refresh and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_days = (datetime.now() - file_time).days
        if age_days <= 7:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame(cache_data['data'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            print(f"  {symbol}: loaded {len(df):,} daily candles from cache")
            return df

    coinbase_api = "https://api.exchange.coinbase.com"
    all_data = []
    end_time = datetime.utcnow()
    total_days = years * 365
    chunk_days = 300  # 300 candles per chunk
    num_chunks = math.ceil(total_days / chunk_days)

    print(f"  {symbol}: fetching {years}yr daily data ({num_chunks} chunks)...", end='', flush=True)

    for i in range(num_chunks):
        chunk_end = end_time - timedelta(days=i * chunk_days)
        chunk_start = chunk_end - timedelta(days=chunk_days)

        earliest = end_time - timedelta(days=total_days)
        if chunk_start < earliest:
            chunk_start = earliest

        url = f"{coinbase_api}/products/{symbol}/candles"
        params = {
            'start': chunk_start.isoformat(),
            'end': chunk_end.isoformat(),
            'granularity': 86400  # daily
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
            time_module.sleep(0.35)
        except Exception as e:
            print(f" error chunk {i}: {e}", end='')
            time_module.sleep(1)

    if not all_data:
        print(f" FAILED (no data)")
        return None

    df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'])

    # Cache
    df_export = df.copy()
    df_export['time'] = df_export['time'].astype(str)
    cache_data = {
        'metadata': {
            'cached_at': datetime.now().isoformat(),
            'symbol': symbol,
            'total_candles': len(df),
            'start_date': str(df['time'].min().date()),
            'end_date': str(df['time'].max().date()),
        },
        'data': df_export.to_dict('records')
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    print(f" {len(df):,} candles ({df['time'].min().date()} to {df['time'].max().date()})")
    return df


def fetch_all_coins(coins=None, years=4):
    """Fetch daily data for all coins in the universe"""
    coins = coins or COIN_UNIVERSE
    print(f"\n{'='*80}")
    print(f"FETCHING DAILY DATA FOR {len(coins)} COINS ({years} years)")
    print(f"{'='*80}")

    coin_data = {}
    for symbol in coins:
        df = fetch_and_cache_daily_data(symbol, years=years)
        if df is not None and len(df) > 100:
            coin_data[symbol] = df
        else:
            print(f"  {symbol}: SKIPPED (insufficient data)")

    print(f"\nLoaded {len(coin_data)} coins successfully")
    return coin_data


# ============================================================================
# INDICATOR COMPUTATION
# ============================================================================

def calculate_indicators(df, params):
    """Calculate all indicators for a single coin"""
    df = df.copy()
    p = params

    # Donchian Channel
    df['donchian_high'] = df['high'].rolling(window=p['donchian_period']).max()
    df['donchian_low'] = df['low'].rolling(window=p['donchian_period']).min()
    df['exit_low'] = df['low'].rolling(window=p['exit_period']).min()

    # ATR(14) — Wilder's EWM
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/p['atr_period'], min_periods=p['atr_period'], adjust=False).mean()

    # Volume SMA(20)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()

    # EMA(21) — trend filter
    df['ema_21'] = df['close'].ewm(span=p['ema_period'], adjust=False).mean()

    # RSI(14) — blow-off detection
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


# ============================================================================
# SINGLE-COIN BACKTEST (position-size aware)
# ============================================================================

def backtest_single_coin(df, params, symbol='BTC-USD'):
    """Backtest Donchian breakout on a single coin.

    Returns list of trade dicts. Does NOT manage portfolio-level capital —
    that's done by the portfolio backtest wrapper. This tracks entry/exit
    signals and per-trade P&L percentages.
    """
    df = calculate_indicators(df, params)
    df = df.dropna(subset=['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi']).reset_index(drop=True)

    if len(df) < 30:
        return []

    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    high_watermark = 0
    partials_taken = 0
    remaining_fraction = 1.0

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # === EXIT LOGIC ===
        if position == 'LONG':
            high_watermark = max(high_watermark, float(current['high']))

            exit_reason = None
            exit_price = None

            # Blow-off detection: tighten stop
            volume_ratio = current['volume'] / current['volume_sma'] if current['volume_sma'] > 0 else 0
            is_blowoff = (volume_ratio > params['volume_blowoff']
                          and current['rsi'] > params['rsi_blowoff'])
            stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

            # 1. Trailing stop: high watermark - Nx ATR
            trailing_stop = high_watermark - (stop_mult * current['atr'])
            if current['close'] <= trailing_stop:
                exit_reason = f'Trailing stop ({stop_mult}x ATR)'
                exit_price = float(current['close'])

            # 2. Donchian exit: close below 10-day low
            if not exit_reason and pd.notna(prev['exit_low']) and current['close'] < prev['exit_low']:
                exit_reason = 'Donchian exit (10-day low)'
                exit_price = float(current['close'])

            # 3. Emergency stop: 15% below entry (crypto-sized)
            if not exit_reason and current['close'] <= entry_price * 0.85:
                exit_reason = 'Emergency stop (15% below entry)'
                exit_price = float(current['close'])

            # Partial profit taking (check before full exit)
            if not exit_reason:
                gain_pct = ((current['close'] - entry_price) / entry_price) * 100
                if partials_taken == 0 and gain_pct >= params['tp1_pct']:
                    # TP1: sell 25%
                    partial_price = float(current['close']) * (1 - cost_per_side)
                    partial_pnl = ((partial_price - entry_price) / entry_price) * 100
                    trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                        'fraction': params['tp1_fraction'] * remaining_fraction,
                        'win': True,
                    })
                    remaining_fraction *= (1 - params['tp1_fraction'])
                    partials_taken = 1
                elif partials_taken == 1 and gain_pct >= params['tp2_pct']:
                    # TP2: sell another 25%
                    partial_price = float(current['close']) * (1 - cost_per_side)
                    partial_pnl = ((partial_price - entry_price) / entry_price) * 100
                    trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                        'fraction': params['tp2_fraction'] * remaining_fraction,
                        'win': True,
                    })
                    remaining_fraction *= (1 - params['tp2_fraction'])
                    partials_taken = 2

            if exit_reason:
                exit_price_adj = exit_price * (1 - cost_per_side)
                pnl_pct = ((exit_price_adj - entry_price) / entry_price) * 100
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': current['time'],
                    'exit_price': exit_price_adj,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'fraction': remaining_fraction,
                    'win': pnl_pct > 0,
                })
                position = None
                remaining_fraction = 1.0
                partials_taken = 0
                continue

        # === ENTRY LOGIC ===
        if position is None:
            # 1. Breakout: close above previous bar's Donchian high
            if pd.isna(prev['donchian_high']):
                continue
            breakout = current['close'] > prev['donchian_high']

            # 2. Volume confirmation
            if params['volume_mult'] > 0:
                volume_ok = (current['volume_sma'] > 0 and
                             current['volume'] > params['volume_mult'] * current['volume_sma'])
            else:
                volume_ok = True  # disabled

            # 3. Trend filter: price above EMA(21)
            trend_ok = current['close'] > current['ema_21']

            if breakout and volume_ok and trend_ok:
                entry_price = float(current['close']) * (1 + cost_per_side)
                entry_time = current['time']
                high_watermark = float(current['high'])
                partials_taken = 0
                remaining_fraction = 1.0
                position = 'LONG'

    # Close any open position at end
    if position == 'LONG':
        last = df.iloc[-1]
        exit_price_adj = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price_adj - entry_price) / entry_price) * 100
        trades.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': last['time'],
            'exit_price': exit_price_adj,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest period',
            'fraction': remaining_fraction,
            'win': pnl_pct > 0,
        })

    return trades


# ============================================================================
# PORTFOLIO-LEVEL BACKTEST (multi-coin, position sizing)
# ============================================================================

def backtest_portfolio(coin_data, params):
    """Run portfolio-level backtest across all coins.

    Manages capital allocation: 2% risk per trade, max 4 concurrent positions.
    Simulates daily bar-by-bar across all coins simultaneously.
    """
    label = params['label']
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']

    print(f"\n{'='*80}")
    print(f"  PORTFOLIO BACKTEST: {label}")
    print(f"  Starting capital: ${capital:,.2f}")
    print(f"  Max positions: {max_positions}")
    print(f"  Risk per trade: {params['risk_per_trade_pct']}%")
    print(f"  Fees: {params['fee_pct']}% + {params['slippage_pct']}% slippage per side")
    print(f"{'='*80}")

    # Pre-calculate indicators for all coins
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_indicators(df, params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 30:
            prepared[symbol] = df_ind

    # Build unified daily timeline
    all_dates = set()
    for symbol, df in prepared.items():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Build lookup: symbol -> {date -> row}
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup

    # Also need previous day's row for each symbol
    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # Portfolio state
    positions = {}  # symbol -> {entry_price, entry_time, high_watermark, partials_taken, remaining_fraction, size_usd}
    trades = []
    equity_curve = []

    for date in all_dates:
        # === CHECK EXITS first ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_high = float(row['high'])
            current_atr = float(row['atr'])
            pos['high_watermark'] = max(pos['high_watermark'], current_high)

            exit_reason = None

            # Blow-off detection
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            volume_ratio = float(row['volume']) / vol_sma
            is_blowoff = (volume_ratio > params['volume_blowoff']
                          and float(row['rsi']) > params['rsi_blowoff'])
            stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

            # 1. Trailing stop
            trailing_stop = pos['high_watermark'] - (stop_mult * current_atr)
            if current_close <= trailing_stop:
                exit_reason = f'Trailing stop ({stop_mult}x ATR)'

            # 2. Donchian exit
            prev_row = prev_lookups[symbol].get(date)
            if not exit_reason and prev_row is not None and pd.notna(prev_row['exit_low']):
                if current_close < float(prev_row['exit_low']):
                    exit_reason = 'Donchian exit (10-day low)'

            # 3. Emergency stop (15%)
            if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                exit_reason = 'Emergency stop (15%)'

            # Partial profit taking
            if not exit_reason:
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp1_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 1
                    pos['remaining_fraction'] *= (1 - params['tp1_fraction'])
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
                    })

                elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp2_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 2
                    pos['remaining_fraction'] *= (1 - params['tp2_fraction'])
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
                    })

            if exit_reason:
                exit_price_adj = current_close * (1 - cost_per_side)
                pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'],
                    'exit_time': row['time'],
                    'exit_price': exit_price_adj,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'],
                    'win': pnl_pct > 0,
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === CHECK ENTRIES ===
        if len(positions) < max_positions:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row['donchian_high']):
                    continue

                current_close = float(row['close'])

                # Entry conditions
                breakout = current_close > float(prev_row['donchian_high'])
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close > float(row['ema_21'])

                if breakout and volume_ok and trend_ok:
                    # Position sizing: risk 2% of total equity
                    total_equity = capital + sum(p['size_usd'] for p in positions.values())
                    risk_amount = total_equity * risk_pct
                    entry_price = current_close * (1 + cost_per_side)
                    atr_val = float(row['atr'])
                    stop_distance = params['atr_mult'] * atr_val
                    stop_pct = stop_distance / entry_price

                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / max_positions

                    # Cap at available capital
                    position_size = min(position_size, capital * 0.95)  # keep 5% reserve
                    if position_size < 100:  # minimum $100 position
                        continue

                    capital -= position_size
                    positions[symbol] = {
                        'entry_price': entry_price,
                        'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0,
                        'remaining_fraction': 1.0,
                        'size_usd': position_size,
                    }

        # Track equity
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                current_val = pos['size_usd'] * (float(row['close']) / pos['entry_price'])
                total_equity += current_val
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close any remaining positions
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        exit_price_adj = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'],
            'exit_time': last['time'],
            'exit_price': exit_price_adj,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'],
            'win': pnl_pct > 0,
        })

    return trades, equity_curve, capital


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def compute_stats(trades, label, starting_capital=10000.0):
    """Compute summary statistics for a set of trades"""
    if not trades:
        return {'label': label, 'trades': 0}

    # Filter out partial takes for trade counting (they're part of same position)
    full_exits = [t for t in trades if 'Partial' not in t['exit_reason']]
    all_trades = trades

    wins = sum(1 for t in full_exits if t['win'])
    losses = len(full_exits) - wins
    win_rate = wins / len(full_exits) if full_exits else 0

    # Weighted P&L using size
    total_gain = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in all_trades if t.get('win', False))
    total_loss = abs(sum(t['size_usd'] * t['pnl_pct'] / 100 for t in all_trades if not t.get('win', True)))

    profit_factor = total_gain / total_loss if total_loss > 0 else float('inf')

    avg_win_pct = np.mean([t['pnl_pct'] for t in full_exits if t['win']]) if wins > 0 else 0
    avg_loss_pct = np.mean([t['pnl_pct'] for t in full_exits if not t['win']]) if losses > 0 else 0

    # Total P&L from trade sizes
    total_pnl_usd = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in all_trades)
    total_return = (total_pnl_usd / starting_capital) * 100

    # Unique coins traded
    coins = set(t['symbol'] for t in trades)

    # Average hold time
    hold_times = []
    for t in full_exits:
        if isinstance(t['entry_time'], str):
            et = pd.to_datetime(t['entry_time'])
        else:
            et = t['entry_time']
        if isinstance(t['exit_time'], str):
            xt = pd.to_datetime(t['exit_time'])
        else:
            xt = t['exit_time']
        hold_times.append((xt - et).total_seconds() / 86400)
    avg_hold_days = np.mean(hold_times) if hold_times else 0

    return {
        'label': label,
        'total_trades': len(full_exits),
        'partial_exits': len(all_trades) - len(full_exits),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'profit_factor': profit_factor,
        'total_return_pct': total_return,
        'total_pnl_usd': total_pnl_usd,
        'coins_traded': len(coins),
        'avg_hold_days': avg_hold_days,
    }


def compute_per_coin_stats(trades):
    """Compute stats per coin"""
    by_coin = {}
    for t in trades:
        sym = t['symbol']
        if sym not in by_coin:
            by_coin[sym] = []
        by_coin[sym].append(t)

    results = {}
    for sym, coin_trades in sorted(by_coin.items()):
        full = [t for t in coin_trades if 'Partial' not in t['exit_reason']]
        if not full:
            continue
        wins = sum(1 for t in full if t['win'])
        total = len(full)
        pnl = sum(t['pnl_pct'] for t in full)
        win_pnl = sum(t['pnl_pct'] for t in full if t['win'])
        loss_pnl = abs(sum(t['pnl_pct'] for t in full if not t['win']))
        pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
        results[sym] = {
            'trades': total,
            'wins': wins,
            'win_rate': wins / total,
            'sum_pnl_pct': pnl,
            'profit_factor': pf,
        }
    return results


def print_results(stats, trades, equity_curve, params):
    """Print comprehensive results"""
    s = stats

    print(f"\n{'='*100}")
    print(f"RESULTS: {s['label']}")
    print(f"{'='*100}")

    print(f"\n  {'Metric':<30s} {'Value':>15s}")
    print(f"  {'-'*50}")
    print(f"  {'Total Trades':<30s} {s['total_trades']:>15d}")
    print(f"  {'Partial Exits':<30s} {s.get('partial_exits', 0):>15d}")
    print(f"  {'Wins':<30s} {s['wins']:>15d}")
    print(f"  {'Losses':<30s} {s['losses']:>15d}")
    print(f"  {'Win Rate':<30s} {s['win_rate']*100:>14.1f}%")
    print(f"  {'Avg Win':<30s} {s['avg_win_pct']:>+14.2f}%")
    print(f"  {'Avg Loss':<30s} {s['avg_loss_pct']:>+14.2f}%")
    print(f"  {'Profit Factor':<30s} {s['profit_factor']:>15.2f}")
    print(f"  {'Total Return':<30s} {s['total_return_pct']:>+14.2f}%")
    print(f"  {'Total P&L ($)':<30s} ${s['total_pnl_usd']:>+13,.2f}")
    print(f"  {'Coins Traded':<30s} {s['coins_traded']:>15d}")
    print(f"  {'Avg Hold (days)':<30s} {s['avg_hold_days']:>15.1f}")

    # Max drawdown from equity curve
    if equity_curve:
        peak = equity_curve[0]['equity']
        max_dd = 0
        for point in equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            dd = (peak - point['equity']) / peak * 100
            if dd > max_dd:
                max_dd = dd
        print(f"  {'Max Drawdown':<30s} {max_dd:>14.2f}%")
        print(f"  {'Final Equity':<30s} ${equity_curve[-1]['equity']:>13,.2f}")

    # Per-coin breakdown
    coin_stats = compute_per_coin_stats(trades)
    if coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN BREAKDOWN:")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}")
        print(f"  {'-'*55}")
        for sym, cs in sorted(coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t.get('size_usd', 0) * t['pnl_pct'] / 100

    print(f"\n  {'='*80}")
    print(f"  EXIT REASONS:")
    print(f"  {'='*80}")
    for reason, rs in sorted(reasons.items(), key=lambda x: -x[1]['count']):
        print(f"  {reason}: {rs['count']} trades, ${rs['pnl']:+,.2f}")

    # Sample trades (first 20)
    full_exits = [t for t in trades if 'Partial' not in t['exit_reason']]
    show = min(20, len(full_exits))
    print(f"\n  {'='*80}")
    print(f"  TRADE LOG (showing {show} of {len(full_exits)} full exits):")
    print(f"  {'='*80}")
    for idx, t in enumerate(full_exits[:show], 1):
        marker = "WIN " if t['win'] else "LOSS"
        entry_date = str(t['entry_time'])[:10]
        exit_date = str(t['exit_time'])[:10]
        print(f"  {idx:3d}. [{marker}] {t['symbol']:<10s} {entry_date} -> {exit_date} | "
              f"${t['entry_price']:>10,.2f} -> ${t['exit_price']:>10,.2f} | "
              f"P&L: {t['pnl_pct']:>+7.2f}% | ${t.get('size_usd', 0):>8,.0f} | "
              f"{t['exit_reason']}")


def save_results(stats, trades, equity_curve, params):
    """Save results to JSON"""
    def clean(trades):
        cleaned = []
        for t in trades:
            tc = {**t}
            tc['entry_time'] = str(tc['entry_time'])
            tc['exit_time'] = str(tc['exit_time'])
            cleaned.append(tc)
        return cleaned

    results = {
        'params': {k: v for k, v in params.items() if not callable(v)},
        'stats': stats,
        'trades': clean(trades),
        'equity_curve': [{'date': str(e['date']), 'equity': e['equity']} for e in equity_curve],
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_donchian_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")
    return filename


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("DONCHIAN CHANNEL BREAKOUT — 4-YEAR MULTI-COIN DAILY BACKTEST")
    print(f"Starting Capital: ${DEFAULT_PARAMS['starting_capital']:,.0f}")
    print(f"Coins: {', '.join(COIN_UNIVERSE)}")
    print(f"Fees: {DEFAULT_PARAMS['fee_pct']}% taker + {DEFAULT_PARAMS['slippage_pct']}% slippage per side")
    print("=" * 100)

    # Fetch data for all coins
    coin_data = fetch_all_coins()

    # Run default params
    print("\n\n" + "#" * 100)
    print("# VARIANT 1: DEFAULT PARAMS")
    print("#" * 100)
    trades1, eq1, cap1 = backtest_portfolio(coin_data, DEFAULT_PARAMS)
    stats1 = compute_stats(trades1, DEFAULT_PARAMS['label'])
    print_results(stats1, trades1, eq1, DEFAULT_PARAMS)
    save_results(stats1, trades1, eq1, DEFAULT_PARAMS)

    # Run no-volume variant
    print("\n\n" + "#" * 100)
    print("# VARIANT 2: NO VOLUME FILTER")
    print("#" * 100)
    trades2, eq2, cap2 = backtest_portfolio(coin_data, NO_VOLUME_PARAMS)
    stats2 = compute_stats(trades2, NO_VOLUME_PARAMS['label'])
    print_results(stats2, trades2, eq2, NO_VOLUME_PARAMS)
    save_results(stats2, trades2, eq2, NO_VOLUME_PARAMS)

    # Run conservative variant
    print("\n\n" + "#" * 100)
    print("# VARIANT 3: CONSERVATIVE")
    print("#" * 100)
    trades3, eq3, cap3 = backtest_portfolio(coin_data, CONSERVATIVE_PARAMS)
    stats3 = compute_stats(trades3, CONSERVATIVE_PARAMS['label'])
    print_results(stats3, trades3, eq3, CONSERVATIVE_PARAMS)
    save_results(stats3, trades3, eq3, CONSERVATIVE_PARAMS)

    # Side-by-side summary
    print("\n\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"\n  {'Metric':<25s} {'Default':>18s} {'No Vol Filter':>18s} {'Conservative':>18s}")
    print(f"  {'-'*80}")

    for label, key, fmt in [
        ('Trades', 'total_trades', 'd'),
        ('Win Rate', 'win_rate', '.1%'),
        ('Avg Win', 'avg_win_pct', '+.2f'),
        ('Avg Loss', 'avg_loss_pct', '+.2f'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('Total Return', 'total_return_pct', '+.1f'),
    ]:
        vals = []
        for s in [stats1, stats2, stats3]:
            v = s.get(key, 0)
            if fmt == '.1%':
                vals.append(f"{v*100:.1f}%")
            elif fmt.startswith('+'):
                vals.append(f"{v:{fmt}}%")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {label:<25s} {vals[0]:>18s} {vals[1]:>18s} {vals[2]:>18s}")


if __name__ == "__main__":
    main()
