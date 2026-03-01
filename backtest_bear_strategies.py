"""Bear Market Strategies Backtest — Pairs, Mean Reversion, Cross-Sectional Momentum

Tests 3 strategies designed for bear/sideways markets (BTC < SMA200):
  1. Pairs Trading (cointegration-based, market-neutral)
  2. Mean Reversion (RSI/BB oversold bounces, bear-only)
  3. Cross-Sectional Momentum (long top 2 / short bottom 2, weekly rebalance)

Combined portfolio analysis: all 3 + existing Donchian longs on $30K.

Usage:
  cd C:\\ResearchAgent && venv/Scripts/python backtest_bear_strategies.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from backtest_donchian_daily import (
    fetch_all_coins, calculate_indicators, compute_stats,
    compute_per_coin_stats, DEFAULT_PARAMS, COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_shorts import compute_btc_bear_filter
from backtest_phase3 import backtest_portfolio_phase3
from backtest_walkforward import (
    compute_sharpe, compute_max_drawdown, print_section,
)
from backtest_trimode import compute_total_tax, compute_stats_from_trades

try:
    from statsmodels.tsa.stattools import coint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Pairs trading will be skipped.")
    print("  Install with: venv/Scripts/pip install statsmodels")


# ============================================================================
# CONFIGURATION
# ============================================================================

STARTING_CAPITAL = 30_000.0

ALL_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD', 'DOGE-USD',
]

# Strategy 1: Pairs Trading
PAIRS_PARAMS = {
    'pairs': [
        ('BTC-USD', 'ETH-USD'),
        ('SOL-USD', 'SUI-USD'),
        ('SOL-USD', 'LINK-USD'),
        ('LINK-USD', 'ADA-USD'),
        ('XRP-USD', 'ADA-USD'),
    ],
    'coint_window': 60,
    'zscore_entry': 2.0,
    'zscore_exit': 0.5,       # exit when |z| < 0.5 (close to mean)
    'zscore_stop': 3.5,
    'max_hold_days': 30,
    'risk_per_trade_pct': 3.0,
    'fee_pct': 0.06,
    'slippage_pct': 0.05,
    'max_positions': 2,
}

# Strategy 2: Mean Reversion
MR_PARAMS = {
    'bb_period': 20,
    'bb_sigma': 2.0,
    'rsi_entry': 25,
    'rsi_exit': 60,
    'volume_mult': 1.2,
    'max_hold_days': 5,
    'emergency_stop_pct': 8.0,
    'atr_stop_mult': 2.0,
    'risk_per_trade_pct': 2.0,
    'fee_pct': 0.45,
    'slippage_pct': 0.05,
    'max_positions': 3,
}

MR_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]

# Strategy 3: Cross-Sectional Momentum
XS_PARAMS = {
    'momentum_lookback': 14,
    'rebalance_days': 7,
    'long_count': 2,
    'short_count': 2,
    'position_size_pct': 2.5,  # % of equity per position
    'fee_pct': 0.06,
    'slippage_pct': 0.05,
}

XS_COINS = ALL_COINS

# Donchian reference params
DONCHIAN_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Donchian Long (Phase 3)',
    'atr_mult': 4.0,
    'starting_capital': STARTING_CAPITAL,
    'max_positions': 4,
}


# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================

def calculate_bollinger_bands(df, period=20, sigma=2.0):
    """Add Bollinger Band columns to dataframe."""
    df = df.copy()
    df['bb_sma'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_sma'] + sigma * bb_std
    df['bb_lower'] = df['bb_sma'] - sigma * bb_std
    return df


def calculate_mr_indicators(df, params):
    """Calculate indicators for mean reversion: base indicators + Bollinger Bands."""
    base_params = {
        'donchian_period': 20, 'exit_period': 10, 'atr_period': 14,
        'atr_mult': params['atr_stop_mult'], 'volume_mult': params['volume_mult'],
        'ema_period': 21, 'rsi_blowoff': 80, 'volume_blowoff': 3.0,
        'atr_mult_tight': 1.5, 'fee_pct': params['fee_pct'],
        'slippage_pct': params['slippage_pct'],
    }
    df = calculate_indicators(df, base_params)
    df = calculate_bollinger_bands(df, params['bb_period'], params['bb_sigma'])
    return df


def compute_log_spread(df_a, df_b, window=60):
    """Compute rolling spread and z-score between two price series.

    Returns DataFrame with: date, price_a, price_b, beta, spread, zscore
    """
    a = df_a[['time', 'close']].copy()
    a['date'] = a['time'].dt.date
    a = a.rename(columns={'close': 'price_a'})

    b = df_b[['time', 'close']].copy()
    b['date'] = b['time'].dt.date
    b = b.rename(columns={'close': 'price_b'})

    merged = pd.merge(a[['date', 'price_a']], b[['date', 'price_b']], on='date')
    merged = merged.sort_values('date').reset_index(drop=True)

    if len(merged) < window:
        return merged

    merged['log_a'] = np.log(merged['price_a'])
    merged['log_b'] = np.log(merged['price_b'])

    betas = []
    spreads = []
    for i in range(len(merged)):
        if i < window - 1:
            betas.append(np.nan)
            spreads.append(np.nan)
            continue
        win_a = merged['log_a'].iloc[i - window + 1:i + 1].values
        win_b = merged['log_b'].iloc[i - window + 1:i + 1].values
        cov_ab = np.cov(win_a, win_b)[0, 1]
        var_b = np.var(win_b, ddof=1)
        beta = np.clip(cov_ab / var_b if var_b > 0 else 1.0, 0.3, 3.0)
        betas.append(beta)
        spreads.append(win_a[-1] - beta * win_b[-1])

    merged['beta'] = betas
    merged['spread'] = spreads

    spread_s = pd.Series(spreads, dtype=float)
    merged['spread_mean'] = spread_s.rolling(window=window, min_periods=window).mean()
    merged['spread_std'] = spread_s.rolling(window=window, min_periods=window).std()
    merged['zscore'] = np.where(
        merged['spread_std'] > 0,
        (merged['spread'] - merged['spread_mean']) / merged['spread_std'],
        0.0,
    )

    return merged


def test_pair_cointegration(log_a, log_b):
    """Run Engle-Granger cointegration test. Returns p-value."""
    if not HAS_STATSMODELS or len(log_a) < 30:
        return 1.0
    try:
        _, pvalue, _ = coint(log_a, log_b)
        return pvalue
    except Exception:
        return 1.0


# ============================================================================
# STRATEGY 1: PAIRS TRADING
# ============================================================================

def backtest_pairs_trading(coin_data, params, starting_capital):
    """Backtest cointegration-based pairs trading.

    Market-neutral: runs in all market conditions.
    """
    cost = (params['fee_pct'] + params['slippage_pct']) / 100
    risk_pct = params['risk_per_trade_pct'] / 100
    max_pos = params['max_positions']

    capital = starting_capital
    positions = {}   # pair_key -> position dict
    trades = []
    equity_curve = []

    # Precompute spread DataFrames for each pair
    pair_data = {}
    for sym_a, sym_b in params['pairs']:
        if sym_a not in coin_data or sym_b not in coin_data:
            continue
        spread_df = compute_log_spread(coin_data[sym_a], coin_data[sym_b], params['coint_window'])
        spread_df = spread_df.dropna(subset=['zscore']).reset_index(drop=True)
        if len(spread_df) > 0:
            pair_key = f"{sym_a}/{sym_b}"
            pair_data[pair_key] = spread_df

    if not pair_data:
        return trades, equity_curve, capital

    # Build unified timeline from all pairs
    all_dates = set()
    pair_lookups = {}
    for pair_key, df in pair_data.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['date']] = row
        pair_lookups[pair_key] = lookup
        all_dates.update(df['date'].tolist())
    all_dates = sorted(all_dates)

    # Rolling cointegration cache
    coint_cache = {}  # (pair_key, date) -> bool

    for date in all_dates:
        # === EXITS ===
        for pair_key in list(positions.keys()):
            pos = positions[pair_key]
            row = pair_lookups[pair_key].get(date)
            if row is None:
                continue

            pos['hold_days'] += 1
            zscore = float(row['zscore'])
            exit_reason = None

            # 1. Mean reversion: |z| < exit threshold
            if abs(zscore) < params['zscore_exit']:
                exit_reason = 'Mean reversion (z~0)'

            # 2. Divergence stop: |z| > 3.5
            if not exit_reason and abs(zscore) > params['zscore_stop']:
                exit_reason = f'Stop (|z|>{params["zscore_stop"]})'

            # 3. Max hold
            if not exit_reason and pos['hold_days'] >= params['max_hold_days']:
                exit_reason = f'Max hold ({params["max_hold_days"]}d)'

            if exit_reason:
                # PnL = long leg + short leg
                price_a = float(row['price_a'])
                price_b = float(row['price_b'])

                if pos['side'] == 'long_A_short_B':
                    long_pnl = ((price_a * (1 - cost)) - pos['entry_price_a']) / pos['entry_price_a']
                    short_pnl = (pos['entry_price_b'] - (price_b * (1 + cost))) / pos['entry_price_b']
                else:  # long_B_short_A
                    long_pnl = ((price_b * (1 - cost)) - pos['entry_price_b']) / pos['entry_price_b']
                    short_pnl = (pos['entry_price_a'] - (price_a * (1 + cost))) / pos['entry_price_a']

                total_pnl_pct = (long_pnl + short_pnl) / 2 * 100  # avg of two legs
                pnl_usd = pos['size_usd'] * (total_pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd

                trades.append({
                    'symbol': pair_key, 'mode': 'PAIRS',
                    'entry_time': pos['entry_date'], 'exit_time': date,
                    'pnl_pct': total_pnl_pct, 'size_usd': pos['size_usd'],
                    'exit_reason': exit_reason, 'win': total_pnl_pct > 0,
                    'hold_days': pos['hold_days'],
                    'entry_zscore': pos['entry_zscore'], 'exit_zscore': zscore,
                })
                del positions[pair_key]

        # === ENTRIES ===
        if len(positions) < max_pos:
            for pair_key, lookup in pair_lookups.items():
                if pair_key in positions or len(positions) >= max_pos:
                    continue

                row = lookup.get(date)
                if row is None:
                    continue

                zscore = float(row['zscore'])
                if abs(zscore) < params['zscore_entry']:
                    continue

                # Rolling cointegration check (cache for 5 days)
                cache_key = pair_key
                if cache_key not in coint_cache or (date - coint_cache[cache_key][1]).days >= 5:
                    df = pair_data[pair_key]
                    idx = df[df['date'] <= date].index
                    if len(idx) < params['coint_window']:
                        continue
                    end_idx = idx[-1] + 1
                    start_idx = max(0, end_idx - params['coint_window'])
                    p_val = test_pair_cointegration(
                        df['log_a'].iloc[start_idx:end_idx].values,
                        df['log_b'].iloc[start_idx:end_idx].values,
                    )
                    coint_cache[cache_key] = (p_val < 0.05, date)

                is_coint = coint_cache.get(cache_key, (False, date))[0]
                if not is_coint:
                    continue

                # Determine direction
                price_a = float(row['price_a'])
                price_b = float(row['price_b'])

                if zscore > params['zscore_entry']:
                    side = 'long_B_short_A'  # A overvalued, short A, long B
                else:
                    side = 'long_A_short_B'  # B overvalued, short B, long A

                # Position sizing: risk-based
                total_equity = capital + sum(p['size_usd'] for p in positions.values())
                # Estimated stop at zscore_stop, roughly proportional to spread_std
                spread_std = float(row['spread_std']) if row['spread_std'] > 0 else 0.05
                est_stop_pct = max((params['zscore_stop'] - abs(zscore)) * spread_std * 0.5, 0.02)
                size = (total_equity * risk_pct) / est_stop_pct
                size = min(size, capital * 0.80)
                if size < 200:
                    continue

                # Entry prices include fees
                entry_a = price_a * (1 + cost) if side == 'long_A_short_B' else price_a * (1 - cost)
                entry_b = price_b * (1 + cost) if side == 'long_B_short_A' else price_b * (1 - cost)

                capital -= size
                positions[pair_key] = {
                    'side': side, 'entry_date': date,
                    'entry_price_a': price_a, 'entry_price_b': price_b,
                    'entry_zscore': zscore, 'beta': float(row['beta']),
                    'size_usd': size, 'hold_days': 0,
                }

        # Equity tracking
        total_eq = capital + sum(p['size_usd'] for p in positions.values())
        equity_curve.append({'date': date, 'equity': total_eq})

    # Close remaining positions
    for pair_key in list(positions.keys()):
        pos = positions[pair_key]
        df = pair_data[pair_key]
        last = df.iloc[-1]
        price_a = float(last['price_a'])
        price_b = float(last['price_b'])

        if pos['side'] == 'long_A_short_B':
            long_pnl = ((price_a * (1 - cost)) - pos['entry_price_a']) / pos['entry_price_a']
            short_pnl = (pos['entry_price_b'] - (price_b * (1 + cost))) / pos['entry_price_b']
        else:
            long_pnl = ((price_b * (1 - cost)) - pos['entry_price_b']) / pos['entry_price_b']
            short_pnl = (pos['entry_price_a'] - (price_a * (1 + cost))) / pos['entry_price_a']

        total_pnl_pct = (long_pnl + short_pnl) / 2 * 100
        pnl_usd = pos['size_usd'] * (total_pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': pair_key, 'mode': 'PAIRS',
            'entry_time': pos['entry_date'], 'exit_time': last['date'],
            'pnl_pct': total_pnl_pct, 'size_usd': pos['size_usd'],
            'exit_reason': 'End of backtest', 'win': total_pnl_pct > 0,
            'hold_days': pos['hold_days'],
            'entry_zscore': pos['entry_zscore'], 'exit_zscore': float(last['zscore']),
        })
        del positions[pair_key]

    return trades, equity_curve, capital


# ============================================================================
# STRATEGY 2: MEAN REVERSION (RSI/BB, BEAR-ONLY)
# ============================================================================

def backtest_mean_reversion(coin_data, params, bull_filter, starting_capital):
    """Backtest RSI/Bollinger Band mean reversion, only active when BTC < SMA(200)."""
    cost = (params['fee_pct'] + params['slippage_pct']) / 100
    risk_pct = params['risk_per_trade_pct'] / 100
    max_pos = params['max_positions']

    capital = starting_capital
    positions = {}
    trades = []
    equity_curve = []

    # Prepare indicators
    prepared = {}
    for symbol in MR_COINS:
        if symbol not in coin_data:
            continue
        df = calculate_mr_indicators(coin_data[symbol], params)
        df = df.dropna(subset=['rsi', 'atr', 'bb_sma', 'bb_lower', 'volume_sma'])
        df = df.reset_index(drop=True)
        if len(df) > 30:
            prepared[symbol] = df

    # Build timeline and lookups
    all_dates = set()
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    for date in all_dates:
        is_bear = not bull_filter.get(date, False)  # active when BTC < SMA200

        # === EXITS ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            row = lookups[symbol].get(date)
            if row is None:
                continue

            pos['hold_days'] += 1
            current_close = float(row['close'])
            exit_reason = None

            # 1. Mean reversion target: close >= middle BB
            if current_close >= float(row['bb_sma']):
                exit_reason = 'BB mean reversion (20-SMA)'

            # 2. RSI exit
            if not exit_reason and float(row['rsi']) > params['rsi_exit']:
                exit_reason = 'RSI exit (>60)'

            # 3. Max hold
            if not exit_reason and pos['hold_days'] >= params['max_hold_days']:
                exit_reason = f'Max hold ({params["max_hold_days"]}d)'

            # 4. Emergency stop
            if not exit_reason and current_close <= pos['entry_price'] * (1 - params['emergency_stop_pct'] / 100):
                exit_reason = f'Emergency stop (-{params["emergency_stop_pct"]}%)'

            # 5. ATR stop
            if not exit_reason and current_close <= pos['atr_stop_level']:
                exit_reason = 'ATR stop (2x ATR)'

            if exit_reason:
                exit_price = current_close * (1 - cost)
                pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol, 'mode': 'MEAN_REV',
                    'entry_time': pos['entry_time'], 'exit_time': row['time'],
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
                    'exit_reason': exit_reason, 'win': pnl_pct > 0,
                    'hold_days': pos['hold_days'],
                })
                del positions[symbol]

        # === ENTRIES (bear only) ===
        if is_bear and len(positions) < max_pos:
            for symbol in MR_COINS:
                if symbol not in prepared or symbol in positions:
                    continue
                if len(positions) >= max_pos:
                    break

                row = lookups[symbol].get(date)
                if row is None or pd.isna(row.get('rsi')) or pd.isna(row.get('bb_lower')):
                    continue

                rsi_ok = float(row['rsi']) < params['rsi_entry']
                bb_ok = float(row['close']) < float(row['bb_lower'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                vol_ok = float(row['volume']) > params['volume_mult'] * vol_sma

                if rsi_ok and bb_ok and vol_ok:
                    entry_price = float(row['close']) * (1 + cost)
                    atr_val = float(row['atr'])
                    atr_stop = entry_price - params['atr_stop_mult'] * atr_val
                    emergency_stop = entry_price * (1 - params['emergency_stop_pct'] / 100)
                    stop_level = max(atr_stop, emergency_stop)  # tighter stop
                    stop_pct = (entry_price - stop_level) / entry_price
                    if stop_pct <= 0:
                        continue

                    total_equity = capital + sum(p['size_usd'] for p in positions.values())
                    size = (total_equity * risk_pct) / stop_pct
                    size = min(size, capital * 0.80)
                    if size < 100:
                        continue

                    capital -= size
                    positions[symbol] = {
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'atr_stop_level': stop_level,
                        'size_usd': size, 'hold_days': 0,
                    }

        # Equity tracking
        total_eq = capital
        for sym, pos in positions.items():
            row = lookups[sym].get(date)
            if row is not None:
                current = float(row['close'])
                unrealized = pos['size_usd'] * ((current - pos['entry_price']) / pos['entry_price'])
                total_eq += pos['size_usd'] + unrealized
            else:
                total_eq += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_eq})

    # Close remaining
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        df = prepared[symbol]
        last = df.iloc[-1]
        exit_price = float(last['close']) * (1 - cost)
        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'mode': 'MEAN_REV',
            'entry_time': pos['entry_time'], 'exit_time': last['time'],
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
            'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
            'exit_reason': 'End of backtest', 'win': pnl_pct > 0,
            'hold_days': pos['hold_days'],
        })

    return trades, equity_curve, capital


# ============================================================================
# STRATEGY 3: CROSS-SECTIONAL MOMENTUM (LONG/SHORT)
# ============================================================================

def backtest_cross_sectional_momentum(coin_data, params, starting_capital):
    """Backtest cross-sectional momentum: long top N, short bottom N, weekly rebalance."""
    cost = (params['fee_pct'] + params['slippage_pct']) / 100
    pos_size_pct = params['position_size_pct'] / 100
    lookback = params['momentum_lookback']

    capital = starting_capital
    positions = {}  # symbol -> {side, entry_price, size_usd}
    trades = []
    equity_curve = []

    # Build price matrix: {symbol: {date: close_price}}
    price_matrix = {}
    all_dates = set()
    for symbol in XS_COINS:
        if symbol not in coin_data:
            continue
        df = coin_data[symbol].copy()
        prices = {}
        for _, row in df.iterrows():
            d = row['time'].date() if hasattr(row['time'], 'date') else row['time']
            prices[d] = float(row['close'])
        price_matrix[symbol] = prices
        all_dates.update(prices.keys())
    all_dates = sorted(all_dates)

    if not all_dates:
        return trades, equity_curve, capital

    last_rebalance = None
    dates_list = list(all_dates)

    for i, date in enumerate(dates_list):
        # Check if rebalance due
        should_rebalance = (last_rebalance is None or
                            (date - last_rebalance).days >= params['rebalance_days'])

        if should_rebalance:
            # 1. Close all existing positions
            for symbol, pos in list(positions.items()):
                price = price_matrix.get(symbol, {}).get(date)
                if price is None:
                    # Use last available price
                    for d in reversed(dates_list[:i]):
                        price = price_matrix.get(symbol, {}).get(d)
                        if price is not None:
                            break
                if price is None:
                    price = pos['entry_price']

                if pos['side'] == 'long':
                    exit_price = price * (1 - cost)
                    pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                else:
                    exit_price = price * (1 + cost)
                    pnl_pct = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100

                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol, 'mode': f'XS_{pos["side"].upper()}',
                    'entry_time': pos['entry_date'], 'exit_time': date,
                    'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
                    'exit_reason': 'Rebalance', 'win': pnl_pct > 0,
                    'side': pos['side'],
                })
            positions = {}

            # 2. Compute momentum returns for coins with enough data
            returns = {}
            for symbol, prices in price_matrix.items():
                price_today = prices.get(date)
                # Find price lookback days ago
                target_date = date - timedelta(days=lookback)
                price_past = None
                # Search for closest date within lookback window
                for d_offset in range(4):  # allow 4-day tolerance
                    check = target_date - timedelta(days=d_offset)
                    if check in prices:
                        price_past = prices[check]
                        break
                if price_today is not None and price_past is not None and price_past > 0:
                    returns[symbol] = (price_today / price_past) - 1

            if len(returns) < params['long_count'] + params['short_count']:
                last_rebalance = date
                total_eq = capital + sum(p['size_usd'] for p in positions.values())
                equity_curve.append({'date': date, 'equity': total_eq})
                continue

            # 3. Rank and select
            sorted_coins = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            long_picks = [s for s, _ in sorted_coins[:params['long_count']]]
            short_picks = [s for s, _ in sorted_coins[-params['short_count']:]]

            # 4. Open new positions
            total_equity = capital
            per_pos_size = total_equity * pos_size_pct
            per_pos_size = min(per_pos_size, capital / (params['long_count'] + params['short_count'] + 0.5))

            for symbol in long_picks:
                price = price_matrix[symbol].get(date)
                if price is None or per_pos_size < 50:
                    continue
                entry_price = price * (1 + cost)
                capital -= per_pos_size
                positions[symbol] = {
                    'side': 'long', 'entry_price': entry_price,
                    'entry_date': date, 'size_usd': per_pos_size,
                }

            for symbol in short_picks:
                price = price_matrix[symbol].get(date)
                if price is None or per_pos_size < 50:
                    continue
                entry_price = price * (1 - cost)
                capital -= per_pos_size
                positions[symbol] = {
                    'side': 'short', 'entry_price': entry_price,
                    'entry_date': date, 'size_usd': per_pos_size,
                }

            last_rebalance = date

        # Daily equity tracking
        total_eq = capital
        for symbol, pos in positions.items():
            price = price_matrix.get(symbol, {}).get(date)
            if price is not None:
                if pos['side'] == 'long':
                    unrealized = pos['size_usd'] * ((price - pos['entry_price']) / pos['entry_price'])
                else:
                    unrealized = pos['size_usd'] * ((pos['entry_price'] - price) / pos['entry_price'])
                total_eq += pos['size_usd'] + unrealized
            else:
                total_eq += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_eq})

    # Close remaining at end
    for symbol, pos in list(positions.items()):
        last_date = dates_list[-1]
        price = price_matrix.get(symbol, {}).get(last_date, pos['entry_price'])
        if pos['side'] == 'long':
            exit_price = price * (1 - cost)
            pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        else:
            exit_price = price * (1 + cost)
            pnl_pct = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'mode': f'XS_{pos["side"].upper()}',
            'entry_time': pos['entry_date'], 'exit_time': last_date,
            'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
            'exit_reason': 'End of backtest', 'win': pnl_pct > 0,
            'side': pos['side'],
        })

    return trades, equity_curve, capital


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def compute_monthly_returns(equity_curve):
    """Convert equity curve to monthly return series."""
    if not equity_curve:
        return {}
    monthly = {}
    prev_eq = equity_curve[0]['equity']
    prev_month = (equity_curve[0]['date'].year, equity_curve[0]['date'].month)

    for pt in equity_curve:
        d = pt['date']
        month_key = (d.year, d.month)
        if month_key != prev_month:
            monthly[prev_month] = (pt['equity'] / prev_eq - 1) * 100
            prev_eq = pt['equity']
            prev_month = month_key

    # Last month
    if equity_curve:
        last = equity_curve[-1]
        month_key = (last['date'].year, last['date'].month)
        if month_key not in monthly:
            monthly[month_key] = (last['equity'] / prev_eq - 1) * 100

    return monthly


def compute_year_returns(equity_curve):
    """Compute per-year returns from equity curve."""
    if not equity_curve:
        return {}
    yearly = {}
    # Find first and last equity for each year
    year_first = {}
    year_last = {}
    for pt in equity_curve:
        yr = pt['date'].year
        if yr not in year_first:
            year_first[yr] = pt['equity']
        year_last[yr] = pt['equity']

    prev_eq = equity_curve[0]['equity']
    for yr in sorted(year_first.keys()):
        end_eq = year_last[yr]
        yearly[yr] = {
            'start': prev_eq,
            'end': end_eq,
            'return_pct': (end_eq / prev_eq - 1) * 100,
            'return_usd': end_eq - prev_eq,
        }
        prev_eq = end_eq

    return yearly


def print_strategy_report(label, trades, equity_curve, starting_capital):
    """Print summary for a single strategy."""
    real_trades = [t for t in trades if abs(t['pnl_pct']) > 0.001]
    if not real_trades:
        print(f"  {label}: No trades generated")
        return

    wins = [t for t in real_trades if t['win']]
    losses = [t for t in real_trades if not t['win']]
    wr = len(wins) / len(real_trades) * 100 if real_trades else 0

    gross_wins = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in wins)
    gross_losses = abs(sum(t['size_usd'] * t['pnl_pct'] / 100 for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    total_pnl = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in real_trades)
    final_eq = equity_curve[-1]['equity'] if equity_curve else starting_capital

    # Max drawdown
    max_dd = 0
    if equity_curve:
        peak = equity_curve[0]['equity']
        for pt in equity_curve:
            peak = max(peak, pt['equity'])
            dd = (peak - pt['equity']) / peak * 100
            max_dd = max(max_dd, dd)

    # Avg hold
    holds = [t.get('hold_days', 0) for t in real_trades if t.get('hold_days')]
    avg_hold = np.mean(holds) if holds else 0

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    print(f"  {label}")
    print(f"    Trades: {len(real_trades)}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}  |  Avg Hold: {avg_hold:.1f}d")
    print(f"    Avg Win: {avg_win:+.2f}%  |  Avg Loss: {avg_loss:+.2f}%")
    print(f"    Net P&L: ${total_pnl:+,.0f}  |  Return: {total_pnl/starting_capital*100:+.1f}%")
    print(f"    Max DD: {max_dd:.1f}%  |  Final Equity: ${final_eq:,.0f}")

    # Strategy approval check
    path_a = wr >= 55 and pf >= 1.5
    path_b = pf >= 1.8 and len(real_trades) >= 10
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = float('inf')
    path_b = path_b and win_loss_ratio >= 1.5
    verdict = "PASS (Path A)" if path_a else ("PASS (Path B)" if path_b else "FAIL")
    print(f"    Approval: {verdict}")

    return {
        'trades': len(real_trades), 'wr': wr, 'pf': pf, 'max_dd': max_dd,
        'net_pnl': total_pnl, 'final': final_eq, 'avg_hold': avg_hold,
    }


def print_year_by_year(label, equity_curve):
    """Print per-year returns."""
    yearly = compute_year_returns(equity_curve)
    if not yearly:
        return
    print(f"\n    {label} — Year-by-Year:")
    print(f"    {'Year':<6} {'Start':>12} {'End':>12} {'Return':>10} {'Return%':>8}")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        marker = ' *BEAR*' if yr in [2022, 2025] else ''
        print(f"    {yr:<6} ${y['start']:>11,.0f} ${y['end']:>11,.0f} ${y['return_usd']:>+9,.0f} {y['return_pct']:>+7.1f}%{marker}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("BEAR MARKET STRATEGIES BACKTEST")
    print(f"  Starting Capital: ${STARTING_CAPITAL:,.0f} | 4-year backtest")
    print(f"  Strategies: Pairs Trading, Mean Reversion (bear-only), Cross-Sectional Momentum")
    print("=" * 100)

    # Fetch data
    print(f"\nFetching daily data for {len(ALL_COINS)} coins...")
    coin_data = fetch_all_coins(coins=ALL_COINS, years=4)
    print(f"Loaded {len(coin_data)} coins\n")

    btc_df = coin_data.get('BTC-USD')
    bull_filter = compute_btc_bull_filter(btc_df)
    bear_days = sum(1 for v in bull_filter.values() if not v)
    bull_days = sum(1 for v in bull_filter.values() if v)
    print(f"  Bull days: {bull_days} ({bull_days/(bull_days+bear_days)*100:.0f}%)")
    print(f"  Bear days: {bear_days} ({bear_days/(bull_days+bear_days)*100:.0f}%)\n")

    all_results = {}

    # ================================================================
    # STRATEGY 1: PAIRS TRADING
    # ================================================================
    print_section("STRATEGY 1: PAIRS TRADING (COINTEGRATION)")

    if HAS_STATSMODELS:
        print("  Testing pairs:", [f"{a}/{b}" for a, b in PAIRS_PARAMS['pairs']])
        pairs_trades, pairs_eq, pairs_cap = backtest_pairs_trading(
            coin_data, PAIRS_PARAMS, STARTING_CAPITAL)
        stats = print_strategy_report('Pairs Trading', pairs_trades, pairs_eq, STARTING_CAPITAL)
        all_results['Pairs'] = {'trades': pairs_trades, 'eq': pairs_eq, 'stats': stats}
        print_year_by_year('Pairs Trading', pairs_eq)

        # Per-pair breakdown
        pair_groups = defaultdict(list)
        for t in pairs_trades:
            pair_groups[t['symbol']].append(t)
        if pair_groups:
            print(f"\n    Per-Pair Breakdown:")
            print(f"    {'Pair':<20} {'Trades':>7} {'WR':>6} {'Avg P&L':>8} {'Total':>10}")
            for pair_key in sorted(pair_groups.keys()):
                pt = pair_groups[pair_key]
                real = [t for t in pt if abs(t['pnl_pct']) > 0.001]
                if not real:
                    continue
                w = sum(1 for t in real if t['win'])
                wr = w / len(real) * 100
                avg_pnl = np.mean([t['pnl_pct'] for t in real])
                total = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in real)
                print(f"    {pair_key:<20} {len(real):>7} {wr:>5.1f}% {avg_pnl:>+7.2f}% ${total:>+9,.0f}")
    else:
        print("  SKIPPED — statsmodels not installed")
        all_results['Pairs'] = None

    # ================================================================
    # STRATEGY 2: MEAN REVERSION
    # ================================================================
    print_section("STRATEGY 2: MEAN REVERSION (RSI/BB, BEAR-ONLY)")

    print("  Active only when BTC < SMA(200)")
    mr_trades, mr_eq, mr_cap = backtest_mean_reversion(
        coin_data, MR_PARAMS, bull_filter, STARTING_CAPITAL)
    stats = print_strategy_report('Mean Reversion (Bear-Only)', mr_trades, mr_eq, STARTING_CAPITAL)
    all_results['MeanRev'] = {'trades': mr_trades, 'eq': mr_eq, 'stats': stats}
    print_year_by_year('Mean Reversion', mr_eq)

    # Per-coin breakdown
    coin_groups = defaultdict(list)
    for t in mr_trades:
        coin_groups[t['symbol']].append(t)
    if coin_groups:
        print(f"\n    Per-Coin Breakdown:")
        print(f"    {'Coin':<12} {'Trades':>7} {'WR':>6} {'Avg P&L':>8} {'Total':>10}")
        for sym in sorted(coin_groups.keys()):
            ct = coin_groups[sym]
            real = [t for t in ct if abs(t['pnl_pct']) > 0.001]
            if not real:
                continue
            w = sum(1 for t in real if t['win'])
            wr = w / len(real) * 100
            avg_pnl = np.mean([t['pnl_pct'] for t in real])
            total = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in real)
            print(f"    {sym:<12} {len(real):>7} {wr:>5.1f}% {avg_pnl:>+7.2f}% ${total:>+9,.0f}")

    # ================================================================
    # STRATEGY 3: CROSS-SECTIONAL MOMENTUM
    # ================================================================
    print_section("STRATEGY 3: CROSS-SECTIONAL MOMENTUM (L/S)")

    print(f"  Long top {XS_PARAMS['long_count']}, short bottom {XS_PARAMS['short_count']}")
    print(f"  Rebalance every {XS_PARAMS['rebalance_days']} days, {XS_PARAMS['momentum_lookback']}-day lookback")
    xs_trades, xs_eq, xs_cap = backtest_cross_sectional_momentum(
        coin_data, XS_PARAMS, STARTING_CAPITAL)
    stats = print_strategy_report('Cross-Sectional Momentum', xs_trades, xs_eq, STARTING_CAPITAL)
    all_results['XS'] = {'trades': xs_trades, 'eq': xs_eq, 'stats': stats}
    print_year_by_year('Cross-Sectional Momentum', xs_eq)

    # Long vs short side breakdown
    long_trades = [t for t in xs_trades if t.get('side') == 'long' or 'LONG' in t.get('mode', '')]
    short_trades = [t for t in xs_trades if t.get('side') == 'short' or 'SHORT' in t.get('mode', '')]
    if long_trades or short_trades:
        print(f"\n    Side Breakdown:")
        for side_label, side_trades in [('Long leg', long_trades), ('Short leg', short_trades)]:
            real = [t for t in side_trades if abs(t['pnl_pct']) > 0.001]
            if not real:
                continue
            w = sum(1 for t in real if t['win'])
            wr = w / len(real) * 100
            total = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in real)
            print(f"    {side_label:<15} {len(real):>5} trades, WR {wr:.1f}%, P&L ${total:>+,.0f}")

    # ================================================================
    # DONCHIAN REFERENCE
    # ================================================================
    print_section("REFERENCE: DONCHIAN LONGS (PHASE 3)")

    donch_coins = {k: v for k, v in coin_data.items() if k in COIN_UNIVERSE}
    donch_trades, donch_eq, _, _ = backtest_portfolio_phase3(
        donch_coins, DONCHIAN_PARAMS, bull_filter, pyramiding=True)
    donch_stats = print_strategy_report('Donchian Longs', donch_trades, donch_eq, STARTING_CAPITAL)
    all_results['Donchian'] = {'trades': donch_trades, 'eq': donch_eq, 'stats': donch_stats}
    print_year_by_year('Donchian Longs', donch_eq)

    # ================================================================
    # YEAR-BY-YEAR COMPARISON
    # ================================================================
    print_section("YEAR-BY-YEAR COMPARISON (ALL STRATEGIES)")

    strategies = []
    for name, data in all_results.items():
        if data and data['eq']:
            strategies.append((name, compute_year_returns(data['eq'])))

    if strategies:
        years = sorted(set(yr for _, yearly in strategies for yr in yearly.keys()))
        header = f"    {'Year':<6}"
        for name, _ in strategies:
            header += f" {name:>12}"
        print(header)
        for yr in years:
            row = f"    {yr:<6}"
            for name, yearly in strategies:
                if yr in yearly:
                    row += f" ${yearly[yr]['return_usd']:>+10,.0f}"
                else:
                    row += f" {'—':>12}"
            marker = ' <-- BEAR' if yr in [2022, 2025] else ''
            print(f"{row}{marker}")

    # ================================================================
    # CORRELATION MATRIX
    # ================================================================
    print_section("MONTHLY RETURN CORRELATION MATRIX")

    monthly_returns = {}
    for name, data in all_results.items():
        if data and data['eq']:
            monthly_returns[name] = compute_monthly_returns(data['eq'])

    if len(monthly_returns) >= 2:
        # Build DataFrame of monthly returns
        all_months = sorted(set(m for mr in monthly_returns.values() for m in mr.keys()))
        corr_data = {}
        for name, mr in monthly_returns.items():
            corr_data[name] = [mr.get(m, 0) for m in all_months]
        corr_df = pd.DataFrame(corr_data, index=all_months)
        corr_matrix = corr_df.corr()

        # Print matrix
        names = list(monthly_returns.keys())
        header = f"    {'':>12}"
        for n in names:
            header += f" {n:>12}"
        print(header)
        for n1 in names:
            row = f"    {n1:>12}"
            for n2 in names:
                row += f" {corr_matrix.loc[n1, n2]:>11.3f}"
            print(row)
        print("\n    Low/negative correlation = diversification benefit")

    # ================================================================
    # COMBINED PORTFOLIO PROJECTION
    # ================================================================
    print_section("COMBINED PORTFOLIO PROJECTION")

    # Sum up year-by-year returns across all strategies
    donch_yearly = compute_year_returns(all_results['Donchian']['eq']) if all_results.get('Donchian') else {}

    combined_yearly = {}
    for name, data in all_results.items():
        if data and data['eq']:
            yearly = compute_year_returns(data['eq'])
            for yr, vals in yearly.items():
                if yr not in combined_yearly:
                    combined_yearly[yr] = 0
                combined_yearly[yr] += vals['return_usd']

    if donch_yearly and combined_yearly:
        years = sorted(set(list(donch_yearly.keys()) + list(combined_yearly.keys())))
        print(f"  If each strategy runs on separate $30K capital pools:")
        print(f"    {'Year':<6} {'Donchian Only':>14} {'Combined All':>14} {'Improvement':>14}")
        total_d = 0
        total_c = 0
        for yr in years:
            d = donch_yearly.get(yr, {}).get('return_usd', 0)
            c = combined_yearly.get(yr, 0)
            diff = c - d
            total_d += d
            total_c += c
            marker = ' <-- BEAR' if yr in [2022, 2025] else ''
            print(f"    {yr:<6} ${d:>+13,.0f} ${c:>+13,.0f} ${diff:>+13,.0f}{marker}")
        print(f"    {'TOTAL':<6} ${total_d:>+13,.0f} ${total_c:>+13,.0f} ${total_c-total_d:>+13,.0f}")

    # ================================================================
    # PATH TO $120K/YEAR
    # ================================================================
    print_section("PATH TO $120K/YEAR INCOME")

    # Use combined 4-year return to compute CAGR
    total_combined_pnl = sum(combined_yearly.values()) if combined_yearly else 0
    # Total capital deployed = $30K per strategy * N strategies
    active_strategies = sum(1 for d in all_results.values() if d and d['stats'])
    total_deployed = STARTING_CAPITAL  # Each strategy on same capital
    combined_return_pct = total_combined_pnl / total_deployed * 100

    if total_combined_pnl > 0 and combined_yearly:
        n_years = len(combined_yearly)
        combined_final = total_deployed + total_combined_pnl
        cagr = ((combined_final / total_deployed) ** (1 / n_years)) - 1

        print(f"  {active_strategies} strategies on ${STARTING_CAPITAL:,.0f}")
        print(f"  Combined {n_years}-year net: +${total_combined_pnl:,.0f} ({combined_return_pct:+.1f}%)")
        print(f"  Combined CAGR: {cagr * 100:.1f}%")
        print(f"  Donchian-only CAGR: {((total_deployed + total_d) / total_deployed) ** (1/n_years) * 100 - 100:.1f}%")

        print(f"\n  PROJECTION at {cagr * 100:.1f}% CAGR:")
        print(f"  {'Year':>6} {'Capital':>14} {'Gross Gain':>12} {'Tax':>10} {'Net Income':>12}")
        equity = STARTING_CAPITAL
        for yr in range(1, 11):
            gross = equity * cagr
            tax_est = gross * 0.15 if gross > 15000 else 0
            net = gross - tax_est
            marker = " <-- TARGET" if net >= 120000 else ""
            print(f"  {yr:>6} ${equity:>13,.0f} ${gross:>11,.0f} ${tax_est:>9,.0f} ${net:>11,.0f}{marker}")
            if net >= 120000:
                break
            equity += net

        # What starting capital would hit $120K in year 1?
        if cagr > 0:
            needed_gross = 120000 / 0.85  # ~15% tax
            needed_capital = needed_gross / cagr
            print(f"\n  To hit $120K/year NET in year 1:")
            print(f"    Needed capital: ${needed_capital:,.0f} at {cagr*100:.1f}% CAGR")
            print(f"    With 2x leverage: ${needed_capital/2:,.0f}")
    else:
        print("  Insufficient data for projection")

    print("\n" + "=" * 100)
    print("BACKTEST COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
