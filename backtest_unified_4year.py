"""Unified 4-Year Hourly Backtest — Strict vs Loosened Comparison ($1,000 start)

Runs all 3 strategies across all regimes on 4 years of hourly BTC data.
Two modes:
  STRICT  = exact production parameters (what the live bot uses)
  LOOSENED = relaxed entry conditions to generate more trades on hourly data

Comparison printed at the end.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time as time_module
import requests
from market_regime import RegimeClassifier


# ============================================================================
# PARAMETER SETS
# ============================================================================

STRICT_PARAMS = {
    'label': 'STRICT (Production)',
    # Williams %R (RANGING)
    'wr_threshold': -80,        # crosses above -80
    'wr_sma_filter': True,      # price must be below SMA(21)
    'wr_min_hold': 6,           # minimum 6 bars (hours) before exit check
    # BB Reversion (VOLATILE)
    'bb_sigma': 2.0,            # 2 standard deviations
    'bb_rsi_entry': 35,         # RSI < 35
    # ADX Momentum (TRENDING) — loosened to match production (Feb 25)
    'adx_di_cross_required': False,  # DI crossover removed
    'adx_rsi_low': 30,
    'adx_rsi_high': 80,
    # Stop loss ATR multipliers
    'atr_mult_trend': 2.5,     # ADX trailing stop: high_watermark - N * ATR
    'atr_mult_reversion': 2.0, # BB fixed stop: entry - N * ATR
    # Transaction costs (per side)
    'slippage_pct': 0.05,      # 5 bps slippage per side
    'fee_pct': 0.25,           # 25 bps Coinbase taker fee per side
}

LOOSENED_PARAMS = {
    'label': 'LOOSENED (Relaxed)',
    # Williams %R (RANGING)
    'wr_threshold': -75,        # crosses above -75 (from -80)
    'wr_sma_filter': False,     # SMA filter removed
    'wr_min_hold': 6,           # minimum 6 bars (hours) before exit check
    # BB Reversion (VOLATILE)
    'bb_sigma': 1.5,            # 1.5 sigma (from 2.0)
    'bb_rsi_entry': 40,         # RSI < 40 (from 35)
    # ADX Momentum (TRENDING)
    'adx_di_cross_required': False,   # DI crossover removed
    'adx_rsi_low': 30,
    'adx_rsi_high': 80,
    # Stop loss ATR multipliers
    'atr_mult_trend': 2.5,     # ADX trailing stop: high_watermark - N * ATR
    'atr_mult_reversion': 2.0, # BB fixed stop: entry - N * ATR
    # Transaction costs (per side)
    'slippage_pct': 0.05,      # 5 bps slippage per side
    'fee_pct': 0.25,           # 25 bps Coinbase taker fee per side
}


# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

def fetch_and_cache_hourly_data(cache_file='btc_4year_hourly_cache.json', years=4, force_refresh=False):
    """Fetch and cache 4 years of hourly BTC data from Coinbase"""

    if not force_refresh and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_days = (datetime.now() - file_time).days
        if age_days <= 7:
            print(f"Hourly cache found ({age_days} days old), loading...")
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame(cache_data['data'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            print(f"Loaded {len(df):,} hourly candles")
            print(f"Range: {df['time'].min()} to {df['time'].max()}")
            return df

    coinbase_api = "https://api.exchange.coinbase.com"
    all_data = []
    end_time = datetime.utcnow()
    total_hours = years * 365 * 24
    chunk_hours = 300
    num_chunks = math.ceil(total_hours / chunk_hours)

    print(f"Fetching {years} years of hourly BTC data ({num_chunks} chunks, ~{num_chunks // 2}s)...")

    for i in range(num_chunks):
        chunk_end = end_time - timedelta(hours=i * chunk_hours)
        chunk_start = chunk_end - timedelta(hours=chunk_hours)

        earliest = end_time - timedelta(hours=total_hours)
        if chunk_start < earliest:
            chunk_start = earliest

        url = f"{coinbase_api}/products/BTC-USD/candles"
        params = {
            'start': chunk_start.isoformat(),
            'end': chunk_end.isoformat(),
            'granularity': 3600
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
            if (i + 1) % 20 == 0:
                print(f"  Chunk {i+1}/{num_chunks} done...")
            time_module.sleep(0.5)
        except Exception as e:
            print(f"  Chunk {i+1} error: {e}")
            time_module.sleep(1)

    if not all_data:
        print("ERROR: No data fetched")
        return None

    df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'])

    print(f"Fetched {len(df):,} hourly candles")
    print(f"Range: {df['time'].min()} to {df['time'].max()}")

    df_export = df.copy()
    df_export['time'] = df_export['time'].astype(str)
    cache_data = {
        'metadata': {
            'cached_at': datetime.now().isoformat(),
            'start_date': str(df['time'].min().date()),
            'end_date': str(df['time'].max().date()),
            'total_candles': len(df),
            'granularity': '1H',
            'symbol': 'BTC-USD'
        },
        'data': df_export.to_dict('records')
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f"Cached to {cache_file}")

    return df


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================

def classify_regimes(df):
    """Classify each bar's market regime using unified RegimeClassifier"""
    df = RegimeClassifier.classify_dataframe(df, min_warmup=50)
    df = df[df['regime'] != 'UNKNOWN']
    regime_counts = df['regime'].value_counts()
    total = len(df)
    print(f"\nRegime distribution ({total:,} bars):")
    for regime in ['TRENDING', 'VOLATILE', 'RANGING']:
        count = regime_counts.get(regime, 0)
        print(f"  {regime}: {count:,} bars ({count/total*100:.1f}%)")
    return df, regime_counts


# ============================================================================
# INDICATOR COMPUTATION
# ============================================================================

def calculate_all_indicators(df, params):
    """Calculate all indicators needed by all 3 strategies"""
    df = df.copy()
    period = 14

    # ===== Williams %R (RANGING) =====
    high_roll = df['high'].rolling(window=period).max()
    low_roll = df['low'].rolling(window=period).min()
    df['williams_r'] = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
    df['sma_21'] = df['close'].rolling(window=21).mean()

    # ===== Bollinger Bands (VOLATILE) — sigma from params =====
    bb_period = 20
    bb_sigma = params['bb_sigma']
    df['sma_20'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_lower'] = df['sma_20'] - bb_sigma * df['bb_std']
    df['bb_upper'] = df['sma_20'] + bb_sigma * df['bb_std']

    # ===== RSI(14) =====
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ===== ATR(14) =====
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # ===== ADX, +DI, -DI (TRENDING) =====
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
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    df['plus_di'] = 100 * smooth_plus_dm / df['atr']
    df['minus_di'] = 100 * smooth_minus_dm / df['atr']
    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # ===== SMA(50) =====
    df['sma_50'] = df['close'].rolling(window=50).mean()

    return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_unified(df, params):
    """Run unified backtest across all regimes with all 3 strategies"""
    label = params['label']
    print(f"\n{'='*80}")
    print(f"  Running: {label}")
    print(f"{'='*80}")

    df = calculate_all_indicators(df, params)
    df = df.dropna().reset_index(drop=True)

    portfolio = 1000.0
    position = None
    entry_price = 0
    entry_time = None
    entry_bar = 0
    active_strategy = None
    trades = []
    high_watermark = 0
    non_matching_count = 0
    wr_min_hold = params.get('wr_min_hold', 0)
    atr_mult_trend = params.get('atr_mult_trend', 1.5)
    atr_mult_reversion = params.get('atr_mult_reversion', 1.5)
    cost_per_side = (params.get('slippage_pct', 0) + params.get('fee_pct', 0)) / 100

    STRATEGY_REGIME = {
        'Williams %R Mean Reversion': 'RANGING',
        'ADX Momentum Thrust': 'TRENDING',
        'Bollinger Band Mean Reversion': 'VOLATILE'
    }

    for i in range(5, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        regime = current['regime']

        # === POSITION EXIT LOGIC ===
        if position == 'LONG':
            exit_reason = None
            bars_held = i - entry_bar

            if active_strategy == 'Williams %R Mean Reversion':
                # Suppress WR exit signals during minimum hold period
                if bars_held >= wr_min_hold:
                    if current['williams_r'] < -20:
                        exit_reason = 'Williams %R < -20 (overbought zone)'
                    elif current['close'] >= current['sma_21'] * 1.015:
                        exit_reason = 'Price reached SMA(21) + 1.5% target'

            elif active_strategy == 'ADX Momentum Thrust':
                high_watermark = max(high_watermark, float(current['close']))
                if current['minus_di'] > current['plus_di'] and prev['minus_di'] <= prev['plus_di']:
                    exit_reason = '-DI crossed above +DI'
                elif current['adx'] < 20:
                    exit_reason = 'ADX dropped below 20'
                elif current['rsi'] > 75:
                    exit_reason = 'RSI > 75 (overbought)'
                elif current['close'] <= high_watermark - (atr_mult_trend * current['atr']):
                    exit_reason = f'Trailing stop ({atr_mult_trend}x ATR from high)'

            elif active_strategy == 'Bollinger Band Mean Reversion':
                if current['close'] >= current['sma_20']:
                    exit_reason = 'Price reached SMA(20) middle band'
                elif current['rsi'] > 70:
                    exit_reason = 'RSI > 70 (overbought)'
                elif current['close'] <= entry_price - (atr_mult_reversion * current['atr']):
                    exit_reason = f'Stop loss ({atr_mult_reversion}x ATR below entry)'

            # Emergency stop (5% below entry — all strategies)
            if not exit_reason and current['close'] <= entry_price * 0.95:
                exit_reason = 'Emergency stop (5% below entry)'

            # Force-sell on regime mismatch (3-bar buffer)
            if regime != STRATEGY_REGIME.get(active_strategy):
                non_matching_count += 1
                if non_matching_count >= 3 and not exit_reason:
                    exit_reason = f'Force-sell: regime changed to {regime}'
            else:
                non_matching_count = 0

            if exit_reason:
                exit_price = float(current['close']) * (1 - cost_per_side)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                portfolio *= (1 + pnl_pct / 100)
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': float(entry_price),
                    'exit_time': current['time'],
                    'exit_price': exit_price,
                    'pnl_pct': float(pnl_pct),
                    'exit_reason': exit_reason,
                    'strategy': active_strategy,
                    'portfolio': float(portfolio),
                    'win': bool(pnl_pct > 0)
                })
                position = None
                active_strategy = None
                non_matching_count = 0
                continue

        # === POSITION ENTRY LOGIC ===
        if position is None:

            # RANGING -> Williams %R Mean Reversion
            if regime == 'RANGING':
                wr_cross = (prev['williams_r'] <= params['wr_threshold'] and
                            current['williams_r'] > params['wr_threshold'])
                sma_ok = (not params['wr_sma_filter']) or (current['close'] < current['sma_21'])

                if wr_cross and sma_ok:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'Williams %R Mean Reversion'

            # TRENDING -> ADX Momentum Thrust
            elif regime == 'TRENDING':
                adx_rising = df.iloc[i]['adx'] > df.iloc[i - 5]['adx']
                adx_above_20 = current['adx'] > 20
                plus_di_above = current['plus_di'] > current['minus_di']
                rsi_ok = params['adx_rsi_low'] <= current['rsi'] <= params['adx_rsi_high']
                price_above_sma = current['close'] > current['sma_50']

                # DI crossover check (only if required by params)
                di_cross_ok = True
                if params['adx_di_cross_required']:
                    di_cross_ok = False
                    for offset in range(0, 15):
                        idx = i - offset
                        if idx >= 1:
                            bar = df.iloc[idx]
                            bar_prev = df.iloc[idx - 1]
                            if (bar['plus_di'] > bar['minus_di'] and
                                    bar_prev['plus_di'] <= bar_prev['minus_di']):
                                di_cross_ok = True
                                break

                if adx_above_20 and adx_rising and plus_di_above and di_cross_ok and rsi_ok and price_above_sma:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'ADX Momentum Thrust'
                    high_watermark = float(current['close'])

            # VOLATILE -> Bollinger Band Mean Reversion
            elif regime == 'VOLATILE':
                if current['close'] < current['bb_lower'] and current['rsi'] < params['bb_rsi_entry']:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'Bollinger Band Mean Reversion'

    # Close any open position at end
    if position == 'LONG':
        last = df.iloc[-1]
        exit_price = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        portfolio *= (1 + pnl_pct / 100)
        trades.append({
            'entry_time': entry_time,
            'entry_price': float(entry_price),
            'exit_time': last['time'],
            'exit_price': exit_price,
            'pnl_pct': float(pnl_pct),
            'exit_reason': 'End of backtest period',
            'strategy': active_strategy,
            'portfolio': float(portfolio),
            'win': bool(pnl_pct > 0)
        })

    return trades, portfolio


# ============================================================================
# RESULTS & COMPARISON
# ============================================================================

def compute_stats(trades, label):
    """Compute summary stats for a set of trades"""
    if not trades:
        return {'label': label, 'trades': 0}

    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    win_rate = wins / len(trades)
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0
    total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
    total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
    final_portfolio = trades[-1]['portfolio']
    total_return = ((final_portfolio - 1000) / 1000) * 100

    # Max drawdown
    peak = 1000.0
    max_dd = 0
    for t in trades:
        if t['portfolio'] > peak:
            peak = t['portfolio']
        dd = (peak - t['portfolio']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'label': label,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'final_portfolio': final_portfolio,
        'max_drawdown': max_dd,
    }


def compute_strategy_stats(trades, strategy_name):
    """Compute stats for a single strategy within a run"""
    strat_trades = [t for t in trades if t['strategy'] == strategy_name]
    if not strat_trades:
        return None
    wins = sum(1 for t in strat_trades if t['win'])
    total = len(strat_trades)
    pnl = sum(t['pnl_pct'] for t in strat_trades)
    win_pnl = sum(t['pnl_pct'] for t in strat_trades if t['win'])
    loss_pnl = abs(sum(t['pnl_pct'] for t in strat_trades if not t['win']))
    pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
    return {
        'trades': total,
        'wins': wins,
        'win_rate': wins / total if total > 0 else 0,
        'pnl': pnl,
        'profit_factor': pf,
    }


def print_comparison(strict_stats, loosened_stats, strict_trades, loosened_trades):
    """Print side-by-side comparison"""
    s = strict_stats
    l = loosened_stats

    print("\n" + "=" * 100)
    print("UNIFIED 4-YEAR HOURLY BACKTEST — STRICT vs LOOSENED COMPARISON")
    print("=" * 100)

    # Overall comparison
    print(f"\n{'METRIC':<30s} {'STRICT (Production)':>22s} {'LOOSENED (Relaxed)':>22s} {'DELTA':>12s}")
    print("-" * 90)

    def row(label, sv, lv, fmt='.2f', pct=False):
        suffix = '%' if pct else ''
        s_str = f"{sv:{fmt}}{suffix}" if sv is not None else "N/A"
        l_str = f"{lv:{fmt}}{suffix}" if lv is not None else "N/A"
        if sv is not None and lv is not None:
            delta = lv - sv
            d_str = f"{delta:+{fmt}}{suffix}"
        else:
            d_str = ""
        print(f"  {label:<28s} {s_str:>22s} {l_str:>22s} {d_str:>12s}")

    row('Total Trades', s.get('trades', 0), l.get('trades', 0), 'd')
    row('Wins', s.get('wins', 0), l.get('wins', 0), 'd')
    row('Losses', s.get('losses', 0), l.get('losses', 0), 'd')
    row('Win Rate', s.get('win_rate', 0) * 100, l.get('win_rate', 0) * 100, '.1f', pct=True)
    row('Avg Win', s.get('avg_win', 0), l.get('avg_win', 0), '.2f', pct=True)
    row('Avg Loss', s.get('avg_loss', 0), l.get('avg_loss', 0), '.2f', pct=True)
    row('Profit Factor', s.get('profit_factor', 0), l.get('profit_factor', 0), '.2f')
    row('Total Return', s.get('total_return', 0), l.get('total_return', 0), '.2f', pct=True)
    row('Final Portfolio ($)', s.get('final_portfolio', 1000), l.get('final_portfolio', 1000), ',.2f')
    row('Max Drawdown', s.get('max_drawdown', 0), l.get('max_drawdown', 0), '.2f', pct=True)

    # Per-strategy breakdown
    strategies = [
        ('Williams %R Mean Reversion', 'RANGING'),
        ('ADX Momentum Thrust', 'TRENDING'),
        ('Bollinger Band Mean Reversion', 'VOLATILE'),
    ]

    print(f"\n{'=' * 100}")
    print("BY STRATEGY:")
    print(f"{'=' * 100}")
    print(f"\n  {'Strategy':<35s} {'Mode':<12s} {'Trades':>7s} {'Wins':>5s} "
          f"{'WR':>7s} {'PF':>7s} {'P&L':>9s}")
    print("  " + "-" * 90)

    for strat_name, regime in strategies:
        ss = compute_strategy_stats(strict_trades, strat_name)
        ls = compute_strategy_stats(loosened_trades, strat_name)
        for mode, st in [('STRICT', ss), ('LOOSENED', ls)]:
            if st:
                print(f"  {strat_name:<35s} {mode:<12s} {st['trades']:>7d} {st['wins']:>5d} "
                      f"{st['win_rate']*100:>6.1f}% {st['profit_factor']:>7.2f} {st['pnl']:>+8.2f}%")
            else:
                print(f"  {strat_name:<35s} {mode:<12s}       0     0    N/A     N/A      N/A")

    # Parameter differences
    print(f"\n{'=' * 100}")
    print("PARAMETER DIFFERENCES:")
    print(f"{'=' * 100}")
    print(f"  {'Parameter':<35s} {'STRICT':>18s} {'LOOSENED':>18s}")
    print("  " + "-" * 75)
    print(f"  {'Williams %R threshold':<35s} {'-80':>18s} {'-75':>18s}")
    print(f"  {'Williams %R SMA filter':<35s} {'Yes (< SMA 21)':>18s} {'No':>18s}")
    print(f"  {'Williams %R min hold (hours)':<35s} {'6':>18s} {'6':>18s}")
    print(f"  {'BB sigma':<35s} {'2.0':>18s} {'1.5':>18s}")
    print(f"  {'BB RSI entry':<35s} {'< 35':>18s} {'< 40':>18s}")
    print(f"  {'ADX DI crossover required':<35s} {'Yes (15 bars)':>18s} {'No':>18s}")
    print(f"  {'ADX RSI range':<35s} {'35-78':>18s} {'30-80':>18s}")

    # Exit reason breakdown for both
    for mode, mode_trades in [('STRICT', strict_trades), ('LOOSENED', loosened_trades)]:
        print(f"\n{'=' * 100}")
        print(f"EXIT REASONS ({mode}):")
        print(f"{'=' * 100}")
        reasons = {}
        for t in mode_trades:
            r = t['exit_reason']
            if r not in reasons:
                reasons[r] = {'count': 0, 'pnl': 0}
            reasons[r]['count'] += 1
            reasons[r]['pnl'] += t['pnl_pct']
        for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]['count']):
            print(f"  {reason}: {stats['count']} trades, {stats['pnl']:+.2f}% total P&L")

    # Trade log (first 30 of each)
    for mode, mode_trades in [('STRICT', strict_trades), ('LOOSENED', loosened_trades)]:
        print(f"\n{'=' * 100}")
        show_count = min(30, len(mode_trades))
        print(f"TRADE LOG ({mode}) — showing {show_count} of {len(mode_trades)} trades:")
        print(f"{'=' * 100}")
        for idx, t in enumerate(mode_trades[:30], 1):
            marker = "WIN " if t['win'] else "LOSS"
            strat_short = {
                'Williams %R Mean Reversion': 'WR ',
                'ADX Momentum Thrust': 'ADX',
                'Bollinger Band Mean Reversion': 'BB '
            }.get(t['strategy'], '???')
            entry_date = str(t['entry_time'])[:16]
            exit_date = str(t['exit_time'])[:16]
            print(f"  {idx:3d}. [{marker}] [{strat_short}] {entry_date} -> {exit_date} | "
                  f"${t['entry_price']:>10,.2f} -> ${t['exit_price']:>10,.2f} | "
                  f"P&L: {t['pnl_pct']:+7.2f}% | ${t['portfolio']:>10,.2f} | "
                  f"{t['exit_reason']}")


def save_results(strict_stats, loosened_stats, strict_trades, loosened_trades):
    """Save comparison results to JSON"""
    def clean_trades(trades):
        return [{**t, 'entry_time': str(t['entry_time']), 'exit_time': str(t['exit_time'])} for t in trades]

    results = {
        'strict': {**strict_stats, 'trades_detail': clean_trades(strict_trades)},
        'loosened': {**loosened_stats, 'trades_detail': clean_trades(loosened_trades)},
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_comparison_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("UNIFIED 4-YEAR HOURLY BACKTEST — STRICT vs LOOSENED COMPARISON")
    print("Starting Portfolio: $1,000 each")
    print("=" * 100 + "\n")

    # Load data once
    df = fetch_and_cache_hourly_data()
    if df is None:
        return

    df, regime_counts = classify_regimes(df)

    # Run STRICT backtest
    strict_trades, strict_portfolio = backtest_unified(df, STRICT_PARAMS)
    strict_stats = compute_stats(strict_trades, 'STRICT')

    # Run LOOSENED backtest
    loosened_trades, loosened_portfolio = backtest_unified(df, LOOSENED_PARAMS)
    loosened_stats = compute_stats(loosened_trades, 'LOOSENED')

    # Print comparison
    print_comparison(strict_stats, loosened_stats, strict_trades, loosened_trades)

    # Save
    save_results(strict_stats, loosened_stats, strict_trades, loosened_trades)


if __name__ == "__main__":
    main()
