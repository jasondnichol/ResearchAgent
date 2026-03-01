"""
15-Minute Timeframe Backtest — MTF Momentum
=============================================
Tests whether the proven 1h MTF Momentum strategy ($108/day, PF 1.65)
produces a better edge on 15-minute bars.

Hypothesis: 15m gives more trade opportunities and more precise entries.

Approach:
1. Load 3 months of cached 15m data
2. Add indicators adapted for 15m (shorter EMA periods)
3. Add daily context with look-ahead fix (prior-day only)
4. Add regime data (ADX >= 25 from BTC daily bars)
5. Sweep parameters for MTF Momentum on 15m
6. Test a "scalp" variant with tighter stops
7. Walk-forward IS/OOS split (2mo train / 1mo test)
8. Compare vs 1h reference: $108/day, PF 1.65, 60% profitable days

Usage:
    PYTHONIOENCODING=utf-8 venv/Scripts/python backtest_15m_test.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from fetch_intraday import fetch_all_intraday, INTRADAY_COINS

# ── Configuration ────────────────────────────────────────────────────────────

STARTING_CAPITAL = 10_000
FUTURES_FEE = 0.0006       # 0.06% per side (Coinbase CFM taker)
MAX_POSITIONS = 3
RISK_PER_TRADE = 0.015     # 1.5% risk per trade
MAX_DAILY_LOSS = 0.03      # 3% daily loss limit
LEVERAGE = 2.0

TRADE_COINS = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'SUI-USD', 'LINK-USD',
               'ADA-USD', 'DOGE-USD', 'NEAR-USD']

# 1h reference metrics (from walk-forward testing)
REF_1H_DAILY_PNL = 108
REF_1H_PF = 1.65
REF_1H_PROFITABLE_DAYS = 60  # %


# ── Indicator Calculations for 15m ──────────────────────────────────────────

def add_indicators_15m(df):
    """Add technical indicators tuned for 15-minute bars.

    Key adjustments vs 1h:
    - EMA(50) on 15m = 50 bars = 12.5 hours (vs 50h on 1h)
    - RSI(14) on 15m = 14 bars = 3.5 hours
    - ATR(14) on 15m = 14 bars = 3.5 hours
    - Volume SMA uses 20 bars (5 hours of data)
    """
    df = df.copy()

    # EMAs adapted for 15m
    for period in [9, 21, 50, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # SMAs
    for period in [20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()

    # RSI(14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR(14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Volume metrics
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20'].replace(0, np.nan)

    # Momentum returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)   # 1 hour
    df['return_16'] = df['close'].pct_change(16)  # 4 hours (~ return_4 on 1h)
    df['return_96'] = df['close'].pct_change(96)  # 24 hours (~ return_24 on 1h)

    return df


# ── Daily Context with Look-Ahead Fix ──────────────────────────────────────

def fix_daily_context_15m(df_15m):
    """Add daily-timeframe context to 15m data using PRIOR day's values only.

    Maps prior-day daily EMA/trend/ATR to ALL 15m bars of the current day.
    This avoids look-ahead bias.
    """
    df = df_15m.copy()
    df['date'] = df['time'].dt.date

    # Build daily bars from 15m data
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index()

    # Daily EMAs
    daily['d_ema_21'] = daily['close'].ewm(span=21, adjust=False).mean()
    daily['d_sma_200'] = daily['close'].rolling(200).mean()
    daily['d_trend'] = (daily['close'] > daily['d_ema_21']).astype(int)

    # Daily ATR
    d_hl = daily['high'] - daily['low']
    d_hc = (daily['high'] - daily['close'].shift(1)).abs()
    d_lc = (daily['low'] - daily['close'].shift(1)).abs()
    d_tr = pd.concat([d_hl, d_hc, d_lc], axis=1).max(axis=1)
    daily['d_atr'] = d_tr.rolling(14).mean()

    # SHIFT BY 1 DAY - use prior day's values for today
    daily['d_ema_21_shifted'] = daily['d_ema_21'].shift(1)
    daily['d_sma_200_shifted'] = daily['d_sma_200'].shift(1)
    daily['d_trend_shifted'] = daily['d_trend'].shift(1)
    daily['d_atr_shifted'] = daily['d_atr'].shift(1)

    # Map shifted values back to 15m bars
    daily_map = daily.set_index('date')[[
        'd_ema_21_shifted', 'd_sma_200_shifted', 'd_trend_shifted', 'd_atr_shifted'
    ]].to_dict('index')

    df['d_ema_21'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_ema_21_shifted'))
    df['d_sma_200'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_sma_200_shifted'))
    df['d_trend'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_trend_shifted', 0))
    df['d_atr'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_atr_shifted'))

    return df


# ── Regime Classification ──────────────────────────────────────────────────

def compute_daily_regime_15m(coin_data, btc_key='BTC-USD'):
    """Compute daily regime (ADX, ATR rank, BTC slope) from 15m data.

    Uses prior-day values only (shift by 1).
    """
    # Build daily bars from 15m
    daily_bars = {}
    for symbol, df in coin_data.items():
        df_copy = df.copy()
        df_copy['date'] = df_copy['time'].dt.date
        daily = df_copy.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).reset_index()
        daily_bars[symbol] = daily

    btc = daily_bars.get(btc_key)
    if btc is None or len(btc) < 30:
        return {}

    # True Range
    btc['prev_close'] = btc['close'].shift(1)
    btc['tr'] = pd.concat([
        btc['high'] - btc['low'],
        (btc['high'] - btc['prev_close']).abs(),
        (btc['low'] - btc['prev_close']).abs(),
    ], axis=1).max(axis=1)

    # +DM / -DM
    btc['plus_dm'] = np.where(
        (btc['high'] - btc['high'].shift(1)) > (btc['low'].shift(1) - btc['low']),
        np.maximum(btc['high'] - btc['high'].shift(1), 0), 0)
    btc['minus_dm'] = np.where(
        (btc['low'].shift(1) - btc['low']) > (btc['high'] - btc['high'].shift(1)),
        np.maximum(btc['low'].shift(1) - btc['low'], 0), 0)

    # Smoothed (Wilder's, period 14)
    period = 14
    btc['atr_14'] = btc['tr'].ewm(alpha=1/period, adjust=False).mean()
    btc['plus_di'] = 100 * btc['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / btc['atr_14']
    btc['minus_di'] = 100 * btc['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / btc['atr_14']
    btc['dx'] = 100 * (btc['plus_di'] - btc['minus_di']).abs() / (btc['plus_di'] + btc['minus_di']).replace(0, np.nan)
    btc['adx'] = btc['dx'].ewm(alpha=1/period, adjust=False).mean()

    # ATR rank (percentile over 20 days)
    btc['atr_rank'] = btc['atr_14'].rolling(20).rank(pct=True)

    # EMA(21) slope (3-day rate of change)
    btc['ema_21'] = btc['close'].ewm(span=21, adjust=False).mean()
    btc['ema_slope'] = (btc['ema_21'] - btc['ema_21'].shift(3)) / btc['ema_21'].shift(3) * 100

    # Classify - shift by 1 to use PRIOR day
    regime_map = {}
    for i in range(1, len(btc)):
        prev = btc.iloc[i-1]
        date = btc.iloc[i]['date']

        adx = prev['adx'] if not pd.isna(prev['adx']) else 15
        atr_rank = prev['atr_rank'] if not pd.isna(prev['atr_rank']) else 0.5
        ema_slope = prev['ema_slope'] if not pd.isna(prev['ema_slope']) else 0

        regime_map[date] = {
            'adx': adx,
            'atr_rank': atr_rank,
            'btc_slope': ema_slope,
            'regime': 'trending' if adx > 25 else 'ranging',
        }

    return regime_map


def add_regime_to_15m(df, regime_map):
    """Map daily regime classification onto 15m bars."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['regime'] = df['date'].map(lambda d: regime_map.get(d, {}).get('regime', 'unknown'))
    df['d_adx'] = df['date'].map(lambda d: regime_map.get(d, {}).get('adx', 15))
    df['d_atr_rank'] = df['date'].map(lambda d: regime_map.get(d, {}).get('atr_rank', 0.5))
    df['d_btc_slope'] = df['date'].map(lambda d: regime_map.get(d, {}).get('btc_slope', 0))
    return df


# ── Strategy: MTF Momentum for 15m ─────────────────────────────────────────

def strategy_mtf_momentum_15m(df, params):
    """MTF Momentum adapted for 15-minute bars.

    Same logic as 1h version:
    - Daily trend bias (from prior day) + intrabar RSI dip entry + volume confirmation
    - ADX regime filter (only trade when trending)

    Key 15m adjustments:
    - EMA(50) on 15m = 12.5 hours (vs 50 hours on 1h)
    - RSI dip thresholds may need adjustment
    - max_hold in 15m bars (24 bars = 6 hours, 48 = 12h, 96 = 24h)
    """
    vol_thresh = params.get('vol_threshold', 1.2)
    atr_stop = params.get('atr_stop', 2.0)
    atr_target = params.get('atr_target', 3.0)
    max_hold = params.get('max_hold', 48)  # 48 bars = 12 hours
    adx_threshold = params.get('adx_threshold', 25)
    rsi_entry = params.get('rsi_entry', 45)     # RSI dip threshold for longs
    rsi_exit_short = params.get('rsi_exit_short', 55)  # RSI rise threshold for shorts

    signals = []
    for i in range(201, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row.get('d_trend')) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # Regime filter (ADX)
        d_adx = row.get('d_adx', 15)
        if pd.isna(d_adx):
            d_adx = 15
        if d_adx < adx_threshold:
            continue

        # Long: daily bullish, RSI dipped then bouncing, above EMA(21)
        if (row['d_trend'] == 1 and
            prev['rsi'] < rsi_entry and row['rsi'] > prev['rsi'] and
            row['rsi'] > (rsi_entry - 10) and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

        # Short: daily bearish, RSI rose then reversing, below EMA(21)
        if (row['d_trend'] == 0 and
            prev['rsi'] > rsi_exit_short and row['rsi'] < prev['rsi'] and
            row['rsi'] < (rsi_exit_short + 10) and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

    return signals


def strategy_scalp_15m(df, params):
    """Scalp variant: tighter stops, quick targets, short holds.

    - Tighter ATR stop (1.0x)
    - Quick target (1.5x ATR)
    - Max hold 16 bars (4 hours)
    - Higher volume threshold for conviction
    """
    vol_thresh = params.get('vol_threshold', 1.3)
    atr_stop = params.get('atr_stop', 1.0)
    atr_target = params.get('atr_target', 1.5)
    max_hold = params.get('max_hold', 16)  # 16 bars = 4 hours
    adx_threshold = params.get('adx_threshold', 25)
    rsi_entry = params.get('rsi_entry', 40)

    signals = []
    for i in range(201, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row.get('d_trend')) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # Regime filter
        d_adx = row.get('d_adx', 15)
        if pd.isna(d_adx):
            d_adx = 15
        if d_adx < adx_threshold:
            continue

        # Long scalp: quick RSI bounce with volume
        if (row['d_trend'] == 1 and
            prev['rsi'] < rsi_entry and row['rsi'] > prev['rsi'] and
            row['rsi'] > (rsi_entry - 10) and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

        # Short scalp
        if (row['d_trend'] == 0 and
            prev['rsi'] > (100 - rsi_entry) and row['rsi'] < prev['rsi'] and
            row['rsi'] < (100 - rsi_entry + 10) and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

    return signals


# ── Portfolio Backtester (same pattern as backtest_intraday.py) ─────────────

def backtest_strategy(coin_data_processed, strategy_func, strategy_params,
                       starting_capital=STARTING_CAPITAL, max_positions=MAX_POSITIONS,
                       risk_per_trade=RISK_PER_TRADE, leverage=LEVERAGE,
                       fee_rate=FUTURES_FEE, max_daily_loss=MAX_DAILY_LOSS):
    """Portfolio-level backtest across all coins."""
    equity = starting_capital
    peak_equity = starting_capital
    positions = {}
    trades = []
    equity_curve = []
    daily_pnl = defaultdict(float)

    # Gather signals from all coins
    all_signals = []
    for symbol, df in coin_data_processed.items():
        if symbol not in TRADE_COINS:
            continue
        signals = strategy_func(df, strategy_params)
        for s in signals:
            s['symbol'] = symbol
        all_signals.extend(signals)

    all_signals.sort(key=lambda s: s['time'])

    if not all_signals:
        return {'trades': [], 'equity_curve': [], 'stats': None}

    # Build price lookups
    price_lookup = {}
    for symbol, df in coin_data_processed.items():
        if symbol not in TRADE_COINS:
            continue
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time']] = row
        price_lookup[symbol] = lookup

    # All unique timestamps
    all_times = set()
    for symbol, df in coin_data_processed.items():
        if symbol in TRADE_COINS:
            for t in df['time']:
                all_times.add(t)
    all_times = sorted(all_times)

    # Signal lookup by time
    signal_by_time = defaultdict(list)
    for s in all_signals:
        signal_by_time[s['time']].append(s)

    current_day = None
    day_pnl = 0.0
    day_locked = False

    for t in all_times:
        day = t.date() if hasattr(t, 'date') else t

        if day != current_day:
            current_day = day
            day_pnl = 0.0
            day_locked = False

        if day_locked:
            continue

        # Check exits
        to_close = []
        for sym, pos in list(positions.items()):
            row = price_lookup.get(sym, {}).get(t)
            if row is None:
                continue

            close_price = row['close']
            high = row['high']
            low = row['low']
            pos['hold_candles'] += 1

            exit_reason = None
            exit_price = close_price

            if pos['side'] == 'long':
                if low <= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                elif high >= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                elif close_price > pos.get('high_watermark', pos['entry']):
                    pos['high_watermark'] = close_price
                    new_stop = close_price - 1.5 * pos['atr']
                    if new_stop > pos['stop']:
                        pos['stop'] = new_stop
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price

            elif pos['side'] == 'short':
                if high >= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                elif low <= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                elif close_price < pos.get('low_watermark', pos['entry']):
                    pos['low_watermark'] = close_price
                    new_stop = close_price + 1.5 * pos['atr']
                    if new_stop < pos['stop']:
                        pos['stop'] = new_stop
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price

            if exit_reason:
                to_close.append((sym, pos, exit_price, exit_reason))

        for sym, pos, exit_price, exit_reason in to_close:
            if pos['side'] == 'long':
                gross_pnl = (exit_price - pos['entry']) / pos['entry']
            else:
                gross_pnl = (pos['entry'] - exit_price) / pos['entry']

            leveraged_pnl = gross_pnl * leverage
            fee_cost = 2 * fee_rate
            net_pnl_pct = leveraged_pnl - fee_cost
            net_pnl_dollar = pos['size'] * net_pnl_pct

            equity += net_pnl_dollar
            day_pnl += net_pnl_dollar

            if equity > peak_equity:
                peak_equity = equity

            trades.append({
                'symbol': sym, 'side': pos['side'],
                'entry_time': pos['entry_time'], 'exit_time': t,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'hold_candles': pos['hold_candles'],
                'gross_pnl_pct': gross_pnl * 100,
                'net_pnl_pct': net_pnl_pct * 100,
                'net_pnl_dollar': net_pnl_dollar,
                'exit_reason': exit_reason,
                'equity_after': equity,
            })

            del positions[sym]

            if day_pnl / starting_capital < -max_daily_loss:
                day_locked = True
                break

        if day_locked:
            equity_curve.append({'time': t, 'equity': equity})
            continue

        # Check entries
        if t in signal_by_time:
            for sig in signal_by_time[t]:
                sym = sig['symbol']
                if sym in positions:
                    continue
                if len(positions) >= max_positions:
                    break
                if equity < starting_capital * 0.5:
                    break

                stop_distance = abs(sig['entry'] - sig['stop']) / sig['entry']
                if stop_distance <= 0:
                    continue

                risk_amount = equity * risk_per_trade
                position_size = risk_amount / stop_distance
                position_size = min(position_size, equity * 0.95 / leverage)

                positions[sym] = {
                    'side': sig['side'],
                    'entry': sig['entry'],
                    'entry_time': sig['time'],
                    'stop': sig['stop'],
                    'target': sig['target'],
                    'atr': sig['atr'],
                    'size': position_size,
                    'hold_candles': 0,
                    'max_hold': sig['max_hold'],
                    'high_watermark': sig['entry'],
                    'low_watermark': sig['entry'],
                }

                entry_fee = position_size * fee_rate
                equity -= entry_fee

        equity_curve.append({'time': t, 'equity': equity})

    # Close remaining positions
    for sym, pos in list(positions.items()):
        last_df = coin_data_processed[sym]
        exit_price = last_df.iloc[-1]['close']

        if pos['side'] == 'long':
            gross_pnl = (exit_price - pos['entry']) / pos['entry']
        else:
            gross_pnl = (pos['entry'] - exit_price) / pos['entry']

        leveraged_pnl = gross_pnl * leverage
        net_pnl_pct = leveraged_pnl - 2 * fee_rate
        net_pnl_dollar = pos['size'] * net_pnl_pct
        equity += net_pnl_dollar

        trades.append({
            'symbol': sym, 'side': pos['side'],
            'entry_time': pos['entry_time'],
            'exit_time': last_df.iloc[-1]['time'],
            'entry_price': pos['entry'], 'exit_price': exit_price,
            'hold_candles': pos['hold_candles'],
            'gross_pnl_pct': gross_pnl * 100,
            'net_pnl_pct': net_pnl_pct * 100,
            'net_pnl_dollar': net_pnl_dollar,
            'exit_reason': 'end_of_data',
            'equity_after': equity,
        })

    stats = compute_stats(trades, equity, starting_capital)
    return {'trades': trades, 'equity_curve': equity_curve, 'stats': stats}


def compute_stats(trades, final_equity, starting_capital):
    """Compute comprehensive trading statistics."""
    if not trades:
        return None

    n_trades = len(trades)
    winners = [t for t in trades if t['net_pnl_dollar'] > 0]
    losers = [t for t in trades if t['net_pnl_dollar'] <= 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    gross_profit = sum(t['net_pnl_dollar'] for t in winners)
    gross_loss = abs(sum(t['net_pnl_dollar'] for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t['net_pnl_pct'] for t in winners]) if winners else 0
    avg_loss = np.mean([t['net_pnl_pct'] for t in losers]) if losers else 0
    avg_hold = np.mean([t['hold_candles'] for t in trades])

    net_return = (final_equity - starting_capital) / starting_capital * 100

    # Max drawdown
    equity_values = [starting_capital] + [t['equity_after'] for t in trades]
    peak = equity_values[0]
    max_dd = 0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Trades per day
    if len(trades) >= 2:
        first_time = trades[0]['entry_time']
        last_time = trades[-1]['entry_time']
        days = max((last_time - first_time).days, 1)
        trades_per_day = n_trades / days
    else:
        trades_per_day = 0
        days = 0

    # Daily returns
    daily_returns = defaultdict(float)
    for t in trades:
        d = t['exit_time'].date() if hasattr(t['exit_time'], 'date') else t['exit_time']
        daily_returns[d] += t['net_pnl_dollar']

    profitable_days = sum(1 for v in daily_returns.values() if v > 0)
    total_days_traded = len(daily_returns)
    day_win_rate = profitable_days / total_days_traded * 100 if total_days_traded > 0 else 0

    avg_daily_pnl = np.mean(list(daily_returns.values())) if daily_returns else 0
    daily_pnl_std = np.std(list(daily_returns.values())) if len(daily_returns) > 1 else 0
    sharpe = (avg_daily_pnl / daily_pnl_std * math.sqrt(365)) if daily_pnl_std > 0 else 0

    # Exit breakdown
    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t['exit_reason']] += 1

    # Long vs short
    long_trades = [t for t in trades if t['side'] == 'long']
    short_trades = [t for t in trades if t['side'] == 'short']

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'net_return_pct': net_return,
        'net_return_dollar': final_equity - starting_capital,
        'final_equity': final_equity,
        'max_drawdown': max_dd * 100,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'avg_hold_candles': avg_hold,
        'trades_per_day': trades_per_day,
        'total_days': days,
        'profitable_days_pct': day_win_rate,
        'avg_daily_pnl': avg_daily_pnl,
        'sharpe': sharpe,
        'exit_counts': dict(exit_counts),
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'long_wr': len([t for t in long_trades if t['net_pnl_dollar'] > 0]) / max(len(long_trades), 1) * 100,
        'short_wr': len([t for t in short_trades if t['net_pnl_dollar'] > 0]) / max(len(short_trades), 1) * 100,
    }


# ── Reporting ──────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_result_line(name, stats, highlight=False):
    """One-line result."""
    if not stats:
        print(f"  {name:<45s}   -- no trades --")
        return
    s = stats
    marker = " <<<" if highlight else ""
    print(f"  {name:<45s} {s['n_trades']:>4d} tr  {s['trades_per_day']:.1f}/d  "
          f"{s['win_rate']:>5.1f}% WR  PF {s['profit_factor']:.2f}  "
          f"{s['net_return_pct']:>+7.1f}%  DD {s['max_drawdown']:.1f}%  "
          f"${s['avg_daily_pnl']:>+6.0f}/d{marker}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print_section("15-MINUTE TIMEFRAME BACKTEST -- MTF MOMENTUM")
    print(f"  Starting capital: ${STARTING_CAPITAL:,}")
    print(f"  Leverage: {LEVERAGE}x  |  Fee: {FUTURES_FEE*100:.2f}%/side  |  Max positions: {MAX_POSITIONS}")
    print(f"  Risk/trade: {RISK_PER_TRADE*100:.1f}%  |  Max daily loss: {MAX_DAILY_LOSS*100:.0f}%")
    print(f"  Coins: {', '.join(TRADE_COINS)}")
    print(f"  1h Reference: ${REF_1H_DAILY_PNL}/day, PF {REF_1H_PF}, {REF_1H_PROFITABLE_DAYS}% profitable days")

    # ── Load 15m data ──────────────────────────────────────────
    print_section("LOADING 15m DATA (3 months)")
    coin_data_raw = fetch_all_intraday(timeframe='15m', months=3)

    if not coin_data_raw:
        print("  ERROR: No 15m data available. Run fetch_intraday.py 15m 3 first.")
        return

    # ── Compute regime ─────────────────────────────────────────
    print_section("COMPUTING REGIME CLASSIFICATION")
    regime_map = compute_daily_regime_15m(coin_data_raw)
    if regime_map:
        trending_days = sum(1 for v in regime_map.values() if v['regime'] == 'trending')
        ranging_days = sum(1 for v in regime_map.values() if v['regime'] == 'ranging')
        print(f"  Regime: {trending_days} trending days, {ranging_days} ranging days "
              f"({trending_days/(trending_days+ranging_days)*100:.0f}% trending)")

        adx_values = [v['adx'] for v in regime_map.values()]
        print(f"  ADX: min={min(adx_values):.1f}, median={np.median(adx_values):.1f}, max={max(adx_values):.1f}")
    else:
        print("  WARNING: Could not compute regime (BTC data missing?)")

    # ── Process: indicators + daily context + regime ───────────
    print_section("COMPUTING INDICATORS (15m)")
    coin_data = {}
    for symbol, df in coin_data_raw.items():
        print(f"  {symbol}: processing...", end='', flush=True)
        df = add_indicators_15m(df)
        df = fix_daily_context_15m(df)
        if regime_map:
            df = add_regime_to_15m(df, regime_map)
        coin_data[symbol] = df
        print(f" {len(df):,} candles")

    # ══════════════════════════════════════════════════════════
    # SECTION 1: Full-Period Parameter Sweep
    # ══════════════════════════════════════════════════════════
    print_section("SECTION 1: PARAMETER SWEEP (full 3 months)")

    # Define sweep grid
    sweep_configs = []

    # Main MTF Momentum sweep
    for adx_t in [20, 25]:
        for rsi_e in [35, 40, 45]:
            for atr_s in [1.5, 2.0]:
                for atr_tgt in [2.0, 3.0]:
                    for max_h in [24, 48, 96]:  # 6h, 12h, 24h
                        for vol_t in [0.0, 1.0, 1.2]:
                            label = f"ADX{adx_t}_RSI{rsi_e}_S{atr_s}_T{atr_tgt}_H{max_h}_V{vol_t}"
                            params = {
                                'adx_threshold': adx_t,
                                'rsi_entry': rsi_e,
                                'rsi_exit_short': 100 - rsi_e,
                                'atr_stop': atr_s,
                                'atr_target': atr_tgt,
                                'max_hold': max_h,
                                'vol_threshold': vol_t,
                            }
                            sweep_configs.append((label, params))

    print(f"  Testing {len(sweep_configs)} MTF Momentum configs + 1 scalp variant...")
    print(f"\n  {'Config':<55s} {'Tr':>4s} {'T/d':>4s} {'WR':>6s} {'PF':>5s} "
          f"{'Ret':>8s} {'DD':>6s} {'$/d':>7s}")
    print(f"  {'-'*95}")

    all_results = []
    best_daily = -9999
    best_config = None
    best_result = None

    for i, (label, params) in enumerate(sweep_configs):
        r = backtest_strategy(coin_data, strategy_mtf_momentum_15m, params)
        s = r['stats']

        if s and s['n_trades'] >= 5:
            all_results.append((label, params, s, r))

            is_best = False
            if s['avg_daily_pnl'] > best_daily and s['profit_factor'] >= 1.2:
                best_daily = s['avg_daily_pnl']
                best_config = (label, params)
                best_result = r
                is_best = True

            # Only print configs with at least some trades and positive PF
            if s['profit_factor'] >= 1.0 or s['n_trades'] >= 20:
                marker = " <<<" if is_best else ""
                print(f"  {label:<55s} {s['n_trades']:>4d} {s['trades_per_day']:>3.1f} "
                      f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} "
                      f"{s['net_return_pct']:>+7.1f}% {s['max_drawdown']:>5.1f}% "
                      f"${s['avg_daily_pnl']:>+6.0f}{marker}")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  ... processed {i+1}/{len(sweep_configs)} configs ...", flush=True)

    # Sort by daily PnL and show top 10
    all_results.sort(key=lambda x: x[2]['avg_daily_pnl'], reverse=True)

    print_section("TOP 10 CONFIGS BY DAILY P&L")
    print(f"\n  {'Rank':>4s}  {'Config':<55s} {'Tr':>4s} {'T/d':>4s} {'WR':>6s} {'PF':>5s} "
          f"{'Ret':>8s} {'DD':>6s} {'$/d':>7s}")
    print(f"  {'-'*100}")

    for rank, (label, params, s, r) in enumerate(all_results[:10], 1):
        print(f"  {rank:>4d}  {label:<55s} {s['n_trades']:>4d} {s['trades_per_day']:>3.1f} "
              f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} "
              f"{s['net_return_pct']:>+7.1f}% {s['max_drawdown']:>5.1f}% "
              f"${s['avg_daily_pnl']:>+6.0f}")

    # Sort by PF and show top 10
    pf_sorted = sorted([x for x in all_results if x[2]['n_trades'] >= 10],
                        key=lambda x: x[2]['profit_factor'], reverse=True)

    print_section("TOP 10 CONFIGS BY PROFIT FACTOR (min 10 trades)")
    print(f"\n  {'Rank':>4s}  {'Config':<55s} {'Tr':>4s} {'T/d':>4s} {'WR':>6s} {'PF':>5s} "
          f"{'Ret':>8s} {'DD':>6s} {'$/d':>7s}")
    print(f"  {'-'*100}")

    for rank, (label, params, s, r) in enumerate(pf_sorted[:10], 1):
        print(f"  {rank:>4d}  {label:<55s} {s['n_trades']:>4d} {s['trades_per_day']:>3.1f} "
              f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} "
              f"{s['net_return_pct']:>+7.1f}% {s['max_drawdown']:>5.1f}% "
              f"${s['avg_daily_pnl']:>+6.0f}")

    # ══════════════════════════════════════════════════════════
    # SECTION 2: Scalp Variant
    # ══════════════════════════════════════════════════════════
    print_section("SECTION 2: SCALP VARIANT (tight stops, quick targets)")

    scalp_params = {
        'adx_threshold': 25,
        'rsi_entry': 40,
        'atr_stop': 1.0,
        'atr_target': 1.5,
        'max_hold': 16,
        'vol_threshold': 1.3,
    }
    r_scalp = backtest_strategy(coin_data, strategy_scalp_15m, scalp_params)
    print(f"\n  Scalp (ADX>=25, 1.0xATR stop, 1.5xATR target, 16bar hold):")
    print_result_line("Scalp Default", r_scalp['stats'])

    # Scalp variants
    scalp_variants = [
        ('Scalp ADX20 V1.0', {'adx_threshold': 20, 'rsi_entry': 40, 'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 16, 'vol_threshold': 1.0}),
        ('Scalp ADX25 V0.0', {'adx_threshold': 25, 'rsi_entry': 40, 'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 16, 'vol_threshold': 0.0}),
        ('Scalp ADX25 RSI35', {'adx_threshold': 25, 'rsi_entry': 35, 'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 16, 'vol_threshold': 1.3}),
        ('Scalp ADX25 H24', {'adx_threshold': 25, 'rsi_entry': 40, 'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 24, 'vol_threshold': 1.3}),
        ('Scalp ADX25 S1.5 T2.0', {'adx_threshold': 25, 'rsi_entry': 40, 'atr_stop': 1.5, 'atr_target': 2.0, 'max_hold': 16, 'vol_threshold': 1.3}),
    ]

    print(f"\n  {'Variant':<45s} {'Tr':>4s} {'T/d':>4s} {'WR':>6s} {'PF':>5s} "
          f"{'Ret':>8s} {'DD':>6s} {'$/d':>7s}")
    print(f"  {'-'*85}")

    for vname, vparams in scalp_variants:
        r = backtest_strategy(coin_data, strategy_scalp_15m, vparams)
        print_result_line(vname, r['stats'])

    # ══════════════════════════════════════════════════════════
    # SECTION 3: Walk-Forward IS/OOS Split
    # ══════════════════════════════════════════════════════════
    print_section("SECTION 3: WALK-FORWARD (2mo IS / 1mo OOS)")

    # Find date range
    all_starts = []
    all_ends = []
    for symbol, df in coin_data.items():
        if symbol in TRADE_COINS:
            all_starts.append(df['time'].min())
            all_ends.append(df['time'].max())

    data_start = max(all_starts)
    data_end = min(all_ends)
    total_days = (data_end - data_start).days
    print(f"  Data range: {data_start.date()} to {data_end.date()} ({total_days} days)")

    # Split: first 2 months = IS, last 1 month = OOS
    is_end = data_start + timedelta(days=int(total_days * 2 / 3))
    print(f"  IS period: {data_start.date()} to {is_end.date()}")
    print(f"  OOS period: {is_end.date()} to {data_end.date()}")

    # Split data
    is_data = {}
    oos_data = {}
    for symbol, df in coin_data.items():
        if symbol not in TRADE_COINS:
            continue
        is_df = df[df['time'] < is_end].copy().reset_index(drop=True)
        oos_df = df[df['time'] >= is_end].copy().reset_index(drop=True)
        if len(is_df) > 200:
            is_data[symbol] = is_df
        if len(oos_df) > 200:
            oos_data[symbol] = oos_df

    print(f"  IS candles: {sum(len(df) for df in is_data.values()):,}")
    print(f"  OOS candles: {sum(len(df) for df in oos_data.values()):,}")

    # Test top configs from sweep on IS and OOS
    test_configs = []

    # Top 5 by daily PnL
    for label, params, s, r in all_results[:5]:
        test_configs.append((f"[PNL] {label}", strategy_mtf_momentum_15m, params))

    # Top 3 by PF (if different from PnL top 5)
    seen = set(x[0] for x in test_configs)
    for label, params, s, r in pf_sorted[:3]:
        key = f"[PF] {label}"
        if key not in seen:
            test_configs.append((key, strategy_mtf_momentum_15m, params))
            seen.add(key)

    # Add best scalp
    test_configs.append(("Scalp (1.0xATR/1.5xATR/16bar)", strategy_scalp_15m, scalp_params))

    # If best_config found
    if best_config:
        label, params = best_config
        if f"[BEST] {label}" not in seen:
            test_configs.append((f"[BEST] {label}", strategy_mtf_momentum_15m, params))

    print(f"\n  Testing {len(test_configs)} configs on IS and OOS...")
    print(f"\n  {'Config':<55s} {'IS Tr':>5s} {'IS WR':>6s} {'IS PF':>5s} {'IS $/d':>7s}  "
          f"{'OOS Tr':>6s} {'OOS WR':>6s} {'OOS PF':>6s} {'OOS $/d':>7s}")
    print(f"  {'-'*110}")

    oos_results = []
    for name, func, params in test_configs:
        r_is = backtest_strategy(is_data, func, params)
        r_oos = backtest_strategy(oos_data, func, params)
        s_is = r_is['stats']
        s_oos = r_oos['stats']

        is_tr = s_is['n_trades'] if s_is else 0
        is_wr = f"{s_is['win_rate']:.0f}%" if s_is else "--"
        is_pf = f"{s_is['profit_factor']:.2f}" if s_is else "--"
        is_dpnl = f"${s_is['avg_daily_pnl']:+.0f}" if s_is else "--"

        oos_tr = s_oos['n_trades'] if s_oos else 0
        oos_wr = f"{s_oos['win_rate']:.0f}%" if s_oos else "--"
        oos_pf = f"{s_oos['profit_factor']:.2f}" if s_oos else "--"
        oos_dpnl = f"${s_oos['avg_daily_pnl']:+.0f}" if s_oos else "--"

        print(f"  {name:<55s} {is_tr:>5d} {is_wr:>6s} {is_pf:>5s} {is_dpnl:>7s}  "
              f"{oos_tr:>6d} {oos_wr:>6s} {oos_pf:>6s} {oos_dpnl:>7s}")

        if s_oos:
            oos_results.append((name, s_oos))

    # ══════════════════════════════════════════════════════════
    # SECTION 4: Best Config Deep Dive
    # ══════════════════════════════════════════════════════════
    print_section("SECTION 4: BEST CONFIG DEEP DIVE")

    if best_result and best_result['stats']:
        s = best_result['stats']
        label, params = best_config
        print(f"\n  Best config: {label}")
        print(f"  Params: {params}")
        print(f"  Trades: {s['n_trades']}  |  Trades/day: {s['trades_per_day']:.1f}")
        print(f"  Win Rate: {s['win_rate']:.1f}%  |  PF: {s['profit_factor']:.2f}  |  Sharpe: {s['sharpe']:.2f}")
        print(f"  Net Return: {s['net_return_pct']:+.1f}% (${s['net_return_dollar']:+,.0f})")
        print(f"  Max Drawdown: {s['max_drawdown']:.1f}%")
        print(f"  Avg Win: {s['avg_win_pct']:+.2f}%  |  Avg Loss: {s['avg_loss_pct']:.2f}%")
        print(f"  Avg Hold: {s['avg_hold_candles']:.1f} bars ({s['avg_hold_candles']*15/60:.1f} hours)")
        print(f"  Longs: {s['n_long']} ({s['long_wr']:.0f}% WR)  |  Shorts: {s['n_short']} ({s['short_wr']:.0f}% WR)")
        print(f"  Profitable Days: {s['profitable_days_pct']:.1f}%  |  Avg Daily P&L: ${s['avg_daily_pnl']:+.0f}")
        print(f"  Exits: {s['exit_counts']}")

        # Per-coin breakdown
        coin_pnl = defaultdict(float)
        coin_trades_n = defaultdict(int)
        for t in best_result['trades']:
            coin_pnl[t['symbol']] += t['net_pnl_dollar']
            coin_trades_n[t['symbol']] += 1

        print(f"\n  Per-Coin P&L:")
        for sym in sorted(coin_pnl.keys(), key=lambda s: coin_pnl[s], reverse=True):
            print(f"    {sym:10s}: ${coin_pnl[sym]:>+8,.0f} ({coin_trades_n[sym]} trades)")

        # Monthly P&L
        monthly = defaultdict(float)
        for t in best_result['trades']:
            m = t['exit_time'].strftime('%Y-%m') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[:7]
            monthly[m] += t['net_pnl_dollar']

        print(f"\n  Monthly P&L:")
        for m in sorted(monthly.keys()):
            bar = '+' * min(60, max(0, int(monthly[m] / 50))) if monthly[m] > 0 else '-' * min(60, max(0, int(-monthly[m] / 50)))
            print(f"    {m}: ${monthly[m]:>+8,.0f}  {bar}")
    else:
        print("  No profitable config found in sweep.")

    # ══════════════════════════════════════════════════════════
    # SECTION 5: 15m vs 1h Comparison
    # ══════════════════════════════════════════════════════════
    print_section("SECTION 5: 15m vs 1h COMPARISON")

    print(f"\n  {'Metric':<30s} {'1h Reference':>15s} {'15m Best (IS)':>15s}")
    print(f"  {'-'*62}")

    if best_result and best_result['stats']:
        s = best_result['stats']
        dpnl_str = f"${s['avg_daily_pnl']:+.0f}"
        hold_str = f"{s['avg_hold_candles']*15/60:.1f}h"
        print(f"  {'Daily P&L':<30s} {'$' + str(REF_1H_DAILY_PNL):>15s} {dpnl_str:>15s}")
        print(f"  {'Profit Factor':<30s} {REF_1H_PF:>15.2f} {s['profit_factor']:>15.2f}")
        print(f"  {'Profitable Days %':<30s} {REF_1H_PROFITABLE_DAYS:>14.0f}% {s['profitable_days_pct']:>14.1f}%")
        print(f"  {'Trades/day':<30s} {'~2-4':>15s} {s['trades_per_day']:>15.1f}")
        print(f"  {'Win Rate':<30s} {'~55%':>15s} {s['win_rate']:>14.1f}%")
        print(f"  {'Max Drawdown':<30s} {'~5-8%':>15s} {s['max_drawdown']:>14.1f}%")
        print(f"  {'Avg Hold':<30s} {'~12-24h':>15s} {hold_str:>15s}")

        beats_1h = s['avg_daily_pnl'] > REF_1H_DAILY_PNL
        if beats_1h:
            print(f"\n  >>> 15m BEATS 1h reference! Daily P&L: ${s['avg_daily_pnl']:+.0f} vs ${REF_1H_DAILY_PNL}")
        else:
            print(f"\n  >>> 15m does NOT beat 1h reference. ${s['avg_daily_pnl']:+.0f}/day vs ${REF_1H_DAILY_PNL}/day")
    else:
        print(f"  No viable 15m config found to compare.")

    # OOS comparison
    if oos_results:
        best_oos = max(oos_results, key=lambda x: x[1]['avg_daily_pnl'])
        oos_name, oos_s = best_oos
        print(f"\n  Best OOS config: {oos_name}")
        print(f"  OOS Daily P&L: ${oos_s['avg_daily_pnl']:+.0f}  |  OOS PF: {oos_s['profit_factor']:.2f}  |  "
              f"OOS WR: {oos_s['win_rate']:.1f}%")

        oos_beats = oos_s['avg_daily_pnl'] > REF_1H_DAILY_PNL
        if oos_beats:
            print(f"  >>> 15m OOS BEATS 1h reference!")
        else:
            pct_of_ref = oos_s['avg_daily_pnl'] / REF_1H_DAILY_PNL * 100 if REF_1H_DAILY_PNL > 0 else 0
            print(f"  >>> 15m OOS = {pct_of_ref:.0f}% of 1h reference")

    # ── Summary statistics ────────────────────────────────────
    print_section("SUMMARY STATISTICS")

    profitable_configs = sum(1 for _, _, s, _ in all_results if s['profit_factor'] > 1.0)
    total_configs = len(all_results)
    viable_configs = sum(1 for _, _, s, _ in all_results
                         if s['profit_factor'] >= 1.3 and s['avg_daily_pnl'] > 0)

    print(f"\n  Total configs tested: {len(sweep_configs)} + scalp variants")
    print(f"  Configs with trades: {total_configs}")
    print(f"  Profitable (PF > 1.0): {profitable_configs} ({profitable_configs/max(total_configs,1)*100:.0f}%)")
    print(f"  Viable (PF >= 1.3 & positive P&L): {viable_configs}")

    if all_results:
        daily_pnls = [s['avg_daily_pnl'] for _, _, s, _ in all_results]
        pfs = [s['profit_factor'] for _, _, s, _ in all_results]
        print(f"\n  Daily P&L range: ${min(daily_pnls):+.0f} to ${max(daily_pnls):+.0f}")
        print(f"  PF range: {min(pfs):.2f} to {max(pfs):.2f}")
        print(f"  Median daily P&L: ${np.median(daily_pnls):+.0f}")
        print(f"  Median PF: {np.median(pfs):.2f}")

    # ── Final verdict ─────────────────────────────────────────
    print_section("FINAL VERDICT")

    if best_result and best_result['stats']:
        s = best_result['stats']
        is_viable = s['profit_factor'] >= 1.3 and s['avg_daily_pnl'] > 50

        if s['avg_daily_pnl'] > REF_1H_DAILY_PNL and is_viable:
            print(f"\n  RESULT: 15m timeframe shows BETTER edge than 1h.")
            print(f"  Best config: ${s['avg_daily_pnl']:+.0f}/day vs 1h ${REF_1H_DAILY_PNL}/day")
            print(f"  RECOMMENDATION: Validate with longer data (6+ months) before deployment.")
        elif is_viable:
            print(f"\n  RESULT: 15m timeframe shows viable edge but DOES NOT beat 1h.")
            print(f"  Best config: ${s['avg_daily_pnl']:+.0f}/day vs 1h ${REF_1H_DAILY_PNL}/day")
            print(f"  RECOMMENDATION: Stick with 1h timeframe. 15m adds complexity without improvement.")
        else:
            print(f"\n  RESULT: 15m timeframe does NOT produce a viable edge.")
            print(f"  Best daily P&L: ${s['avg_daily_pnl']:+.0f}/day (target: >${REF_1H_DAILY_PNL})")
            print(f"  RECOMMENDATION: 15m is too noisy. Stick with 1h timeframe.")
    else:
        print(f"\n  RESULT: No viable 15m configs found.")
        print(f"  RECOMMENDATION: 15m timeframe not suitable for this strategy.")

    print(f"\n{'='*80}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
