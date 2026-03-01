"""
Intraday Strategy Backtester
==============================
Tests multiple intraday strategies on 1h crypto data with realistic
Coinbase CFM futures fees (0.06% taker per side = 0.12% round trip).

Strategies tested:
1. Intraday Donchian Breakout (adapt our proven daily system to 1h)
2. VWAP Mean Reversion (fade overextended moves back to VWAP)
3. EMA Momentum Scalp (trend-following with volume confirmation)
4. Bollinger Band Squeeze Breakout (volatility expansion after compression)
5. Multi-Timeframe Momentum (daily bias + hourly entry)

Goal: Find strategies producing 2-4 high-conviction trades/day with
55%+ WR and 1.5:1+ reward/risk on futures fees.

Usage:
    PYTHONIOENCODING=utf-8 venv/Scripts/python backtest_intraday.py
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ── Import our data fetcher ─────────────────────────────────────────────────
from fetch_intraday import fetch_all_intraday, INTRADAY_COINS

# ── Configuration ────────────────────────────────────────────────────────────

STARTING_CAPITAL = 10_000
FUTURES_FEE = 0.0006       # 0.06% per side (Coinbase CFM taker)
MAX_POSITIONS = 3          # Concurrent positions
RISK_PER_TRADE = 0.015     # 1.5% risk per trade
MAX_DAILY_LOSS = 0.03      # 3% max daily loss — circuit breaker
LEVERAGE = 2.0             # 2x leverage on futures

# Coins to trade (most liquid + volatile, exclude BTC for intraday — too slow)
TRADE_COINS = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'SUI-USD', 'LINK-USD',
               'ADA-USD', 'DOGE-USD', 'NEAR-USD']

# ── Indicator Calculations ───────────────────────────────────────────────────

def add_indicators(df, params=None):
    """Add all technical indicators needed for intraday strategies."""
    params = params or {}
    df = df.copy()

    # EMA
    for period in [9, 21, 50, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # SMA
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

    # Bollinger Bands (20, 2.0)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    bb_rolling_std = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * bb_rolling_std
    df['bb_lower'] = df['bb_mid'] - bb_std * bb_rolling_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # BB Width percentile (for squeeze detection)
    df['bb_width_pct'] = df['bb_width'].rolling(100).rank(pct=True)

    # Donchian Channels
    for period in [10, 20, 48]:
        df[f'dc_high_{period}'] = df['high'].rolling(period).max()
        df[f'dc_low_{period}'] = df['low'].rolling(period).min()

    # Volume metrics
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20'].replace(0, np.nan)

    # VWAP (rolling — resets every 24 candles for 1h data)
    vwap_period = params.get('vwap_period', 24)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']
    df['vwap'] = tp_vol.rolling(vwap_period).sum() / df['volume'].rolling(vwap_period).sum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100  # % distance from VWAP

    # Momentum
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    df['return_24'] = df['close'].pct_change(24)

    # Price position within range
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

    return df


def add_daily_context(df_1h):
    """Add daily-timeframe context to hourly data for multi-timeframe strategies."""
    df = df_1h.copy()
    df['date'] = df['time'].dt.date

    # Daily aggregates
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
    daily['d_trend'] = (daily['close'] > daily['d_ema_21']).astype(int)  # 1=bullish, 0=bearish

    # Daily ATR
    d_hl = daily['high'] - daily['low']
    d_hc = (daily['high'] - daily['close'].shift(1)).abs()
    d_lc = (daily['low'] - daily['close'].shift(1)).abs()
    d_tr = pd.concat([d_hl, d_hc, d_lc], axis=1).max(axis=1)
    daily['d_atr'] = d_tr.rolling(14).mean()

    # Map back to hourly
    daily_map = daily.set_index('date')[['d_ema_21', 'd_sma_200', 'd_trend', 'd_atr']].to_dict('index')
    df['d_ema_21'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_ema_21'))
    df['d_sma_200'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_sma_200'))
    df['d_trend'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_trend', 0))
    df['d_atr'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_atr'))

    return df


# ── Strategy Implementations ────────────────────────────────────────────────

def strategy_intraday_donchian(df, params):
    """Strategy 1: Intraday Donchian Breakout (long + short)

    Adapts our proven daily system to hourly:
    - Long: close > 48h Donchian high + volume > 1.5x + above EMA(50)
    - Short: close < 48h Donchian low + volume > 1.5x + below EMA(50)
    - Stop: 2x ATR(14)
    - Target: 3x ATR(14) or trailing at 1.5x ATR
    - Max hold: 24 candles (1 day)
    """
    dc_period = params.get('dc_period', 48)  # 48 hours = 2 days
    atr_stop = params.get('atr_stop', 2.0)
    atr_target = params.get('atr_target', 3.0)
    vol_thresh = params.get('vol_threshold', 1.5)
    max_hold = params.get('max_hold', 24)

    signals = []
    for i in range(dc_period + 1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # Long signal
        dc_high = df['high'].iloc[i-dc_period:i].max()
        if (row['close'] > dc_high and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_50']):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

        # Short signal
        dc_low = df['low'].iloc[i-dc_period:i].min()
        if (row['close'] < dc_low and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_50']):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

    return signals


def strategy_vwap_reversion(df, params):
    """Strategy 2: VWAP Mean Reversion

    Fade overextended moves back to VWAP:
    - Long: price < VWAP by >1.5% + RSI < 30 + volume spike (>1.3x)
    - Short: price > VWAP by >1.5% + RSI > 70 + volume spike (>1.3x)
    - Stop: 1.5x ATR beyond entry
    - Target: VWAP (mean reversion)
    - Max hold: 12 candles (12 hours)
    """
    vwap_threshold = params.get('vwap_threshold', 1.5)  # % away from VWAP
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    vol_thresh = params.get('vol_threshold', 1.3)
    atr_stop = params.get('atr_stop', 1.5)
    max_hold = params.get('max_hold', 12)

    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]

        if pd.isna(row['vwap']) or pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # Long (oversold reversion)
        if (row['vwap_dist'] < -vwap_threshold and
            row['rsi'] < rsi_oversold and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['vwap'],  # revert to VWAP
                'atr': row['atr'],
                'max_hold': max_hold,
            })

        # Short (overbought reversion)
        if (row['vwap_dist'] > vwap_threshold and
            row['rsi'] > rsi_overbought and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['vwap'],  # revert to VWAP
                'atr': row['atr'],
                'max_hold': max_hold,
            })

    return signals


def strategy_ema_momentum(df, params):
    """Strategy 3: EMA Momentum Scalp

    Trade momentum after EMA crossovers with volume confirmation:
    - Long: EMA(9) crosses above EMA(21) + close > EMA(50) + vol > 1.5x
    - Short: EMA(9) crosses below EMA(21) + close < EMA(50) + vol > 1.5x
    - Stop: 1.5x ATR
    - Target: 2x ATR (or trailing 1x ATR after 1x ATR profit)
    - Max hold: 8 candles (8 hours)
    """
    vol_thresh = params.get('vol_threshold', 1.5)
    atr_stop = params.get('atr_stop', 1.5)
    atr_target = params.get('atr_target', 2.0)
    max_hold = params.get('max_hold', 8)

    signals = []
    for i in range(52, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # EMA(9) crosses above EMA(21) — bullish
        if (prev['ema_9'] <= prev['ema_21'] and
            row['ema_9'] > row['ema_21'] and
            row['close'] > row['ema_50'] and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

        # EMA(9) crosses below EMA(21) — bearish
        if (prev['ema_9'] >= prev['ema_21'] and
            row['ema_9'] < row['ema_21'] and
            row['close'] < row['ema_50'] and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

    return signals


def strategy_bb_squeeze(df, params):
    """Strategy 4: Bollinger Band Squeeze Breakout

    Trade volatility expansion after compression:
    - Signal: BB width drops below 20th percentile (squeeze)
    - Long: Price breaks above upper BB after squeeze + vol > 1.3x
    - Short: Price breaks below lower BB after squeeze + vol > 1.3x
    - Stop: BB midline (20-SMA)
    - Target: 2.5x ATR
    - Max hold: 16 candles (16 hours)
    """
    squeeze_pct = params.get('squeeze_percentile', 0.20)
    vol_thresh = params.get('vol_threshold', 1.3)
    atr_target = params.get('atr_target', 2.5)
    max_hold = params.get('max_hold', 16)

    signals = []
    in_squeeze = False

    for i in range(101, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row['bb_width_pct']) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # Track squeeze state
        was_squeeze = prev.get('bb_width_pct', 0.5)
        if pd.isna(was_squeeze):
            was_squeeze = 0.5

        if was_squeeze <= squeeze_pct:
            in_squeeze = True

        if not in_squeeze:
            continue

        # Breakout above upper BB
        if (row['close'] > row['bb_upper'] and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'long',
                'entry': row['close'],
                'stop': row['bb_mid'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })
            in_squeeze = False

        # Breakout below lower BB
        elif (row['close'] < row['bb_lower'] and
              row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'short',
                'entry': row['close'],
                'stop': row['bb_mid'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })
            in_squeeze = False

        # Squeeze timeout (if squeeze lasted >10 candles without breakout, reset)
        elif row['bb_width_pct'] > 0.5:
            in_squeeze = False

    return signals


def strategy_mtf_momentum(df, params):
    """Strategy 5: Multi-Timeframe Momentum

    Daily trend bias + hourly entry:
    - Filter: Only trade in direction of daily trend (close > EMA(21) = long bias)
    - Long: Daily bullish + hourly RSI dips to 35-45 + bounces + vol > 1.2x
    - Short: Daily bearish + hourly RSI rises to 55-65 + reverses + vol > 1.2x
    - Stop: 2x ATR
    - Target: 3x ATR
    - Max hold: 24 candles
    """
    vol_thresh = params.get('vol_threshold', 1.2)
    atr_stop = params.get('atr_stop', 2.0)
    atr_target = params.get('atr_target', 3.0)
    max_hold = params.get('max_hold', 24)

    signals = []
    for i in range(52, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row.get('d_trend')) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # Long: daily bullish, RSI dipped then bouncing
        if (row['d_trend'] == 1 and
            prev['rsi'] < 45 and row['rsi'] > prev['rsi'] and
            row['rsi'] > 35 and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_21']):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

        # Short: daily bearish, RSI rose then reversing
        if (row['d_trend'] == 0 and
            prev['rsi'] > 55 and row['rsi'] < prev['rsi'] and
            row['rsi'] < 65 and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_21']):
            signals.append({
                'idx': i,
                'time': row['time'],
                'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'],
                'max_hold': max_hold,
            })

    return signals


# ── Portfolio Backtester ─────────────────────────────────────────────────────

def backtest_strategy(coin_data_processed, strategy_func, strategy_params,
                       starting_capital=STARTING_CAPITAL, max_positions=MAX_POSITIONS,
                       risk_per_trade=RISK_PER_TRADE, leverage=LEVERAGE,
                       fee_rate=FUTURES_FEE, max_daily_loss=MAX_DAILY_LOSS):
    """Run a portfolio-level backtest across all coins.

    Returns dict with trades, equity curve, and statistics.
    """
    equity = starting_capital
    peak_equity = starting_capital
    positions = {}  # {symbol: position_dict}
    trades = []
    equity_curve = []
    daily_pnl = defaultdict(float)

    # Gather all signals from all coins
    all_signals = []
    for symbol, df in coin_data_processed.items():
        if symbol not in TRADE_COINS:
            continue
        signals = strategy_func(df, strategy_params)
        for s in signals:
            s['symbol'] = symbol
        all_signals.extend(signals)

    # Sort by time
    all_signals.sort(key=lambda s: s['time'])

    if not all_signals:
        return {'trades': [], 'equity_curve': [], 'stats': None}

    # Build price lookups for position management
    price_lookup = {}
    for symbol, df in coin_data_processed.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time']] = row
        price_lookup[symbol] = lookup

    # Get all unique timestamps across all coins
    all_times = set()
    for symbol, df in coin_data_processed.items():
        if symbol in TRADE_COINS:
            for t in df['time']:
                all_times.add(t)
    all_times = sorted(all_times)

    # Build signal lookup by time
    signal_by_time = defaultdict(list)
    for s in all_signals:
        signal_by_time[s['time']].append(s)

    # Day tracking for daily loss limit
    current_day = None
    day_pnl = 0.0
    day_locked = False

    # ── Main loop ─────────────────────────────────────────────
    for t in all_times:
        day = t.date() if hasattr(t, 'date') else t

        # Reset daily tracking
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            day_locked = False

        # Check daily loss limit
        if day_locked:
            continue

        # ── Check exits for open positions ────────────────────
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
                # Stop loss hit
                if low <= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                # Target hit
                elif high >= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                # Trailing stop update
                elif close_price > pos.get('high_watermark', pos['entry']):
                    pos['high_watermark'] = close_price
                    new_stop = close_price - 1.5 * pos['atr']
                    if new_stop > pos['stop']:
                        pos['stop'] = new_stop
                # Max hold
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price

            elif pos['side'] == 'short':
                # Stop loss hit
                if high >= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                # Target hit
                elif low <= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                # Trailing stop update
                elif close_price < pos.get('low_watermark', pos['entry']):
                    pos['low_watermark'] = close_price
                    new_stop = close_price + 1.5 * pos['atr']
                    if new_stop < pos['stop']:
                        pos['stop'] = new_stop
                # Max hold
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price

            if exit_reason:
                to_close.append((sym, pos, exit_price, exit_reason))

        # Process exits
        for sym, pos, exit_price, exit_reason in to_close:
            if pos['side'] == 'long':
                gross_pnl = (exit_price - pos['entry']) / pos['entry']
            else:
                gross_pnl = (pos['entry'] - exit_price) / pos['entry']

            # Apply leverage and fees
            leveraged_pnl = gross_pnl * leverage
            fee_cost = 2 * fee_rate  # entry + exit
            net_pnl_pct = leveraged_pnl - fee_cost
            net_pnl_dollar = pos['size'] * net_pnl_pct

            equity += net_pnl_dollar
            day_pnl += net_pnl_dollar

            if equity > peak_equity:
                peak_equity = equity

            trades.append({
                'symbol': sym,
                'side': pos['side'],
                'entry_time': pos['entry_time'],
                'exit_time': t,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'hold_candles': pos['hold_candles'],
                'gross_pnl_pct': gross_pnl * 100,
                'net_pnl_pct': net_pnl_pct * 100,
                'net_pnl_dollar': net_pnl_dollar,
                'exit_reason': exit_reason,
                'equity_after': equity,
            })

            del positions[sym]

            # Check daily loss limit
            if day_pnl / starting_capital < -max_daily_loss:
                day_locked = True
                break

        if day_locked:
            equity_curve.append({'time': t, 'equity': equity})
            continue

        # ── Check entries ─────────────────────────────────────
        if t in signal_by_time:
            for sig in signal_by_time[t]:
                sym = sig['symbol']

                # Skip if already in position for this coin
                if sym in positions:
                    continue

                # Skip if max positions reached
                if len(positions) >= max_positions:
                    break

                # Skip if equity too low
                if equity < starting_capital * 0.5:
                    break

                # Position sizing: risk-based
                stop_distance = abs(sig['entry'] - sig['stop']) / sig['entry']
                if stop_distance <= 0:
                    continue

                risk_amount = equity * risk_per_trade
                position_size = risk_amount / stop_distance
                position_size = min(position_size, equity * 0.95 / leverage)  # cap at 95% of equity per side

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

                # Deduct entry fee
                entry_fee = position_size * fee_rate
                equity -= entry_fee

        equity_curve.append({'time': t, 'equity': equity})

    # Close any remaining positions at last price
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
            'symbol': sym,
            'side': pos['side'],
            'entry_time': pos['entry_time'],
            'exit_time': last_df.iloc[-1]['time'],
            'entry_price': pos['entry'],
            'exit_price': exit_price,
            'hold_candles': pos['hold_candles'],
            'gross_pnl_pct': gross_pnl * 100,
            'net_pnl_pct': net_pnl_pct * 100,
            'net_pnl_dollar': net_pnl_dollar,
            'exit_reason': 'end_of_data',
            'equity_after': equity,
        })

    # Compute statistics
    stats = compute_stats(trades, equity, starting_capital)

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'stats': stats,
    }


# ── Statistics ───────────────────────────────────────────────────────────────

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

    # Max drawdown from equity curve in trades
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

    # Long vs short breakdown
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


# ── Reporting ────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_strategy_results(name, result):
    """Print detailed results for one strategy."""
    stats = result['stats']
    if not stats:
        print(f"\n  {name}: NO TRADES generated")
        return

    trades = result['trades']

    # Verdict
    passed = (stats['win_rate'] >= 55 and stats['profit_factor'] >= 1.5) or \
             (stats['profit_factor'] >= 1.8 and stats['n_trades'] >= 20)
    verdict = "PASS" if passed else "FAIL"
    verdict_color = "\033[92m" if passed else "\033[91m"

    print(f"\n  [{verdict_color}{verdict}\033[0m] {name}")
    print(f"  {'─'*60}")
    print(f"  Trades: {stats['n_trades']}  |  Trades/day: {stats['trades_per_day']:.1f}  |  "
          f"Days traded: {stats['total_days']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%  |  PF: {stats['profit_factor']:.2f}  |  "
          f"Sharpe: {stats['sharpe']:.2f}")
    print(f"  Net Return: {stats['net_return_pct']:+.1f}% (${stats['net_return_dollar']:+,.0f})")
    print(f"  Max Drawdown: {stats['max_drawdown']:.1f}%")
    print(f"  Avg Win: {stats['avg_win_pct']:+.2f}%  |  Avg Loss: {stats['avg_loss_pct']:.2f}%  |  "
          f"Avg Hold: {stats['avg_hold_candles']:.1f}h")
    print(f"  Longs: {stats['n_long']} ({stats['long_wr']:.0f}% WR)  |  "
          f"Shorts: {stats['n_short']} ({stats['short_wr']:.0f}% WR)")
    print(f"  Profitable Days: {stats['profitable_days_pct']:.1f}%  |  "
          f"Avg Daily P&L: ${stats['avg_daily_pnl']:+.0f}")
    print(f"  Exits: {stats['exit_counts']}")

    # Top coins
    coin_pnl = defaultdict(float)
    coin_trades = defaultdict(int)
    for t in trades:
        coin_pnl[t['symbol']] += t['net_pnl_dollar']
        coin_trades[t['symbol']] += 1

    print(f"\n  Per-Coin P&L:")
    for sym in sorted(coin_pnl.keys(), key=lambda s: coin_pnl[s], reverse=True):
        print(f"    {sym:10s}: ${coin_pnl[sym]:+8,.0f} ({coin_trades[sym]} trades)")

    # Monthly returns
    monthly = defaultdict(float)
    for t in trades:
        m = t['exit_time'].strftime('%Y-%m') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[:7]
        monthly[m] += t['net_pnl_dollar']

    print(f"\n  Monthly P&L:")
    for m in sorted(monthly.keys()):
        bar = '+' * max(0, int(monthly[m] / 50)) if monthly[m] > 0 else '-' * max(0, int(-monthly[m] / 50))
        print(f"    {m}: ${monthly[m]:+8,.0f}  {bar}")

    return passed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("INTRADAY STRATEGY BACKTESTER")
    print(f"  Starting capital: ${STARTING_CAPITAL:,}")
    print(f"  Leverage: {LEVERAGE}x  |  Fee: {FUTURES_FEE*100:.2f}%/side  |  "
          f"Max positions: {MAX_POSITIONS}")
    print(f"  Risk/trade: {RISK_PER_TRADE*100:.1f}%  |  Max daily loss: {MAX_DAILY_LOSS*100:.0f}%")
    print(f"  Coins: {', '.join(TRADE_COINS)}")

    # ── Fetch data ────────────────────────────────────────────
    print_section("LOADING DATA")
    coin_data = fetch_all_intraday(timeframe='1h', months=6)

    # ── Add indicators ────────────────────────────────────────
    print_section("COMPUTING INDICATORS")
    coin_data_processed = {}
    for symbol, df in coin_data.items():
        print(f"  {symbol}: adding indicators...", end='')
        df = add_indicators(df)
        df = add_daily_context(df)
        coin_data_processed[symbol] = df
        print(f" done ({len(df):,} candles)")

    # ── Strategy definitions ──────────────────────────────────
    strategies = {
        'Intraday Donchian (48h)': {
            'func': strategy_intraday_donchian,
            'params': {'dc_period': 48, 'atr_stop': 2.0, 'atr_target': 3.0,
                       'vol_threshold': 1.5, 'max_hold': 24},
        },
        'VWAP Mean Reversion': {
            'func': strategy_vwap_reversion,
            'params': {'vwap_threshold': 1.5, 'rsi_oversold': 30, 'rsi_overbought': 70,
                       'vol_threshold': 1.3, 'atr_stop': 1.5, 'max_hold': 12},
        },
        'EMA Momentum Scalp': {
            'func': strategy_ema_momentum,
            'params': {'vol_threshold': 1.5, 'atr_stop': 1.5, 'atr_target': 2.0,
                       'max_hold': 8},
        },
        'BB Squeeze Breakout': {
            'func': strategy_bb_squeeze,
            'params': {'squeeze_percentile': 0.20, 'vol_threshold': 1.3,
                       'atr_target': 2.5, 'max_hold': 16},
        },
        'Multi-TF Momentum': {
            'func': strategy_mtf_momentum,
            'params': {'vol_threshold': 1.2, 'atr_stop': 2.0, 'atr_target': 3.0,
                       'max_hold': 24},
        },
    }

    # ── Run backtests ─────────────────────────────────────────
    results = {}
    for name, config in strategies.items():
        print_section(f"BACKTESTING: {name}")
        result = backtest_strategy(
            coin_data_processed,
            config['func'],
            config['params'],
        )
        results[name] = result
        print_strategy_results(name, result)

    # ── Parameter sweep for best strategies ───────────────────
    print_section("PARAMETER SENSITIVITY (Donchian)")
    donchian_variants = [
        ('DC-24h / 1.5A / 2.5T', {'dc_period': 24, 'atr_stop': 1.5, 'atr_target': 2.5, 'vol_threshold': 1.3, 'max_hold': 16}),
        ('DC-48h / 1.5A / 2.0T', {'dc_period': 48, 'atr_stop': 1.5, 'atr_target': 2.0, 'vol_threshold': 1.5, 'max_hold': 24}),
        ('DC-48h / 2.0A / 4.0T', {'dc_period': 48, 'atr_stop': 2.0, 'atr_target': 4.0, 'vol_threshold': 1.5, 'max_hold': 36}),
        ('DC-72h / 2.0A / 3.0T', {'dc_period': 72, 'atr_stop': 2.0, 'atr_target': 3.0, 'vol_threshold': 1.5, 'max_hold': 36}),
        ('DC-48h / 1.5A / 3.0T (loose vol)', {'dc_period': 48, 'atr_stop': 1.5, 'atr_target': 3.0, 'vol_threshold': 1.2, 'max_hold': 24}),
    ]

    print(f"\n  {'Variant':<35s} {'Trades':>6s} {'WR':>7s} {'PF':>6s} {'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*78}")

    for vname, vparams in donchian_variants:
        r = backtest_strategy(coin_data_processed, strategy_intraday_donchian, vparams)
        s = r['stats']
        if s:
            daily = s['avg_daily_pnl']
            print(f"  {vname:<35s} {s['n_trades']:>6d} {s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% ${daily:>+7.0f}")
        else:
            print(f"  {vname:<35s}   — no trades —")

    # ── VWAP parameter sweep ──────────────────────────────────
    print_section("PARAMETER SENSITIVITY (VWAP)")
    vwap_variants = [
        ('VWAP 1.0% / RSI 35-65', {'vwap_threshold': 1.0, 'rsi_oversold': 35, 'rsi_overbought': 65, 'vol_threshold': 1.2, 'atr_stop': 1.5, 'max_hold': 12}),
        ('VWAP 1.5% / RSI 25-75', {'vwap_threshold': 1.5, 'rsi_oversold': 25, 'rsi_overbought': 75, 'vol_threshold': 1.3, 'atr_stop': 1.5, 'max_hold': 12}),
        ('VWAP 2.0% / RSI 30-70', {'vwap_threshold': 2.0, 'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_threshold': 1.3, 'atr_stop': 2.0, 'max_hold': 16}),
        ('VWAP 1.5% / no RSI', {'vwap_threshold': 1.5, 'rsi_oversold': 50, 'rsi_overbought': 50, 'vol_threshold': 1.5, 'atr_stop': 1.5, 'max_hold': 8}),
    ]

    print(f"\n  {'Variant':<35s} {'Trades':>6s} {'WR':>7s} {'PF':>6s} {'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*78}")

    for vname, vparams in vwap_variants:
        r = backtest_strategy(coin_data_processed, strategy_vwap_reversion, vparams)
        s = r['stats']
        if s:
            daily = s['avg_daily_pnl']
            print(f"  {vname:<35s} {s['n_trades']:>6d} {s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% ${daily:>+7.0f}")
        else:
            print(f"  {vname:<35s}   — no trades —")

    # ── Summary ───────────────────────────────────────────────
    print_section("STRATEGY COMPARISON")
    print(f"\n  {'Strategy':<30s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s} {'Verdict':>8s}")
    print(f"  {'─'*90}")

    for name, result in results.items():
        s = result['stats']
        if s:
            passed = (s['win_rate'] >= 55 and s['profit_factor'] >= 1.5) or \
                     (s['profit_factor'] >= 1.8 and s['n_trades'] >= 20)
            v = "PASS" if passed else "FAIL"
            print(f"  {name:<30s} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} {s['win_rate']:>6.1f}% "
                  f"{s['profit_factor']:>5.2f} {s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}  {v}")
        else:
            print(f"  {name:<30s}   — no trades —")

    # ── Path to $500/day projection ───────────────────────────
    print_section("PATH TO $500/DAY")
    best_name = None
    best_daily = 0
    for name, result in results.items():
        s = result['stats']
        if s and s['avg_daily_pnl'] > best_daily:
            best_daily = s['avg_daily_pnl']
            best_name = name

    if best_name:
        best = results[best_name]['stats']
        print(f"\n  Best strategy: {best_name}")
        print(f"  Avg daily P&L on $10K: ${best['avg_daily_pnl']:+.0f}")
        print(f"  Trades/day: {best['trades_per_day']:.1f}")

        if best['avg_daily_pnl'] > 0:
            days_to_30k = 0
            eq = STARTING_CAPITAL
            while eq < 30000 and days_to_30k < 365:
                eq += best['avg_daily_pnl'] * (eq / STARTING_CAPITAL)  # scale with equity
                days_to_30k += 1

            print(f"\n  Growth projection (compounding):")
            print(f"    $10K → $30K: ~{days_to_30k} days")
            print(f"    At $30K, daily P&L: ~${best['avg_daily_pnl'] * 3:+.0f}")
            print(f"    At $20K (after safenet), daily P&L: ~${best['avg_daily_pnl'] * 2:+.0f}")

            target = 500
            capital_needed = target / (best['avg_daily_pnl'] / STARTING_CAPITAL)
            print(f"\n  Capital needed for ${target}/day: ~${capital_needed:,.0f}")
        else:
            print(f"\n  WARNING: Best strategy has negative daily P&L. Need parameter tuning.")
    else:
        print(f"\n  No strategies produced trades. Check data and parameters.")

    print(f"\n{'='*80}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
