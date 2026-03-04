"""Multi-Signal Short Backtest — Testing 4 Alternative Short Entry Signals

Tests 4 new short entry signals individually and combined with Donchian breakdowns:
  1. MACD Bearish Cross — momentum shift, highest frequency
  2. Volume Exhaustion Reversal — fading rally volume + bearish candle
  3. Bearish Engulfing at Resistance — candlestick pattern at key levels
  4. RSI Bearish Divergence — higher high + lower RSI

All use SMA(200) bear filter (not death cross) for 2x more eligible days.
All share the same exit mechanics as Donchian shorts (trailing stop, high exit, max hold).

Run: python backtest_short_signals.py
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_donchian_daily import (
    fetch_all_coins,
    calculate_indicators,
    compute_stats,
    compute_per_coin_stats,
    DEFAULT_PARAMS,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_phase3 import backtest_portfolio_phase3
from backtest_shorts import (
    SHORT_COINS,
    SHORT_DEFAULT_PARAMS,
    LONG_PROD_PARAMS,
    calculate_short_indicators,
    compute_btc_bear_filter,
    backtest_portfolio_short,
)
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_all_indicators(df, params):
    """Calculate all indicators needed for the 4 signal types + Donchian."""
    df = calculate_short_indicators(df, params)

    # SMA 50/200 (for bear filter and resistance)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # MACD
    fast = params.get('macd_fast', 12)
    slow = params.get('macd_slow', 26)
    sig = params.get('macd_signal_period', 9)
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal_line'] = df['macd'].ewm(span=sig, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal_line']

    return df


# ============================================================================
# SIGNAL DETECTION FUNCTIONS
# ============================================================================

def check_macd_cross(row, prev_row, params):
    """Signal 1: MACD crosses below signal line."""
    if pd.isna(row.get('macd')) or pd.isna(prev_row.get('macd')):
        return False

    # Cross: was above/equal, now below
    cross = (float(prev_row['macd']) >= float(prev_row['macd_signal_line']) and
             float(row['macd']) < float(row['macd_signal_line']))
    if not cross:
        return False

    # Histogram negative
    if float(row['macd_hist']) >= 0:
        return False

    # MACD below zero threshold (normalized to % of price)
    macd_pct = (float(row['macd']) / float(row['close'])) * 100
    threshold = params.get('macd_zero_threshold', 0)
    if macd_pct > threshold:
        return False

    # Price below EMA(21)
    if params.get('require_ema_below', True):
        if float(row['close']) > float(row['ema_21']):
            return False

    # Optional volume filter
    vol_mult = params.get('macd_vol_mult', 0)
    if vol_mult > 0:
        vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
        if float(row['volume']) < vol_mult * vol_sma:
            return False

    return True


def check_volume_exhaustion(row, df_window, params):
    """Signal 2: Volume fading during rally + bearish candle.

    df_window: last N rows of dataframe ending at current row.
    """
    lookback = params.get('rally_lookback', 5)
    if len(df_window) < lookback + 1:
        return False

    current = df_window.iloc[-1]
    rally_start = df_window.iloc[-(lookback + 1)]

    # Rally: price rose over lookback
    rally_pct = (float(current['close']) - float(rally_start['close'])) / float(rally_start['close']) * 100
    min_rally = params.get('rally_min_pct', 3.0)
    if rally_pct < min_rally:
        return False

    # Volume fade: today's volume below threshold
    vol_sma = float(current['volume_sma']) if current['volume_sma'] > 0 else 1
    fade_mult = params.get('volume_fade_mult', 0.8)
    if float(current['volume']) > fade_mult * vol_sma:
        return False

    # Bearish candle
    if float(current['close']) >= float(current['open']):
        return False

    # Close in bottom portion of range
    candle_range = float(current['high']) - float(current['low'])
    if candle_range == 0:
        return False
    body_pct = params.get('candle_body_pct', 0.33)
    close_position = (float(current['close']) - float(current['low'])) / candle_range
    if close_position > body_pct:
        return False

    # Trend alignment: below EMA(21)
    if float(current['close']) > float(current['ema_21']):
        return False

    return True


def check_bearish_engulfing(row, prev_row, df_window, params):
    """Signal 3: Bearish engulfing at resistance."""
    # Engulfing pattern
    prev_green = float(prev_row['close']) > float(prev_row['open'])
    today_red = float(row['close']) < float(row['open'])
    engulfs = (float(row['open']) > float(prev_row['close']) and
               float(row['close']) < float(prev_row['open']))

    if not (prev_green and today_red and engulfs):
        return False

    # Minimum body size
    body_size = float(row['open']) - float(row['close'])
    min_body = params.get('min_body_atr', 0.5) * float(row['atr'])
    if body_size < min_body:
        return False

    # Resistance confluence (at least 1)
    tolerance = params.get('resistance_tolerance', 1.5) / 100
    at_resistance = False

    # SMA(50) resistance
    sma50 = row.get('sma_50')
    if sma50 is not None and not pd.isna(sma50):
        if float(row['high']) >= float(sma50) and float(row['close']) < float(sma50):
            at_resistance = True

    # EMA(21) resistance
    if not at_resistance:
        if float(row['high']) >= float(row['ema_21']) and float(row['close']) < float(row['ema_21']):
            at_resistance = True

    # Prior swing high resistance
    if not at_resistance:
        offset = params.get('swing_offset', 3)
        lookback = params.get('swing_lookback', 20)
        if len(df_window) >= lookback + offset:
            swing_section = df_window.iloc[-(lookback + offset):-(offset)]
            if len(swing_section) > 0:
                swing_high = swing_section['high'].max()
                if abs(float(row['high']) - float(swing_high)) / float(swing_high) <= tolerance:
                    at_resistance = True

    return at_resistance


def check_rsi_divergence(row, df_window, params):
    """Signal 4: Price higher high but RSI lower high (bearish divergence)."""
    lookback = params.get('div_lookback', 14)
    offset = params.get('div_offset', 3)

    if len(df_window) < lookback + offset + 1:
        return False

    current = df_window.iloc[-1]

    # RSI range filter
    rsi_val = float(current['rsi']) if not pd.isna(current['rsi']) else 50
    rsi_min = params.get('rsi_min', 30)
    rsi_max = params.get('rsi_max', 65)
    if rsi_val < rsi_min or rsi_val > rsi_max:
        return False

    # Find prior swing high
    search_section = df_window.iloc[-(lookback + offset + 1):-(offset)]
    if len(search_section) < 3:
        return False

    prior_high_idx = search_section['high'].astype(float).idxmax()
    prior_high_price = float(search_section.loc[prior_high_idx, 'high'])
    prior_high_rsi = float(search_section.loc[prior_high_idx, 'rsi']) if not pd.isna(search_section.loc[prior_high_idx, 'rsi']) else 50

    # Current region high
    recent = df_window.iloc[-(offset + 1):]
    current_high_price = float(recent['high'].max())
    current_high_idx = recent['high'].astype(float).idxmax()
    current_high_rsi = float(df_window.loc[current_high_idx, 'rsi']) if not pd.isna(df_window.loc[current_high_idx, 'rsi']) else 50

    # Price higher high
    if current_high_price <= prior_high_price:
        return False

    # RSI lower high (bearish divergence)
    rsi_diff = prior_high_rsi - current_high_rsi
    if rsi_diff < params.get('rsi_diff_min', 5.0):
        return False

    # Bearish candle confirmation
    if params.get('require_red_candle', True):
        if float(current['close']) >= float(current['open']):
            return False

    # Trend: below SMA50
    sma50 = current.get('sma_50')
    if sma50 is not None and not pd.isna(sma50):
        if float(current['close']) > float(sma50):
            return False

    return True


# ============================================================================
# MULTI-SIGNAL SHORT BACKTEST
# ============================================================================

def backtest_multi_signal_short(coin_data, params, bear_filter,
                                 signals_enabled=None):
    """Portfolio backtest with multiple short entry signal types.

    signals_enabled: list of signal names to use. Default: all.
        Options: 'donchian', 'macd', 'volume_exhaustion', 'bearish_engulfing', 'rsi_divergence'
    """
    if signals_enabled is None:
        signals_enabled = ['donchian', 'macd', 'volume_exhaustion',
                           'bearish_engulfing', 'rsi_divergence']

    label = params.get('label', 'Multi-Signal')
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']
    funding_daily = params.get('funding_rate_daily', 0) / 100

    # Pre-calculate indicators
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_all_indicators(df, params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'donchian_low', 'atr',
                                        'volume_sma', 'ema_21', 'rsi', 'exit_high'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 50:
            prepared[symbol] = df_ind

    # Build timelines and lookups
    all_dates = set()
    for symbol, df in prepared.items():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup

    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j - 1])
        prev_lookups[symbol] = prev_lookup

    # Index lookups for window access
    idx_lookups = {}
    for symbol, df in prepared.items():
        idx_map = {}
        for i, row in df.iterrows():
            idx_map[row['time'].date()] = i
        idx_lookups[symbol] = idx_map

    # State
    positions = {}
    trades = []
    equity_curve = []
    signal_counts = {s: 0 for s in signals_enabled}

    for date in all_dates:
        is_bear = bear_filter.get(date, False)

        # === EXITS ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_low = float(row['low'])
            current_atr = float(row['atr'])

            if current_low < pos['low_watermark']:
                pos['low_watermark'] = current_low

            pos['hold_days'] += 1
            exit_reason = None
            exit_price = None

            # Trailing stop
            trailing_stop = pos['low_watermark'] + params['atr_mult'] * current_atr
            if current_close > trailing_stop:
                exit_reason = 'Trailing stop (ATR)'
                exit_price = trailing_stop

            # N-day high exit
            if not exit_reason and not pd.isna(row.get('exit_high')):
                if current_close > float(row['exit_high']):
                    exit_reason = f'{params["exit_period"]}-day high exit'
                    exit_price = current_close

            # Emergency stop
            if not exit_reason:
                loss_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if loss_pct >= params['emergency_stop_pct']:
                    exit_reason = f'Emergency stop ({loss_pct:.1f}%)'
                    exit_price = current_close

            # Max hold
            if not exit_reason and pos['hold_days'] >= params['max_hold_days']:
                exit_reason = f'Max hold ({params["max_hold_days"]}d)'
                exit_price = current_close

            # Blow-off (oversold bounce risk)
            if not exit_reason:
                rsi_val = float(row['rsi']) if not pd.isna(row['rsi']) else 50
                vol_val = float(row['volume']) if not pd.isna(row['volume']) else 0
                vol_sma = float(row['volume_sma']) if not pd.isna(row['volume_sma']) and row['volume_sma'] > 0 else 1
                if rsi_val < params.get('rsi_blowoff', 20) and vol_val > params.get('volume_blowoff', 3.0) * vol_sma:
                    tight_stop = pos['low_watermark'] + params.get('atr_mult_tight', 1.0) * current_atr
                    if current_close > tight_stop:
                        exit_reason = 'Blow-off tightened stop'
                        exit_price = tight_stop

            # Partial TPs
            if not exit_reason:
                gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                    fraction = params['tp1_fraction']
                    partial_size = pos['size_usd'] * fraction
                    exit_price_adj = current_close * (1 + cost_per_side)
                    price_pnl = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
                    fund_pnl = funding_daily * pos['hold_days'] * 100
                    pnl_pct = price_pnl + fund_pnl
                    pnl_usd = partial_size * (pnl_pct / 100)
                    capital += partial_size + pnl_usd
                    pos['size_usd'] -= partial_size
                    pos['remaining_fraction'] -= fraction
                    pos['partials_taken'] = 1
                    trades.append({
                        'symbol': symbol, 'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'], 'exit_time': row['time'],
                        'exit_price': exit_price_adj, 'pnl_pct': pnl_pct,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP1 ({fraction*100:.0f}%)',
                        'size_usd': partial_size, 'side': 'SHORT',
                        'win': pnl_pct > 0, 'entry_type': pos.get('entry_type', 'unknown'),
                    })
                elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                    fraction = params['tp2_fraction']
                    partial_size = pos['size_usd'] * (fraction / pos['remaining_fraction'])
                    exit_price_adj = current_close * (1 + cost_per_side)
                    price_pnl = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
                    fund_pnl = funding_daily * pos['hold_days'] * 100
                    pnl_pct = price_pnl + fund_pnl
                    pnl_usd = partial_size * (pnl_pct / 100)
                    capital += partial_size + pnl_usd
                    pos['size_usd'] -= partial_size
                    pos['remaining_fraction'] -= fraction
                    pos['partials_taken'] = 2
                    trades.append({
                        'symbol': symbol, 'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'], 'exit_time': row['time'],
                        'exit_price': exit_price_adj, 'pnl_pct': pnl_pct,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP2 ({fraction*100:.0f}%)',
                        'size_usd': partial_size, 'side': 'SHORT',
                        'win': pnl_pct > 0, 'entry_type': pos.get('entry_type', 'unknown'),
                    })

            if exit_reason:
                exit_price_adj = (exit_price or current_close) * (1 + cost_per_side)
                price_pnl_pct = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
                funding_pnl_pct = funding_daily * pos['hold_days'] * 100
                total_pnl_pct = price_pnl_pct + funding_pnl_pct
                pnl_usd = pos['size_usd'] * (total_pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol, 'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'], 'exit_time': row['time'],
                    'exit_price': exit_price_adj, 'pnl_pct': total_pnl_pct,
                    'hold_days': pos['hold_days'], 'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'], 'side': 'SHORT',
                    'win': total_pnl_pct > 0,
                    'entry_type': pos.get('entry_type', 'unknown'),
                })
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del positions[s]

        # === NEW SHORT ENTRIES ===
        if len(positions) >= max_positions or not is_bear:
            # Equity tracking
            total_equity = capital
            for sym, pos in positions.items():
                r = lookups[sym].get(date)
                if r is not None:
                    pc = (pos['entry_price'] - float(r['close'])) / pos['entry_price']
                    total_equity += max(pos['size_usd'] * (1 + pc), 0)
                else:
                    total_equity += pos['size_usd']
            equity_curve.append({'date': date, 'equity': total_equity})
            continue

        for symbol in prepared:
            if symbol in positions or len(positions) >= max_positions:
                continue

            row = lookups[symbol].get(date)
            prev_row = prev_lookups[symbol].get(date)
            if row is None:
                continue

            df_sym = prepared[symbol]
            row_idx = idx_lookups[symbol].get(date)
            if row_idx is None:
                continue

            # Get a window for signals that need history
            window_size = max(params.get('div_lookback', 14) + params.get('div_offset', 3) + 5,
                              params.get('rally_lookback', 5) + 5,
                              params.get('swing_lookback', 20) + params.get('swing_offset', 3) + 5)
            win_start = max(0, row_idx - window_size)
            df_window = df_sym.iloc[win_start:row_idx + 1]

            # Try each signal type (priority order)
            entry_type = None
            custom_stop = None

            # 1. Donchian breakdown
            if 'donchian' in signals_enabled and prev_row is not None and not pd.isna(prev_row.get('donchian_low')):
                current_close = float(row['close'])
                breakdown = current_close < float(prev_row['donchian_low'])
                dc_vol = params.get('volume_mult', 1.5)
                if dc_vol > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > dc_vol * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close < float(row['ema_21'])
                if breakdown and volume_ok and trend_ok:
                    entry_type = 'donchian'

            # 2. MACD cross
            if entry_type is None and 'macd' in signals_enabled and prev_row is not None:
                if check_macd_cross(row, prev_row, params):
                    entry_type = 'macd'
                    # Stop above recent swing high
                    sl_lookback = params.get('macd_sl_lookback', 10)
                    sl_start = max(0, row_idx - sl_lookback)
                    swing_high = float(df_sym.iloc[sl_start:row_idx + 1]['high'].max())
                    custom_stop = swing_high + params.get('macd_sl_atr_mult', 1.5) * float(row['atr'])

            # 3. Volume exhaustion
            if entry_type is None and 'volume_exhaustion' in signals_enabled:
                if check_volume_exhaustion(row, df_window, params):
                    entry_type = 'volume_exhaustion'
                    # Stop above rally swing high
                    lookback = params.get('rally_lookback', 5)
                    sl_start = max(0, row_idx - lookback)
                    swing_high = float(df_sym.iloc[sl_start:row_idx + 1]['high'].max())
                    custom_stop = swing_high + params.get('ver_sl_atr_mult', 1.5) * float(row['atr'])

            # 4. Bearish engulfing
            if entry_type is None and 'bearish_engulfing' in signals_enabled and prev_row is not None:
                if check_bearish_engulfing(row, prev_row, df_window, params):
                    entry_type = 'bearish_engulfing'
                    # Stop above engulfing candle high
                    custom_stop = float(row['high']) + params.get('engulf_sl_atr_mult', 1.0) * float(row['atr'])

            # 5. RSI divergence
            if entry_type is None and 'rsi_divergence' in signals_enabled:
                if check_rsi_divergence(row, df_window, params):
                    entry_type = 'rsi_divergence'
                    # Stop above divergence window high
                    lookback = params.get('div_lookback', 14)
                    sl_start = max(0, row_idx - lookback)
                    swing_high = float(df_sym.iloc[sl_start:row_idx + 1]['high'].max())
                    custom_stop = swing_high + params.get('div_sl_atr_mult', 1.5) * float(row['atr'])

            if entry_type is None:
                continue

            # Position sizing
            current_close = float(row['close'])
            total_equity = capital + sum(
                p['size_usd'] * (1 + (p['entry_price'] - float(lookups[s].get(date, row)['close'])) / p['entry_price'])
                if lookups.get(s, {}).get(date) is not None else p['size_usd']
                for s, p in positions.items()
            )
            risk_amount = total_equity * risk_pct
            entry_price = current_close * (1 - cost_per_side)

            if custom_stop is not None:
                stop_distance = custom_stop - entry_price
            else:
                stop_distance = params['atr_mult'] * float(row['atr'])

            stop_pct = stop_distance / entry_price if entry_price > 0 else 0
            if stop_pct > 0:
                position_size = risk_amount / stop_pct
            else:
                position_size = total_equity / max_positions

            position_size = min(position_size, capital * 0.95)
            if position_size < 100:
                continue

            capital -= position_size
            positions[symbol] = {
                'entry_price': entry_price,
                'entry_time': row['time'],
                'low_watermark': float(row['low']),
                'partials_taken': 0,
                'remaining_fraction': 1.0,
                'size_usd': position_size,
                'pyramided': False,
                'hold_days': 0,
                'entry_type': entry_type,
            }
            signal_counts[entry_type] = signal_counts.get(entry_type, 0) + 1

        # Equity tracking
        total_equity = capital
        for sym, pos in positions.items():
            r = lookups[sym].get(date)
            if r is not None:
                pc = (pos['entry_price'] - float(r['close'])) / pos['entry_price']
                total_equity += max(pos['size_usd'] * (1 + pc), 0)
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        exit_price_adj = float(last['close']) * (1 + cost_per_side)
        price_pnl_pct = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
        funding_pnl_pct = funding_daily * pos['hold_days'] * 100
        total_pnl_pct = price_pnl_pct + funding_pnl_pct
        pnl_usd = pos['size_usd'] * (total_pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'], 'exit_time': last['time'],
            'exit_price': exit_price_adj, 'pnl_pct': total_pnl_pct,
            'hold_days': pos['hold_days'], 'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'], 'side': 'SHORT',
            'win': total_pnl_pct > 0, 'entry_type': pos.get('entry_type', 'unknown'),
        })

    return trades, equity_curve, capital, signal_counts


def print_entry_breakdown(trades):
    """Print win/loss stats by entry type."""
    full_exits = [t for t in trades if 'Partial' not in t.get('exit_reason', '')]
    by_type = {}
    for t in full_exits:
        et = t.get('entry_type', 'unknown')
        if et not in by_type:
            by_type[et] = []
        by_type[et].append(t)

    print(f"\n  {'Entry Type':<22s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'PF':>6s} {'Sum P&L':>10s} {'Avg Hold':>9s}")
    for et in sorted(by_type.keys()):
        ts = by_type[et]
        wins = [t for t in ts if t.get('win')]
        losses = [t for t in ts if not t.get('win')]
        gross_wins = sum(t['pnl_pct'] for t in wins) if wins else 0
        gross_losses = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0
        pf = gross_wins / gross_losses if gross_losses > 0 else (99.0 if gross_wins > 0 else 0)
        wr = len(wins) / len(ts) * 100 if ts else 0
        pnl = sum(t['pnl_pct'] for t in ts)
        avg_hold = sum(t['hold_days'] for t in ts) / len(ts) if ts else 0
        print(f"  {et:<22s} {len(ts):>6d} {len(wins):>5d} {wr:>5.1f}% {pf:>5.2f} {pnl:>+9.1f}% {avg_hold:>7.1f}d")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("MULTI-SIGNAL SHORT BACKTEST")
    print("  Testing: Donchian + MACD Cross + Volume Exhaustion + Bearish Engulfing + RSI Divergence")
    print(f"  Coins: {', '.join(c.replace('-USD','') for c in SHORT_COINS)}")
    print("=" * 100)

    # Fetch data
    print("\n  Fetching historical data (4 years)...")
    coin_data = fetch_all_coins(coins=SHORT_COINS, years=4)
    btc_df = coin_data.get('BTC-USD')
    if btc_df is None:
        print("ERROR: No BTC data")
        return

    # Compute filters
    print("\n  Computing filters...")
    bear_simple = compute_btc_bear_filter(btc_df, require_death_cross=False)
    bear_strict = compute_btc_bear_filter(btc_df, require_death_cross=True)
    bull_filter = compute_btc_bull_filter(btc_df)

    # Base params
    base_params = {
        **SHORT_DEFAULT_PARAMS,
        # MACD params
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal_period': 9,
        'macd_zero_threshold': 0, 'require_ema_below': True, 'macd_vol_mult': 0,
        'macd_sl_lookback': 10, 'macd_sl_atr_mult': 1.5,
        # Volume exhaustion params
        'rally_lookback': 5, 'rally_min_pct': 3.0,
        'volume_fade_mult': 0.8, 'candle_body_pct': 0.33,
        'ver_sl_atr_mult': 1.5,
        # Bearish engulfing params
        'min_body_atr': 0.5, 'swing_lookback': 20, 'swing_offset': 3,
        'resistance_tolerance': 1.5, 'engulf_sl_atr_mult': 1.0,
        # RSI divergence params
        'div_lookback': 14, 'div_offset': 3, 'rsi_diff_min': 5.0,
        'rsi_min': 30, 'rsi_max': 65, 'require_red_candle': True,
        'div_sl_atr_mult': 1.5,
    }

    # ==================================================================
    # SECTION 1: INDIVIDUAL SIGNAL TESTS (SMA200 bear filter)
    # ==================================================================
    print_section("SECTION 1: INDIVIDUAL SIGNALS (SMA200 bear filter)")

    signal_types = [
        ('Donchian only', ['donchian']),
        ('MACD Cross only', ['macd']),
        ('Volume Exhaustion only', ['volume_exhaustion']),
        ('Bearish Engulfing only', ['bearish_engulfing']),
        ('RSI Divergence only', ['rsi_divergence']),
    ]

    individual_results = {}
    for name, signals in signal_types:
        p = {**base_params, 'label': name}
        trades, eq, cap, counts = backtest_multi_signal_short(
            coin_data, p, bear_simple, signals_enabled=signals)
        stats = compute_stats(trades, name)
        individual_results[name] = {'stats': stats, 'eq': eq, 'trades': trades, 'counts': counts}
        print_stats_row(name, stats, eq)
        print(f"    Signal entries: {counts}")

    # Per-signal deep dive
    for name, data in individual_results.items():
        if data['stats'].get('total_trades', 0) > 0:
            print(f"\n  --- {name} ---")
            print_entry_breakdown(data['trades'])

    # ==================================================================
    # SECTION 2: PAIRWISE COMBINATIONS
    # ==================================================================
    print_section("SECTION 2: PAIRWISE COMBINATIONS (SMA200 bear filter)")

    combos = [
        ('Donchian + MACD', ['donchian', 'macd']),
        ('Donchian + Vol Exhaustion', ['donchian', 'volume_exhaustion']),
        ('Donchian + Engulfing', ['donchian', 'bearish_engulfing']),
        ('Donchian + RSI Div', ['donchian', 'rsi_divergence']),
        ('MACD + Vol Exhaustion', ['macd', 'volume_exhaustion']),
        ('MACD + Engulfing', ['macd', 'bearish_engulfing']),
    ]

    combo_results = {}
    for name, signals in combos:
        p = {**base_params, 'label': name}
        trades, eq, cap, counts = backtest_multi_signal_short(
            coin_data, p, bear_simple, signals_enabled=signals)
        stats = compute_stats(trades, name)
        combo_results[name] = {'stats': stats, 'eq': eq, 'trades': trades, 'counts': counts}
        print_stats_row(name, stats, eq)
        print(f"    Entries: {counts}")

    # Entry breakdown for best combo
    best_combo = max(combo_results.items(), key=lambda x: x[1]['stats'].get('total_return_pct', -999))
    print(f"\n  Best combo: {best_combo[0]}")
    print_entry_breakdown(best_combo[1]['trades'])

    # ==================================================================
    # SECTION 3: ALL SIGNALS COMBINED
    # ==================================================================
    print_section("SECTION 3: ALL SIGNALS COMBINED")

    all_signals = ['donchian', 'macd', 'volume_exhaustion', 'bearish_engulfing', 'rsi_divergence']

    # SMA200 filter
    p = {**base_params, 'label': 'All signals (SMA200)'}
    all_trades, all_eq, all_cap, all_counts = backtest_multi_signal_short(
        coin_data, p, bear_simple, signals_enabled=all_signals)
    all_stats = compute_stats(all_trades, 'All signals (SMA200)')
    print_stats_row('All signals (SMA200)', all_stats, all_eq)
    print(f"    Entries: {all_counts}")
    print_entry_breakdown(all_trades)

    # Death cross filter
    p2 = {**base_params, 'label': 'All signals (DeathX)'}
    alldc_trades, alldc_eq, alldc_cap, alldc_counts = backtest_multi_signal_short(
        coin_data, p2, bear_strict, signals_enabled=all_signals)
    alldc_stats = compute_stats(alldc_trades, 'All signals (DeathX)')
    print_stats_row('All signals (DeathX)', alldc_stats, alldc_eq)
    print(f"    Entries: {alldc_counts}")

    # ==================================================================
    # SECTION 4: PARAMETER SWEEP — MACD (highest frequency signal)
    # ==================================================================
    print_section("SECTION 4: MACD PARAMETER SWEEP")

    macd_sweep = []
    for zero_thresh in [-0.5, 0, 0.5, 1.0]:
        for vol in [0, 1.0, 1.5]:
            for sl_look in [5, 10, 15]:
                for sl_atr in [1.0, 1.5, 2.0, 2.5]:
                    lbl = f"ZT{zero_thresh}_V{vol}_SL{sl_look}_A{sl_atr}"
                    p = {**base_params,
                         'label': lbl,
                         'macd_zero_threshold': zero_thresh,
                         'macd_vol_mult': vol,
                         'macd_sl_lookback': sl_look,
                         'macd_sl_atr_mult': sl_atr,
                         }
                    macd_sweep.append(p)

    print(f"  Running {len(macd_sweep)} MACD variants...")
    macd_results = []
    for i, p in enumerate(macd_sweep):
        trades, eq, cap, counts = backtest_multi_signal_short(
            coin_data, p, bear_simple, signals_enabled=['macd'])
        stats = compute_stats(trades, p['label'])
        macd_results.append({'label': p['label'], 'params': p, 'stats': stats, 'eq': eq})
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(macd_sweep)}")

    macd_results.sort(key=lambda r: -r['stats'].get('total_return_pct', -999))

    print(f"\n  TOP 15 MACD VARIANTS:")
    for r in macd_results[:15]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    # Parameter patterns
    top20 = macd_results[:20]
    print(f"\n  PARAMETER PATTERNS (Top 20 MACD):")
    for pname, key in [('Zero threshold', 'macd_zero_threshold'),
                        ('Volume mult', 'macd_vol_mult'),
                        ('SL lookback', 'macd_sl_lookback'),
                        ('SL ATR mult', 'macd_sl_atr_mult')]:
        vals = [r['params'][key] for r in top20]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        print(f"    {pname:<18s}: {dict(sorted(counts.items()))}")

    # ==================================================================
    # SECTION 5: BEST COMBINED — PARAMETER TUNING
    # ==================================================================
    print_section("SECTION 5: BEST COMBINED — TUNING EXIT PARAMS")

    # Use best MACD params with donchian
    best_macd = macd_results[0]['params'] if macd_results else base_params
    tune_results = []
    for atr_mult in [1.5, 2.0, 2.5, 3.0]:
        for exit_period in [10, 15, 20]:
            for max_hold in [20, 30, 45]:
                lbl = f"ATR{atr_mult}_Exit{exit_period}_Hold{max_hold}"
                p = {**best_macd,
                     'label': lbl,
                     'atr_mult': atr_mult,
                     'exit_period': exit_period,
                     'max_hold_days': max_hold,
                     }
                tune_results.append(p)

    print(f"  Running {len(tune_results)} exit tuning variants...")
    exit_results = []
    for i, p in enumerate(tune_results):
        trades, eq, cap, counts = backtest_multi_signal_short(
            coin_data, p, bear_simple, signals_enabled=['donchian', 'macd'])
        stats = compute_stats(trades, p['label'])
        exit_results.append({'label': p['label'], 'params': p, 'stats': stats, 'eq': eq, 'counts': counts})

    exit_results.sort(key=lambda r: -r['stats'].get('total_return_pct', -999))

    print(f"\n  TOP 10 EXIT TUNING VARIANTS:")
    for r in exit_results[:10]:
        print_stats_row(r['label'], r['stats'], r['eq'])
        print(f"    Entries: {r['counts']}")

    # ==================================================================
    # SECTION 6: WALK-FORWARD
    # ==================================================================
    print_section("SECTION 6: WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')
    train_data, test_data = split_coin_data(coin_data, cutoff)
    btc_train = train_data.get('BTC-USD')
    btc_test = test_data.get('BTC-USD')

    train_bear = compute_btc_bear_filter(btc_train, require_death_cross=False) if btc_train is not None else {}
    test_bear = compute_btc_bear_filter(btc_test, require_death_cross=False) if btc_test is not None else {}
    train_bear_dc = compute_btc_bear_filter(btc_train, require_death_cross=True) if btc_train is not None else {}
    test_bear_dc = compute_btc_bear_filter(btc_test, require_death_cross=True) if btc_test is not None else {}

    # Walk-forward variants
    wf_configs = [
        ('Donchian only (DeathX)', SHORT_DEFAULT_PARAMS, ['donchian'], train_bear_dc, test_bear_dc),
        ('All signals (SMA200)', base_params, all_signals, train_bear, test_bear),
        ('Donchian + MACD (SMA200)', base_params, ['donchian', 'macd'], train_bear, test_bear),
    ]

    # Add best exit-tuned variant
    if exit_results:
        best_exit = exit_results[0]['params']
        wf_configs.append(('Best tuned (SMA200)', best_exit, ['donchian', 'macd'], train_bear, test_bear))

    for name, params, signals, tr_bear, te_bear in wf_configs:
        p = {**params, 'label': f'{name} [train]'}
        train_trades, train_eq, _, train_counts = backtest_multi_signal_short(
            train_data, p, tr_bear, signals_enabled=signals)
        train_stats = compute_stats(train_trades, f'{name} [train]')

        p2 = {**params, 'label': f'{name} [OOS]'}
        test_trades, test_eq, _, test_counts = backtest_multi_signal_short(
            test_data, p2, te_bear, signals_enabled=signals)
        test_stats = compute_stats(test_trades, f'{name} [OOS]')

        print(f"\n  {name}:")
        print(f"    Train: {train_stats.get('total_trades',0)}t, "
              f"WR:{train_stats.get('win_rate',0)*100:.1f}%, "
              f"PF:{train_stats.get('profit_factor',0):.2f}, "
              f"Ret:{train_stats.get('total_return_pct',0):+.1f}%  "
              f"Entries: {train_counts}")
        print(f"    OOS:   {test_stats.get('total_trades',0)}t, "
              f"WR:{test_stats.get('win_rate',0)*100:.1f}%, "
              f"PF:{test_stats.get('profit_factor',0):.2f}, "
              f"Ret:{test_stats.get('total_return_pct',0):+.1f}%  "
              f"Entries: {test_counts}")

    # ==================================================================
    # SECTION 7: FULL PORTFOLIO (LONG + BEST SHORTS)
    # ==================================================================
    print_section("SECTION 7: FULL PORTFOLIO (Long + Best Shorts)")

    # Long baseline
    long_trades, long_eq, long_cap, _ = backtest_portfolio_phase3(
        coin_data, LONG_PROD_PARAMS, bull_filter, pyramiding=True)
    long_stats = compute_stats(long_trades, 'Long-Only')

    # Donchian-only shorts baseline
    dc_trades, dc_eq, _, _ = backtest_portfolio_short(
        coin_data, SHORT_DEFAULT_PARAMS, bear_strict)
    dc_stats = compute_stats(dc_trades, 'Donchian-only (DeathX)')

    # Multi-signal shorts
    ms_trades, ms_eq, _, ms_counts = backtest_multi_signal_short(
        coin_data, base_params, bear_simple, signals_enabled=['donchian', 'macd'])
    ms_stats = compute_stats(ms_trades, 'DC + MACD (SMA200)')

    # All signals
    all_trades2, all_eq2, _, all_counts2 = backtest_multi_signal_short(
        coin_data, base_params, bear_simple, signals_enabled=all_signals)
    all_stats2 = compute_stats(all_trades2, 'All signals (SMA200)')

    starting = LONG_PROD_PARAMS['starting_capital']
    long_by_date = {e['date']: e['equity'] for e in long_eq}

    def merge_equity(short_eq):
        short_by_date = {e['date']: e['equity'] for e in short_eq}
        dates = sorted(set(long_by_date.keys()) | set(short_by_date.keys()))
        return [{'date': d, 'equity': starting + (long_by_date.get(d, starting) - starting) +
                 (short_by_date.get(d, starting) - starting)} for d in dates]

    comb_dc = merge_equity(dc_eq)
    comb_ms = merge_equity(ms_eq)
    comb_all = merge_equity(all_eq2)

    print(f"\n  {'='*130}")
    print_stats_row('Long-Only', long_stats, long_eq)
    print_stats_row('Long + DC shorts (DeathX)', compute_stats(long_trades + dc_trades, 'L+DC'), comb_dc)
    print_stats_row('Long + DC+MACD (SMA200)', compute_stats(long_trades + ms_trades, 'L+MS'), comb_ms)
    print_stats_row('Long + All signals (SMA200)', compute_stats(long_trades + all_trades2, 'L+All'), comb_all)

    # Drawdowns
    for name, eq in [('Long-only', long_eq), ('Long+DC', comb_dc),
                     ('Long+DC+MACD', comb_ms), ('Long+All', comb_all)]:
        dd, _ = compute_max_drawdown(eq)
        print(f"    {name:<20s} MaxDD: {dd:.1f}%")

    print(f"\n  Short signal entries: DC+MACD={ms_counts}, All={all_counts2}")

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================
    print_section("FINAL VERDICT")

    dc_ret = dc_stats.get('total_return_pct', 0)
    ms_ret = ms_stats.get('total_return_pct', 0)
    all_ret = all_stats2.get('total_return_pct', 0)

    print(f"  Donchian-only (DeathX): {dc_stats.get('total_trades',0)} trades, {dc_ret:+.1f}%")
    print(f"  DC + MACD (SMA200):     {ms_stats.get('total_trades',0)} trades, {ms_ret:+.1f}%")
    print(f"  All signals (SMA200):   {all_stats2.get('total_trades',0)} trades, {all_ret:+.1f}%")

    best = max([('DC-only', dc_ret), ('DC+MACD', ms_ret), ('All', all_ret)], key=lambda x: x[1])
    print(f"\n  Best short approach: {best[0]} ({best[1]:+.1f}%)")

    print(f"\n{'='*100}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
