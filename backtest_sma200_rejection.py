"""SMA200 Rejection Short Backtest — Alternative Short Entry Signal

Tests shorting when price rallies to SMA(200) from below and gets rejected.
The hypothesis: during bear markets, the SMA(200) acts as resistance. When price
approaches it and fails, that's a high-probability short entry.

This can run standalone AND combined with the existing Donchian shorts to
increase total short trade count (from ~16 to potentially 24-36).

Entry conditions (SMA200 Rejection):
  1. Bear filter active (death cross or BTC < SMA200)
  2. Price within proximity_pct of SMA(200) — approaching resistance
  3. Rejection candle: close < SMA(200) AND close < open (bearish)
  4. Volume >= volume_mult * SMA(volume, 20) on rejection day
  5. RSI > 40 (not already oversold — room to fall)

Exit conditions (same as Donchian shorts):
  - ATR trailing stop (inverted — low watermark + ATR*mult)
  - N-day high breakout exit
  - Max hold days
  - Emergency stop

Test plan:
  1. SMA200 rejection standalone (various parameter combos)
  2. SMA200 rejection + Donchian shorts combined
  3. Walk-forward validation
  4. Combined long+short portfolio
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
    backtest_combined_portfolio,
)
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# SMA200 REJECTION PARAMETERS
# ============================================================================

SMA200_DEFAULT_PARAMS = {
    **SHORT_DEFAULT_PARAMS,
    'label': 'SMA200 Rejection Base',
    # SMA200 proximity: how close price must be to SMA200 (%)
    'proximity_pct': 2.0,
    # Require bearish candle (close < open)
    'require_bearish_candle': True,
    # Require close < SMA200 (not just near it)
    'require_close_below_sma': True,
    # Volume filter for rejection candle
    'volume_mult': 1.2,
    # RSI floor — don't short if already oversold
    'rsi_floor': 40,
    # Days price must have been below SMA200 recently (of last N days)
    'below_sma_days': 5,
    'below_sma_lookback': 10,
    # ATR mult for stop (above SMA200 + buffer)
    'atr_mult': 2.0,
    # Use SMA200 for stop placement instead of pure ATR
    'sma_stop': True,
    'sma_stop_buffer_atr': 1.0,  # stop = SMA200 + 1x ATR above
}


# ============================================================================
# SMA200 REJECTION BACKTEST
# ============================================================================

def calculate_sma200_indicators(df, params):
    """Calculate indicators for SMA200 rejection trading.

    Adds SMA(200), SMA(50), and short indicators on top.
    """
    df = calculate_short_indicators(df, params)
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    return df


def backtest_sma200_rejection(coin_data, params, bear_filter, btc_sma200=None):
    """Portfolio backtest for SMA200 rejection short entries.

    Entry: price approaches SMA200 from below, gets rejected (bearish candle).
    Exit: same trailing stop / high exit / max hold as Donchian shorts.

    btc_sma200: optional dict of date -> sma200 value for BTC (for cross-asset filter)
    """
    label = params.get('label', 'SMA200 Rejection')
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']
    funding_daily = params.get('funding_rate_daily', 0) / 100

    proximity_pct = params.get('proximity_pct', 2.0) / 100
    require_bearish = params.get('require_bearish_candle', True)
    require_below_sma = params.get('require_close_below_sma', True)
    vol_mult = params.get('volume_mult', 1.2)
    rsi_floor = params.get('rsi_floor', 40)
    below_days_req = params.get('below_sma_days', 5)
    below_lookback = params.get('below_sma_lookback', 10)
    use_sma_stop = params.get('sma_stop', True)
    sma_buffer_atr = params.get('sma_stop_buffer_atr', 1.0)

    # Pre-calculate indicators
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_sma200_indicators(df, params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'donchian_low', 'atr',
                                        'volume_sma', 'ema_21', 'rsi',
                                        'exit_high', 'sma_200'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 30:
            prepared[symbol] = df_ind

    # Build unified daily timeline
    all_dates = set()
    for symbol, df in prepared.items():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Build lookups: symbol -> {date -> row}
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup

    # Build recent-history lookups for below-SMA check
    date_history = {}  # symbol -> {date -> list of last N closes vs SMA200}
    for symbol, df in prepared.items():
        dates_list = sorted(df['time'].dt.date.tolist())
        history = {}
        for j in range(below_lookback, len(dates_list)):
            window = []
            for k in range(j - below_lookback, j):
                r = lookups[symbol].get(dates_list[k])
                if r is not None and not pd.isna(r.get('sma_200')):
                    window.append(float(r['close']) < float(r['sma_200']))
            history[dates_list[j]] = window
        date_history[symbol] = history

    # State
    positions = {}
    trades = []
    equity_curve = []

    for date in all_dates:
        is_bear = bear_filter.get(date, False)

        # === EXITS (same as Donchian shorts) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_low = float(row['low'])
            current_high = float(row['high'])
            current_atr = float(row['atr'])

            # Update low watermark
            if current_low < pos['low_watermark']:
                pos['low_watermark'] = current_low

            pos['hold_days'] += 1
            exit_reason = None
            exit_price = None

            # 1. ATR trailing stop (inverted)
            if use_sma_stop and not pd.isna(row.get('sma_200')):
                # Stop above SMA200 + buffer
                sma_val = float(row['sma_200'])
                trailing_stop = max(
                    pos['low_watermark'] + params['atr_mult'] * current_atr,
                    sma_val + sma_buffer_atr * current_atr
                )
            else:
                trailing_stop = pos['low_watermark'] + params['atr_mult'] * current_atr

            if current_close > trailing_stop:
                exit_reason = 'Trailing stop (ATR)'
                exit_price = trailing_stop

            # 2. N-day high exit (short cover)
            if not exit_reason and not pd.isna(row.get('exit_high')):
                if current_close > float(row['exit_high']):
                    exit_reason = f'{params["exit_period"]}-day high exit'
                    exit_price = current_close

            # 3. Emergency stop
            if not exit_reason:
                loss_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if loss_pct >= params['emergency_stop_pct']:
                    exit_reason = f'Emergency stop ({loss_pct:.1f}%)'
                    exit_price = current_close

            # 4. Max hold
            if not exit_reason and pos['hold_days'] >= params['max_hold_days']:
                exit_reason = f'Max hold ({params["max_hold_days"]}d)'
                exit_price = current_close

            # 5. Blow-off detection (oversold bounce risk)
            if not exit_reason:
                rsi_val = float(row['rsi']) if not pd.isna(row['rsi']) else 50
                vol_val = float(row['volume']) if not pd.isna(row['volume']) else 0
                vol_sma = float(row['volume_sma']) if not pd.isna(row['volume_sma']) and row['volume_sma'] > 0 else 1
                if rsi_val < params.get('rsi_blowoff', 20) and vol_val > params.get('volume_blowoff', 3.0) * vol_sma:
                    # Tighten stop
                    tight_stop = pos['low_watermark'] + params.get('atr_mult_tight', 1.0) * current_atr
                    if current_close > tight_stop:
                        exit_reason = 'Blow-off tightened stop'
                        exit_price = tight_stop

            # Partial profit taking
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
                        'price_pnl_pct': price_pnl, 'funding_pnl_pct': fund_pnl,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP1 ({fraction*100:.0f}% @ {params["tp1_pct"]}%)',
                        'size_usd': partial_size, 'side': 'SHORT', 'win': pnl_pct > 0,
                        'entry_type': 'sma200_rejection',
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
                        'price_pnl_pct': price_pnl, 'funding_pnl_pct': fund_pnl,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP2 ({fraction*100:.0f}% @ {params["tp2_pct"]}%)',
                        'size_usd': partial_size, 'side': 'SHORT', 'win': pnl_pct > 0,
                        'entry_type': 'sma200_rejection',
                    })

            # Full exit
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
                    'price_pnl_pct': price_pnl_pct, 'funding_pnl_pct': funding_pnl_pct,
                    'hold_days': pos['hold_days'], 'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'], 'side': 'SHORT',
                    'win': total_pnl_pct > 0,
                    'entry_type': 'sma200_rejection',
                })
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del positions[s]

        # === NEW SHORT ENTRIES: SMA200 REJECTION ===
        if len(positions) < max_positions and is_bear:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(date)
                if row is None:
                    continue
                if pd.isna(row.get('sma_200')):
                    continue

                current_close = float(row['close'])
                current_open = float(row['open'])
                sma_200 = float(row['sma_200'])
                current_atr = float(row['atr'])

                # 1. Proximity check: price within proximity_pct of SMA200
                distance_pct = abs(current_close - sma_200) / sma_200
                if distance_pct > proximity_pct:
                    continue

                # 2. Bearish candle: close < open
                if require_bearish and current_close >= current_open:
                    continue

                # 3. Close below SMA200
                if require_below_sma and current_close >= sma_200:
                    continue

                # 4. Volume check
                if vol_mult > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    if float(row['volume']) < vol_mult * vol_sma:
                        continue

                # 5. RSI floor — don't short if already oversold
                rsi_val = float(row['rsi']) if not pd.isna(row['rsi']) else 50
                if rsi_val < rsi_floor:
                    continue

                # 6. Recent history: price was below SMA200 for N of last M days
                recent = date_history.get(symbol, {}).get(date, [])
                if len(recent) >= below_lookback:
                    days_below = sum(recent)
                    if days_below < below_days_req:
                        continue

                # All conditions met — enter short
                total_equity = capital + sum(
                    p['size_usd'] * (1 + (p['entry_price'] - float(lookups[s].get(date, row)['close'])) / p['entry_price'])
                    if lookups.get(s, {}).get(date) is not None else p['size_usd']
                    for s, p in positions.items()
                )
                risk_amount = total_equity * risk_pct
                entry_price = current_close * (1 - cost_per_side)

                # Stop placement: above SMA200 + ATR buffer
                if use_sma_stop:
                    stop_price = sma_200 + sma_buffer_atr * current_atr
                    stop_distance = stop_price - entry_price
                else:
                    stop_distance = params['atr_mult'] * current_atr

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
                }

        # Equity tracking
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                price_change = (pos['entry_price'] - float(row['close'])) / pos['entry_price']
                current_val = pos['size_usd'] * (1 + price_change)
                total_equity += max(current_val, 0)
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining positions
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
            'price_pnl_pct': price_pnl_pct, 'funding_pnl_pct': funding_pnl_pct,
            'hold_days': pos['hold_days'], 'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'], 'side': 'SHORT', 'win': total_pnl_pct > 0,
            'entry_type': 'sma200_rejection',
        })

    return trades, equity_curve, capital


# ============================================================================
# COMBINED: DONCHIAN + SMA200 REJECTION SHORTS
# ============================================================================

def backtest_combined_shorts(coin_data, donchian_params, sma200_params,
                              bear_filter, donchian_bear_filter=None):
    """Run both Donchian shorts and SMA200 rejection shorts on same capital pool.

    Shares the max_positions limit. Donchian entries get priority (checked first),
    SMA200 rejection fills remaining slots.

    If donchian_bear_filter is provided, uses it for Donchian entries and
    bear_filter for SMA200 entries. Otherwise uses bear_filter for both.
    """
    label = 'Combined (Donchian + SMA200 Rejection)'
    cost_per_side = (donchian_params['fee_pct'] + donchian_params['slippage_pct']) / 100
    max_positions = donchian_params['max_positions']
    risk_pct = donchian_params['risk_per_trade_pct'] / 100
    capital = donchian_params['starting_capital']
    funding_daily = donchian_params.get('funding_rate_daily', 0) / 100

    dc_bear = donchian_bear_filter or bear_filter

    # SMA200 params
    proximity_pct = sma200_params.get('proximity_pct', 2.0) / 100
    require_bearish = sma200_params.get('require_bearish_candle', True)
    require_below_sma = sma200_params.get('require_close_below_sma', True)
    sma_vol_mult = sma200_params.get('volume_mult', 1.2)
    rsi_floor = sma200_params.get('rsi_floor', 40)
    below_days_req = sma200_params.get('below_sma_days', 5)
    below_lookback = sma200_params.get('below_sma_lookback', 10)
    use_sma_stop = sma200_params.get('sma_stop', True)
    sma_buffer_atr = sma200_params.get('sma_stop_buffer_atr', 1.0)

    # Pre-calculate indicators (need SMA200 for rejection entries)
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_sma200_indicators(df, donchian_params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'donchian_low', 'atr',
                                        'volume_sma', 'ema_21', 'rsi',
                                        'exit_high', 'sma_200'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 30:
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

    # Recent below-SMA history
    date_history = {}
    for symbol, df in prepared.items():
        dates_list = sorted(df['time'].dt.date.tolist())
        history = {}
        for j in range(below_lookback, len(dates_list)):
            window = []
            for k in range(j - below_lookback, j):
                r = lookups[symbol].get(dates_list[k])
                if r is not None and not pd.isna(r.get('sma_200')):
                    window.append(float(r['close']) < float(r['sma_200']))
            history[dates_list[j]] = window
        date_history[symbol] = history

    positions = {}
    trades = []
    equity_curve = []
    donchian_entries = 0
    sma200_entries = 0

    for date in all_dates:
        is_dc_bear = dc_bear.get(date, False)
        is_sma_bear = bear_filter.get(date, False)

        # === EXITS (all positions, regardless of entry type) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_low = float(row['low'])
            current_atr = float(row['atr'])
            entry_type = pos.get('entry_type', 'donchian')

            # Use the appropriate params for exit logic
            ep = sma200_params if entry_type == 'sma200_rejection' else donchian_params

            if current_low < pos['low_watermark']:
                pos['low_watermark'] = current_low

            pos['hold_days'] += 1
            exit_reason = None
            exit_price = None

            # Trailing stop
            if entry_type == 'sma200_rejection' and use_sma_stop and not pd.isna(row.get('sma_200')):
                sma_val = float(row['sma_200'])
                trailing_stop = max(
                    pos['low_watermark'] + ep['atr_mult'] * current_atr,
                    sma_val + sma_buffer_atr * current_atr
                )
            else:
                trailing_stop = pos['low_watermark'] + ep['atr_mult'] * current_atr

            if current_close > trailing_stop:
                exit_reason = 'Trailing stop (ATR)'
                exit_price = trailing_stop

            # N-day high exit
            if not exit_reason and not pd.isna(row.get('exit_high')):
                if current_close > float(row['exit_high']):
                    exit_reason = f'{ep["exit_period"]}-day high exit'
                    exit_price = current_close

            # Emergency stop
            if not exit_reason:
                loss_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if loss_pct >= ep['emergency_stop_pct']:
                    exit_reason = f'Emergency stop ({loss_pct:.1f}%)'
                    exit_price = current_close

            # Max hold
            if not exit_reason and pos['hold_days'] >= ep['max_hold_days']:
                exit_reason = f'Max hold ({ep["max_hold_days"]}d)'
                exit_price = current_close

            # Partial TPs
            if not exit_reason:
                gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= ep['tp1_pct']:
                    fraction = ep['tp1_fraction']
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
                        'win': pnl_pct > 0, 'entry_type': entry_type,
                    })
                elif pos['partials_taken'] == 1 and gain_pct >= ep['tp2_pct']:
                    fraction = ep['tp2_fraction']
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
                        'win': pnl_pct > 0, 'entry_type': entry_type,
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
                    'win': total_pnl_pct > 0, 'entry_type': entry_type,
                })
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del positions[s]

        # === NEW ENTRIES: DONCHIAN FIRST, THEN SMA200 REJECTION ===

        # --- Donchian short entries (death cross gated) ---
        if len(positions) < max_positions and is_dc_bear:
            for symbol in prepared:
                if symbol in positions or len(positions) >= max_positions:
                    continue

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row.get('donchian_low')):
                    continue

                current_close = float(row['close'])
                breakdown = current_close < float(prev_row['donchian_low'])

                dc_vol_mult = donchian_params.get('volume_mult', 1.5)
                if dc_vol_mult > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > dc_vol_mult * vol_sma
                else:
                    volume_ok = True

                trend_ok = current_close < float(row['ema_21'])

                if breakdown and volume_ok and trend_ok:
                    total_equity = capital + sum(
                        p['size_usd'] * (1 + (p['entry_price'] - float(lookups[s].get(date, row)['close'])) / p['entry_price'])
                        if lookups.get(s, {}).get(date) is not None else p['size_usd']
                        for s, p in positions.items()
                    )
                    risk_amount = total_equity * risk_pct
                    entry_price = current_close * (1 - cost_per_side)
                    atr_val = float(row['atr'])
                    stop_distance = donchian_params['atr_mult'] * atr_val
                    stop_pct = stop_distance / entry_price

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
                        'entry_type': 'donchian',
                    }
                    donchian_entries += 1

        # --- SMA200 Rejection entries (bear filter gated) ---
        if len(positions) < max_positions and is_sma_bear:
            for symbol in prepared:
                if symbol in positions or len(positions) >= max_positions:
                    continue

                row = lookups[symbol].get(date)
                if row is None or pd.isna(row.get('sma_200')):
                    continue

                current_close = float(row['close'])
                current_open = float(row['open'])
                sma_200 = float(row['sma_200'])
                current_atr = float(row['atr'])

                # Proximity check
                distance_pct = abs(current_close - sma_200) / sma_200
                if distance_pct > proximity_pct:
                    continue

                if require_bearish and current_close >= current_open:
                    continue

                if require_below_sma and current_close >= sma_200:
                    continue

                if sma_vol_mult > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    if float(row['volume']) < sma_vol_mult * vol_sma:
                        continue

                rsi_val = float(row['rsi']) if not pd.isna(row['rsi']) else 50
                if rsi_val < rsi_floor:
                    continue

                recent = date_history.get(symbol, {}).get(date, [])
                if len(recent) >= below_lookback:
                    days_below = sum(recent)
                    if days_below < below_days_req:
                        continue

                # Enter short
                total_equity = capital + sum(
                    p['size_usd'] * (1 + (p['entry_price'] - float(lookups[s].get(date, row)['close'])) / p['entry_price'])
                    if lookups.get(s, {}).get(date) is not None else p['size_usd']
                    for s, p in positions.items()
                )
                risk_amount = total_equity * risk_pct
                entry_price = current_close * (1 - cost_per_side)

                if use_sma_stop:
                    stop_price = sma_200 + sma_buffer_atr * current_atr
                    stop_distance = stop_price - entry_price
                else:
                    stop_distance = sma200_params['atr_mult'] * current_atr

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
                    'entry_type': 'sma200_rejection',
                }
                sma200_entries += 1

        # Equity tracking
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                price_change = (pos['entry_price'] - float(row['close'])) / pos['entry_price']
                current_val = pos['size_usd'] * (1 + price_change)
                total_equity += max(current_val, 0)
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
            'win': total_pnl_pct > 0, 'entry_type': pos.get('entry_type', 'donchian'),
        })

    return trades, equity_curve, capital, donchian_entries, sma200_entries


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("SMA200 REJECTION SHORT BACKTEST")
    print(f"  Coins: {', '.join(c.replace('-USD','') for c in SHORT_COINS)}")
    print(f"  Hypothesis: Short when price rallies to SMA200 from below and gets rejected")
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

    # ==================================================================
    # SECTION 1: SMA200 REJECTION STANDALONE — PARAMETER SWEEP
    # ==================================================================
    print_section("SECTION 1: SMA200 REJECTION STANDALONE — PARAMETER SWEEP")

    sweep_configs = []
    for prox in [1.0, 1.5, 2.0, 3.0, 4.0]:
        for vol in [0, 1.0, 1.2, 1.5]:
            for rsi in [30, 40, 50]:
                for buf in [0.5, 1.0, 1.5]:
                    lbl = f"P{prox}_V{vol}_R{rsi}_B{buf}"
                    p = {
                        **SMA200_DEFAULT_PARAMS,
                        'label': lbl,
                        'proximity_pct': prox,
                        'volume_mult': vol,
                        'rsi_floor': rsi,
                        'sma_stop_buffer_atr': buf,
                    }
                    sweep_configs.append(p)

    # Test with both bear filters
    print(f"\n  Running {len(sweep_configs)} variants with SMA200 bear filter...")
    sma_results = []
    for i, params in enumerate(sweep_configs):
        trades, eq, cap = backtest_sma200_rejection(coin_data, params, bear_simple)
        stats = compute_stats(trades, params['label'])
        sma_results.append({
            'label': params['label'],
            'params': params,
            'stats': stats,
            'eq': eq,
            'trades': trades,
        })
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(sweep_configs)}")

    sma_results.sort(key=lambda r: -r['stats'].get('total_return_pct', -999))

    print(f"\n  {'='*130}")
    print(f"  TOP 20 SMA200 REJECTION VARIANTS (SMA200 bear filter)")
    print(f"  {'='*130}")
    for r in sma_results[:20]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    # Same with death cross filter
    print(f"\n  Running {len(sweep_configs)} variants with DEATH CROSS bear filter...")
    dc_results = []
    for i, params in enumerate(sweep_configs):
        trades, eq, cap = backtest_sma200_rejection(coin_data, params, bear_strict)
        stats = compute_stats(trades, params['label'])
        dc_results.append({
            'label': params['label'],
            'params': params,
            'stats': stats,
            'eq': eq,
            'trades': trades,
        })
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(sweep_configs)}")

    dc_results.sort(key=lambda r: -r['stats'].get('total_return_pct', -999))

    print(f"\n  {'='*130}")
    print(f"  TOP 20 SMA200 REJECTION VARIANTS (Death Cross bear filter)")
    print(f"  {'='*130}")
    for r in dc_results[:20]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    # Parameter pattern analysis
    for filter_name, results in [('SMA200', sma_results), ('Death Cross', dc_results)]:
        top20 = results[:20]
        print(f"\n  PARAMETER PATTERNS — Top 20 ({filter_name}):")
        for param_name, key in [('Proximity %', 'proximity_pct'),
                                 ('Volume mult', 'volume_mult'),
                                 ('RSI floor', 'rsi_floor'),
                                 ('SMA buffer ATR', 'sma_stop_buffer_atr')]:
            vals = [r['params'][key] for r in top20]
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            print(f"    {param_name:<18s}: {dict(sorted(counts.items()))}")

    # ==================================================================
    # SECTION 2: BEST STANDALONE — DEEP DIVE
    # ==================================================================
    print_section("SECTION 2: BEST STANDALONE — TRADE ANALYSIS")

    # Use the best SMA200 filter result
    best = sma_results[0] if sma_results and sma_results[0]['stats'].get('total_trades', 0) > 0 else None
    best_dc = dc_results[0] if dc_results and dc_results[0]['stats'].get('total_trades', 0) > 0 else None

    for label, result in [('SMA200 filter', best), ('Death Cross filter', best_dc)]:
        if not result:
            print(f"\n  No trades with {label}")
            continue

        print(f"\n  Best {label}: {result['label']}")
        ts = result['trades']

        if ts:
            full_exits = [t for t in ts if 'Partial' not in t['exit_reason']]
            per_coin = compute_per_coin_stats(ts)

            print(f"\n  {'Coin':<10s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'PF':>6s} "
                  f"{'Sum P&L%':>10s} {'Avg Hold':>9s}")
            for coin, cs in sorted(per_coin.items()):
                coin_exits = [t for t in full_exits if t['symbol'] == coin]
                avg_hold = (sum(t['hold_days'] for t in coin_exits) / len(coin_exits)
                            if coin_exits else 0)
                print(f"  {coin:<10s} {cs['trades']:>6d} {cs['wins']:>5d} "
                      f"{cs['win_rate']*100:>5.1f}% {cs['profit_factor']:>5.2f} "
                      f"{cs['sum_pnl_pct']:>+9.1f}% {avg_hold:>7.1f}d")

            # Exit reasons
            reasons = {}
            for t in full_exits:
                r = t['exit_reason'].split('(')[0].strip()
                reasons[r] = reasons.get(r, 0) + 1
            print(f"\n  Exit reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(full_exits) * 100 if full_exits else 0
                print(f"    {reason:<30s} {count:>4d} ({pct:.1f}%)")

            # Win/loss
            wins = [t for t in full_exits if t.get('win')]
            losses = [t for t in full_exits if not t.get('win')]
            if wins:
                print(f"\n  Wins: {len(wins)}, Avg: {sum(t['pnl_pct'] for t in wins)/len(wins):+.1f}%, "
                      f"Max: {max(t['pnl_pct'] for t in wins):+.1f}%")
            if losses:
                print(f"  Losses: {len(losses)}, Avg: {sum(t['pnl_pct'] for t in losses)/len(losses):+.1f}%, "
                      f"Max: {min(t['pnl_pct'] for t in losses):+.1f}%")

    # ==================================================================
    # SECTION 3: DONCHIAN BASELINE vs SMA200 REJECTION
    # ==================================================================
    print_section("SECTION 3: DONCHIAN BASELINE vs SMA200 REJECTION")

    # Donchian-only baseline (death cross)
    dc_trades, dc_eq, dc_cap, _ = backtest_portfolio_short(
        coin_data, SHORT_DEFAULT_PARAMS, bear_strict, pyramiding=False)
    dc_stats = compute_stats(dc_trades, 'Donchian-only (DeathX)')

    # Best SMA200-only
    best_sma_params = sma_results[0]['params'] if sma_results else SMA200_DEFAULT_PARAMS
    best_dc_params = dc_results[0]['params'] if dc_results else SMA200_DEFAULT_PARAMS

    sma_only_trades, sma_only_eq, _ = backtest_sma200_rejection(
        coin_data, best_sma_params, bear_simple)
    sma_only_stats = compute_stats(sma_only_trades, f'SMA200 Rejection (SMA200 bear)')

    dc_only_trades, dc_only_eq, _ = backtest_sma200_rejection(
        coin_data, best_dc_params, bear_strict)
    dc_only_stats = compute_stats(dc_only_trades, f'SMA200 Rejection (DeathX)')

    print(f"\n  {'='*130}")
    print(f"  COMPARISON")
    print(f"  {'='*130}")
    print_stats_row('Donchian-only (DeathX)', dc_stats, dc_eq)
    print_stats_row(f'SMA200 Rejection (SMA200 bear)', sma_only_stats, sma_only_eq)
    print_stats_row(f'SMA200 Rejection (DeathX)', dc_only_stats, dc_only_eq)

    # ==================================================================
    # SECTION 4: COMBINED DONCHIAN + SMA200 REJECTION
    # ==================================================================
    print_section("SECTION 4: COMBINED DONCHIAN + SMA200 REJECTION")

    # Combined with death cross for Donchian, SMA200-below for rejection
    combined_trades, combined_eq, combined_cap, dc_ent, sma_ent = backtest_combined_shorts(
        coin_data, SHORT_DEFAULT_PARAMS, best_sma_params,
        bear_simple, donchian_bear_filter=bear_strict)
    combined_stats = compute_stats(combined_trades, 'Combined (DC:DeathX + SMA200:SMA200)')

    # Combined with death cross for both
    combined_dc_trades, combined_dc_eq, combined_dc_cap, dc_ent2, sma_ent2 = backtest_combined_shorts(
        coin_data, SHORT_DEFAULT_PARAMS, best_dc_params,
        bear_strict)
    combined_dc_stats = compute_stats(combined_dc_trades, 'Combined (both DeathX)')

    print(f"\n  {'='*130}")
    print(f"  COMBINED RESULTS")
    print(f"  {'='*130}")
    print_stats_row('Donchian-only (DeathX)', dc_stats, dc_eq)
    print_stats_row('Combined (DC:DeathX + SMA200:SMA200)', combined_stats, combined_eq)
    print(f"    → Donchian entries: {dc_ent}, SMA200 entries: {sma_ent}")
    print_stats_row('Combined (both DeathX)', combined_dc_stats, combined_dc_eq)
    print(f"    → Donchian entries: {dc_ent2}, SMA200 entries: {sma_ent2}")

    # Entry type breakdown for best combined
    best_combined_trades = combined_trades if combined_stats.get('total_return_pct', -999) > combined_dc_stats.get('total_return_pct', -999) else combined_dc_trades
    full_exits = [t for t in best_combined_trades if 'Partial' not in t['exit_reason']]
    by_type = {}
    for t in full_exits:
        et = t.get('entry_type', 'donchian')
        if et not in by_type:
            by_type[et] = {'wins': 0, 'losses': 0, 'pnl': 0}
        if t.get('win'):
            by_type[et]['wins'] += 1
        else:
            by_type[et]['losses'] += 1
        by_type[et]['pnl'] += t['pnl_pct']

    print(f"\n  Entry type breakdown (best combined):")
    for et, data in by_type.items():
        total = data['wins'] + data['losses']
        wr = data['wins'] / total * 100 if total > 0 else 0
        print(f"    {et:<25s}: {total:>3d} trades, {wr:.1f}% WR, {data['pnl']:+.1f}% total P&L")

    # ==================================================================
    # SECTION 5: WALK-FORWARD VALIDATION
    # ==================================================================
    print_section("SECTION 5: WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')
    train_data, test_data = split_coin_data(coin_data, cutoff)
    btc_train = train_data.get('BTC-USD')
    btc_test = test_data.get('BTC-USD')

    train_bear_sma = compute_btc_bear_filter(btc_train, require_death_cross=False) if btc_train is not None else {}
    test_bear_sma = compute_btc_bear_filter(btc_test, require_death_cross=False) if btc_test is not None else {}
    train_bear_dc = compute_btc_bear_filter(btc_train, require_death_cross=True) if btc_train is not None else {}
    test_bear_dc = compute_btc_bear_filter(btc_test, require_death_cross=True) if btc_test is not None else {}

    # Walk-forward: SMA200 rejection standalone
    print(f"\n  --- SMA200 Rejection Standalone ---")
    for filter_label, train_filter, test_filter in [
        ('SMA200', train_bear_sma, test_bear_sma),
        ('DeathX', train_bear_dc, test_bear_dc),
    ]:
        # Use top 3 configs
        results_list = sma_results if filter_label == 'SMA200' else dc_results
        for r in results_list[:3]:
            params = r['params']
            train_trades, train_eq, _ = backtest_sma200_rejection(train_data, params, train_filter)
            train_stats = compute_stats(train_trades, f"{params['label']} [{filter_label} train]")

            test_trades, test_eq, _ = backtest_sma200_rejection(test_data, params, test_filter)
            test_stats = compute_stats(test_trades, f"{params['label']} [{filter_label} OOS]")

            print(f"\n  {params['label']} ({filter_label}):")
            print(f"    Train: {train_stats.get('total_trades',0)}t, "
                  f"WR:{train_stats.get('win_rate',0)*100:.1f}%, "
                  f"PF:{train_stats.get('profit_factor',0):.2f}, "
                  f"Ret:{train_stats.get('total_return_pct',0):+.1f}%")
            print(f"    OOS:   {test_stats.get('total_trades',0)}t, "
                  f"WR:{test_stats.get('win_rate',0)*100:.1f}%, "
                  f"PF:{test_stats.get('profit_factor',0):.2f}, "
                  f"Ret:{test_stats.get('total_return_pct',0):+.1f}%")

    # Walk-forward: Combined Donchian + SMA200
    print(f"\n  --- Combined Donchian + SMA200 Rejection ---")
    for filter_label, train_bear, test_bear, train_dc, test_dc in [
        ('Mixed', train_bear_sma, test_bear_sma, train_bear_dc, test_bear_dc),
        ('Both DC', train_bear_dc, test_bear_dc, train_bear_dc, test_bear_dc),
    ]:
        train_trades, train_eq, _, de, se = backtest_combined_shorts(
            train_data, SHORT_DEFAULT_PARAMS, best_sma_params if filter_label == 'Mixed' else best_dc_params,
            train_bear, donchian_bear_filter=train_dc)
        train_stats = compute_stats(train_trades, f"Combined [{filter_label} train]")

        test_trades, test_eq, _, de2, se2 = backtest_combined_shorts(
            test_data, SHORT_DEFAULT_PARAMS, best_sma_params if filter_label == 'Mixed' else best_dc_params,
            test_bear, donchian_bear_filter=test_dc)
        test_stats = compute_stats(test_trades, f"Combined [{filter_label} OOS]")

        print(f"\n  Combined ({filter_label}):")
        print(f"    Train: {train_stats.get('total_trades',0)}t (DC:{de}, SMA:{se}), "
              f"WR:{train_stats.get('win_rate',0)*100:.1f}%, "
              f"PF:{train_stats.get('profit_factor',0):.2f}, "
              f"Ret:{train_stats.get('total_return_pct',0):+.1f}%")
        print(f"    OOS:   {test_stats.get('total_trades',0)}t (DC:{de2}, SMA:{se2}), "
              f"WR:{test_stats.get('win_rate',0)*100:.1f}%, "
              f"PF:{test_stats.get('profit_factor',0):.2f}, "
              f"Ret:{test_stats.get('total_return_pct',0):+.1f}%")

    # Donchian-only WF for comparison
    train_dc_only, train_dc_eq, _, _ = backtest_portfolio_short(
        train_data, SHORT_DEFAULT_PARAMS, train_bear_dc)
    train_dc_stats = compute_stats(train_dc_only, 'Donchian-only [train]')
    test_dc_only, test_dc_eq, _, _ = backtest_portfolio_short(
        test_data, SHORT_DEFAULT_PARAMS, test_bear_dc)
    test_dc_stats = compute_stats(test_dc_only, 'Donchian-only [OOS]')

    print(f"\n  Donchian-only baseline (DeathX):")
    print(f"    Train: {train_dc_stats.get('total_trades',0)}t, "
          f"WR:{train_dc_stats.get('win_rate',0)*100:.1f}%, "
          f"PF:{train_dc_stats.get('profit_factor',0):.2f}, "
          f"Ret:{train_dc_stats.get('total_return_pct',0):+.1f}%")
    print(f"    OOS:   {test_dc_stats.get('total_trades',0)}t, "
          f"WR:{test_dc_stats.get('win_rate',0)*100:.1f}%, "
          f"PF:{test_dc_stats.get('profit_factor',0):.2f}, "
          f"Ret:{test_dc_stats.get('total_return_pct',0):+.1f}%")

    # ==================================================================
    # SECTION 6: COMBINED LONG + SHORT PORTFOLIO
    # ==================================================================
    print_section("SECTION 6: COMBINED LONG + SHORT (Full Portfolio)")

    # Long-only baseline
    long_trades, long_eq, long_cap, long_pyrs = backtest_portfolio_phase3(
        coin_data, LONG_PROD_PARAMS, bull_filter, pyramiding=True)
    long_stats = compute_stats(long_trades, 'Long-Only (Phase 3)')

    # Donchian-only shorts
    short_dc_trades, short_dc_eq, _, _ = backtest_portfolio_short(
        coin_data, SHORT_DEFAULT_PARAMS, bear_strict)
    short_dc_stats = compute_stats(short_dc_trades, 'Short Donchian-only (DeathX)')

    # Combined shorts (best variant)
    best_comb_filter = bear_simple
    best_comb_dc_filter = bear_strict
    best_comb_sma_params = best_sma_params
    comb_short_trades, comb_short_eq, _, _, _ = backtest_combined_shorts(
        coin_data, SHORT_DEFAULT_PARAMS, best_comb_sma_params,
        best_comb_filter, donchian_bear_filter=best_comb_dc_filter)
    comb_short_stats = compute_stats(comb_short_trades, 'Short Combined (DC+SMA200)')

    starting = LONG_PROD_PARAMS['starting_capital']

    # Merge equity curves
    long_by_date = {e['date']: e['equity'] for e in long_eq}
    dc_by_date = {e['date']: e['equity'] for e in short_dc_eq}
    comb_by_date = {e['date']: e['equity'] for e in comb_short_eq}
    all_eq_dates = sorted(set(long_by_date.keys()) | set(dc_by_date.keys()) | set(comb_by_date.keys()))

    combined_dc_eq_full = []
    combined_dual_eq_full = []
    for d in all_eq_dates:
        lv = long_by_date.get(d, starting)
        dcv = dc_by_date.get(d, starting)
        cv = comb_by_date.get(d, starting)
        combined_dc_eq_full.append({'date': d, 'equity': starting + (lv - starting) + (dcv - starting)})
        combined_dual_eq_full.append({'date': d, 'equity': starting + (lv - starting) + (cv - starting)})

    long_dc_stats = compute_stats(long_trades + short_dc_trades, 'Long + Short Donchian')
    long_dual_stats = compute_stats(long_trades + comb_short_trades, 'Long + Short Combined')

    print(f"\n  {'='*130}")
    print(f"  FULL PORTFOLIO COMPARISON")
    print(f"  {'='*130}")
    print_stats_row('Long-Only (Phase 3)', long_stats, long_eq)
    print_stats_row('+ Donchian Shorts (DeathX)', long_dc_stats, combined_dc_eq_full)
    print_stats_row('+ Combined Shorts (DC+SMA200)', long_dual_stats, combined_dual_eq_full)

    # Max drawdowns
    long_dd, _ = compute_max_drawdown(long_eq)
    dc_dd, _ = compute_max_drawdown(combined_dc_eq_full)
    dual_dd, _ = compute_max_drawdown(combined_dual_eq_full)
    print(f"\n  Max Drawdown:")
    print(f"    Long-only:              {long_dd:.1f}%")
    print(f"    Long + Donchian shorts: {dc_dd:.1f}%")
    print(f"    Long + Combined shorts: {dual_dd:.1f}%")

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================
    print_section("FINAL VERDICT — SMA200 REJECTION")

    # Summarize
    print(f"\n  {'='*90}")
    print(f"  STANDALONE SMA200 REJECTION")
    print(f"  {'='*90}")
    if sma_results and sma_results[0]['stats'].get('total_trades', 0) > 0:
        s = sma_results[0]['stats']
        dd, _ = compute_max_drawdown(sma_results[0]['eq'])
        print(f"  Best (SMA200 bear): {sma_results[0]['label']}")
        print(f"    {s.get('total_trades',0):>3d}t  WR:{s.get('win_rate',0)*100:.1f}%  "
              f"PF:{s.get('profit_factor',0):.2f}  Ret:{s.get('total_return_pct',0):+.1f}%  DD:{dd:.1f}%")
    if dc_results and dc_results[0]['stats'].get('total_trades', 0) > 0:
        s = dc_results[0]['stats']
        dd, _ = compute_max_drawdown(dc_results[0]['eq'])
        print(f"  Best (DeathX): {dc_results[0]['label']}")
        print(f"    {s.get('total_trades',0):>3d}t  WR:{s.get('win_rate',0)*100:.1f}%  "
              f"PF:{s.get('profit_factor',0):.2f}  Ret:{s.get('total_return_pct',0):+.1f}%  DD:{dd:.1f}%")

    print(f"\n  {'='*90}")
    print(f"  COMBINED DONCHIAN + SMA200 REJECTION")
    print(f"  {'='*90}")
    cs = combined_stats
    cdd, _ = compute_max_drawdown(combined_eq)
    print(f"  {cs.get('total_trades',0):>3d}t  WR:{cs.get('win_rate',0)*100:.1f}%  "
          f"PF:{cs.get('profit_factor',0):.2f}  Ret:{cs.get('total_return_pct',0):+.1f}%  DD:{cdd:.1f}%")

    print(f"\n  {'='*90}")
    print(f"  FULL PORTFOLIO (LONG + COMBINED SHORTS)")
    print(f"  {'='*90}")
    ls = long_dual_stats
    ldd, _ = compute_max_drawdown(combined_dual_eq_full)
    print(f"  {ls.get('total_trades',0):>3d}t  WR:{ls.get('win_rate',0)*100:.1f}%  "
          f"PF:{ls.get('profit_factor',0):.2f}  Ret:{ls.get('total_return_pct',0):+.1f}%  DD:{ldd:.1f}%")

    # Verdict
    donchian_ret = dc_stats.get('total_return_pct', 0)
    combined_ret = combined_stats.get('total_return_pct', 0)
    improvement = combined_ret - donchian_ret

    print(f"\n  VERDICT:")
    if combined_ret > donchian_ret and combined_stats.get('profit_factor', 0) >= 1.0:
        print(f"  SMA200 REJECTION ADDS VALUE: +{improvement:.1f}% over Donchian-only shorts")
        print(f"  Recommend deploying combined short entry signals.")
    elif combined_stats.get('total_trades', 0) == 0:
        print(f"  NO TRADES GENERATED — signal conditions may be too strict.")
        print(f"  Try relaxing proximity_pct or volume_mult.")
    else:
        print(f"  SMA200 REJECTION DID NOT IMPROVE: {improvement:+.1f}% vs Donchian-only")
        print(f"  Stick with Donchian-only shorts for now.")

    print(f"\n{'='*100}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
