"""Return Optimization Research — Path to $120K/Year

Sweeps every lever available to maximize returns:
  1. Position limits: 4, 6, 8
  2. Risk per trade: 2%, 3%, 4%, 5%
  3. Futures leverage: 1x, 1.5x, 2x, 3x
  4. Fee mode: spot+futures vs all-futures
  5. Progressive risk scaling: protect the seed, then play with house money

All configs run on $30K starting capital, 4-year backtest, tax-aware.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

from backtest_donchian_daily import (
    fetch_all_coins, calculate_indicators, DEFAULT_PARAMS, COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_shorts import (
    compute_btc_bear_filter, calculate_short_indicators,
    SHORT_DEFAULT_PARAMS, SHORT_COINS,
)
from backtest_trimode import (
    compute_total_tax, prepare_mode_data, compute_total_equity,
    has_coin_conflict, compute_stats_from_trades,
    SPOT_LONG_COINS, FUTURES_LONG_COINS, SHORT_COINS_LIST, ALL_COINS,
    PYRAMID_GAIN_PCT, PYRAMID_RISK_PCT,
)
from backtest_walkforward import compute_max_drawdown, print_section

# ============================================================================
# CONFIGURATION
# ============================================================================

STARTING_CAPITAL = 30_000.0

# Base params (used as templates)
LONG_BASE = {
    **DEFAULT_PARAMS,
    'atr_mult': 4.0,
    'tp1_pct': 10.0, 'tp2_pct': 20.0,
    'tp1_fraction': 0.25, 'tp2_fraction': 0.25,
    'rsi_blowoff': 80, 'volume_blowoff': 3.0, 'atr_mult_tight': 1.5,
}

SHORT_BASE = {
    **SHORT_DEFAULT_PARAMS,
    'donchian_period': 10, 'exit_period': 15,
    'atr_mult': 2.0, 'volume_mult': 2.0,
    'fee_pct': 0.06, 'slippage_pct': 0.05,
    'funding_rate_daily': -0.03,
    'emergency_stop_pct': 15.0, 'max_hold_days': 30,
    'rsi_blowoff': 20, 'volume_blowoff': 3.0, 'atr_mult_tight': 1.0,
}

# Spot fees vs futures fees
SPOT_COST = (0.45 + 0.05) / 100   # 0.50% per side
FUTURES_COST = (0.06 + 0.05) / 100  # 0.11% per side
FUNDING_DAILY = -0.03 / 100         # shorts receive


# ============================================================================
# PROGRESSIVE RISK PROFILES
# ============================================================================

def get_progressive_params(equity, starting, profile):
    """Return (risk_pct, max_pos, leverage) based on equity growth and profile.

    Profiles scale risk as equity grows beyond the initial investment.
    """
    growth = equity / starting  # 1.0 = flat, 2.0 = doubled

    if profile == 'conservative':
        return 0.02, 4, 1.0

    elif profile == 'moderate_scale':
        # At 1.5x: loosen to 3%/6/1.5x
        if growth < 1.5:
            return 0.02, 4, 1.0
        else:
            return 0.03, 6, 1.5

    elif profile == 'aggressive_scale':
        # At 2x: loosen to 4%/8/2x
        if growth < 2.0:
            return 0.02, 4, 1.0
        else:
            return 0.04, 8, 2.0

    elif profile == 'house_money':
        # At 2x: full send with house money
        if growth < 2.0:
            return 0.02, 4, 1.0
        else:
            return 0.05, 8, 3.0

    elif profile == 'stepped':
        # Gradual 4-step escalation
        if growth < 1.5:
            return 0.02, 4, 1.0
        elif growth < 2.0:
            return 0.03, 6, 1.5
        elif growth < 3.0:
            return 0.04, 8, 2.0
        else:
            return 0.05, 8, 3.0

    elif profile == 'early_aggressive':
        # Start aggressive immediately, max out after 1.5x
        if growth < 1.5:
            return 0.04, 6, 2.0
        else:
            return 0.05, 8, 3.0

    elif profile == 'futures_only_stepped':
        # All-futures + stepped scaling
        if growth < 1.5:
            return 0.03, 6, 1.5
        elif growth < 2.0:
            return 0.04, 8, 2.0
        else:
            return 0.05, 8, 3.0

    elif profile == 'kelly_half':
        # Half-Kelly: risk scales with win rate/PF (approximated)
        # Base half-Kelly for our system: ~4% at 64% WR / 2.7 PF
        if growth < 1.5:
            return 0.03, 6, 1.0
        elif growth < 2.0:
            return 0.04, 6, 1.5
        else:
            return 0.05, 8, 2.0

    return 0.02, 4, 1.0


PROGRESSIVE_PROFILES = [
    'conservative', 'moderate_scale', 'aggressive_scale',
    'house_money', 'stepped', 'early_aggressive',
    'futures_only_stepped', 'kelly_half',
]


# ============================================================================
# PARAMETRIC TRI-MODE BACKTEST ENGINE
# ============================================================================

def backtest_parametric(coin_data, bull_filter, bear_filter,
                        sl_prepared, sl_lookups, sl_prev,
                        fl_prepared, fl_lookups, fl_prev,
                        sh_prepared, sh_lookups, sh_prev,
                        all_dates,
                        starting_capital=30000.0,
                        risk_pct=0.02, max_positions=4, leverage=1.0,
                        all_futures=False, enable_taxes=True,
                        progressive_profile=None):
    """Parametric tri-mode backtest — accepts all config as arguments.

    If all_futures=True, spot longs use futures fee structure instead.
    If progressive_profile is set, risk_pct/max_positions/leverage are overridden dynamically.
    """
    all_lookups = {'SPOT_LONG': sl_lookups, 'FUTURES_LONG': fl_lookups, 'SHORT': sh_lookups}
    all_prev = {'SPOT_LONG': sl_prev, 'FUTURES_LONG': fl_prev, 'SHORT': sh_prev}

    sl_cost = FUTURES_COST if all_futures else SPOT_COST
    fl_cost = FUTURES_COST
    sh_cost = FUTURES_COST

    def get_cost(mode):
        return sl_cost if mode == 'SPOT_LONG' else fl_cost if mode == 'FUTURES_LONG' else sh_cost

    capital = starting_capital
    positions = {}
    trades = []
    equity_curve = []
    pyramid_adds = 0
    yearly_realized = {}
    yearly_taxes = {}
    current_year = None

    def record_realized(pnl_usd, year):
        yearly_realized[year] = yearly_realized.get(year, 0.0) + pnl_usd

    for date in all_dates:
        year = date.year

        # Year-end tax deduction
        if current_year is not None and year != current_year and enable_taxes:
            gains = yearly_realized.get(current_year, 0.0)
            fed, state, total = compute_total_tax(gains)
            capital -= total
            yearly_taxes[current_year] = {
                'gross_gains': gains, 'federal': fed, 'state': state, 'total': total,
            }
        current_year = year

        # Dynamic params from progressive profile or static
        if progressive_profile:
            total_eq = compute_total_equity(capital, positions, all_lookups, date)
            cur_risk, cur_max_pos, cur_leverage = get_progressive_params(
                total_eq, starting_capital, progressive_profile)
        else:
            cur_risk, cur_max_pos, cur_leverage = risk_pct, max_positions, leverage

        is_bull = bull_filter.get(date, False)
        is_bear = bear_filter.get(date, False)

        # ============================================================
        # 1. EXITS
        # ============================================================
        keys_to_close = []
        for key, pos in list(positions.items()):
            mode = pos['mode']
            symbol = pos['symbol']
            lookups = all_lookups[mode]
            prev_lk = all_prev[mode]
            cost = get_cost(mode)

            row = lookups.get(symbol, {}).get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_atr = float(row['atr'])
            exit_reason = None

            if mode == 'SHORT':
                pos['low_watermark'] = min(pos['low_watermark'], float(row['low']))
                pos['hold_days'] += 1

                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ratio = float(row['volume']) / vol_sma
                is_bounce = (volume_ratio > SHORT_BASE['volume_blowoff']
                             and float(row['rsi']) < SHORT_BASE['rsi_blowoff'])
                stop_mult = SHORT_BASE['atr_mult_tight'] if is_bounce else SHORT_BASE['atr_mult']

                if current_close >= pos['low_watermark'] + (stop_mult * current_atr):
                    exit_reason = f'Trailing stop ({stop_mult}x ATR)'

                prev_row = prev_lk.get(symbol, {}).get(date)
                if not exit_reason and prev_row is not None and pd.notna(prev_row.get('exit_high')):
                    if current_close > float(prev_row['exit_high']):
                        exit_reason = 'Donchian exit'

                if not exit_reason and current_close >= pos['entry_price'] * (1 + SHORT_BASE['emergency_stop_pct'] / 100):
                    exit_reason = 'Emergency stop'

                if not exit_reason and pos['hold_days'] >= SHORT_BASE['max_hold_days']:
                    exit_reason = 'Max hold'

                # Partial TP (shorts)
                if not exit_reason:
                    gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= SHORT_BASE['tp1_pct']:
                        p_exit = current_close * (1 + cost)
                        p_pnl = ((pos['entry_price'] - p_exit) / pos['entry_price']) * 100 + FUNDING_DAILY * pos['hold_days'] * 100
                        p_size = pos['size_usd'] * SHORT_BASE['tp1_fraction']
                        p_gain = p_size * (p_pnl / 100)
                        capital += p_size + p_gain
                        record_realized(p_gain, year)
                        pos['size_usd'] -= p_size
                        pos['partials_taken'] = 1
                        trades.append({'symbol': symbol, 'mode': 'SHORT', 'pnl_pct': p_pnl,
                                       'size_usd': p_size, 'win': True, 'exit_reason': 'TP1',
                                       'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                       'entry_price': pos['entry_price'], 'exit_price': p_exit})
                    elif pos['partials_taken'] == 1 and gain_pct >= SHORT_BASE['tp2_pct']:
                        p_exit = current_close * (1 + cost)
                        p_pnl = ((pos['entry_price'] - p_exit) / pos['entry_price']) * 100 + FUNDING_DAILY * pos['hold_days'] * 100
                        p_size = pos['size_usd'] * SHORT_BASE['tp2_fraction']
                        p_gain = p_size * (p_pnl / 100)
                        capital += p_size + p_gain
                        record_realized(p_gain, year)
                        pos['size_usd'] -= p_size
                        pos['partials_taken'] = 2
                        trades.append({'symbol': symbol, 'mode': 'SHORT', 'pnl_pct': p_pnl,
                                       'size_usd': p_size, 'win': True, 'exit_reason': 'TP2',
                                       'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                       'entry_price': pos['entry_price'], 'exit_price': p_exit})

                if exit_reason:
                    exit_price = current_close * (1 + cost)
                    price_pnl = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
                    fund_pnl = FUNDING_DAILY * pos['hold_days'] * 100
                    total_pnl = price_pnl + fund_pnl
                    pnl_usd = pos['size_usd'] * (total_pnl / 100)
                    capital += pos['size_usd'] + pnl_usd
                    record_realized(pnl_usd, year)
                    trades.append({'symbol': symbol, 'mode': 'SHORT', 'pnl_pct': total_pnl,
                                   'size_usd': pos['size_usd'], 'win': total_pnl > 0,
                                   'exit_reason': exit_reason,
                                   'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                   'entry_price': pos['entry_price'], 'exit_price': exit_price})
                    keys_to_close.append(key)

            else:
                # Long exit (spot or futures)
                params = LONG_BASE
                pos['high_watermark'] = max(pos['high_watermark'], float(row['high']))

                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ratio = float(row['volume']) / vol_sma
                is_blowoff = (volume_ratio > params['volume_blowoff']
                              and float(row['rsi']) > params['rsi_blowoff'])
                stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

                if current_close <= pos['high_watermark'] - (stop_mult * current_atr):
                    exit_reason = f'Trailing stop ({stop_mult}x ATR)'

                prev_row = prev_lk.get(symbol, {}).get(date)
                if not exit_reason and prev_row is not None and pd.notna(prev_row['exit_low']):
                    if current_close < float(prev_row['exit_low']):
                        exit_reason = 'Donchian exit'

                if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                    exit_reason = 'Emergency stop'

                # Partial TP (longs)
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                        p_price = current_close * (1 - cost)
                        p_pnl = ((p_price - pos['entry_price']) / pos['entry_price']) * 100
                        p_size = pos['size_usd'] * params['tp1_fraction']
                        p_gain = p_size * (p_pnl / 100)
                        capital += p_size + p_gain
                        record_realized(p_gain, year)
                        pos['size_usd'] -= p_size
                        pos['partials_taken'] = 1
                        trades.append({'symbol': symbol, 'mode': mode, 'pnl_pct': p_pnl,
                                       'size_usd': p_size, 'win': True, 'exit_reason': 'TP1',
                                       'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                       'entry_price': pos['entry_price'], 'exit_price': p_price})
                    elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                        p_price = current_close * (1 - cost)
                        p_pnl = ((p_price - pos['entry_price']) / pos['entry_price']) * 100
                        p_size = pos['size_usd'] * params['tp2_fraction']
                        p_gain = p_size * (p_pnl / 100)
                        capital += p_size + p_gain
                        record_realized(p_gain, year)
                        pos['size_usd'] -= p_size
                        pos['partials_taken'] = 2
                        trades.append({'symbol': symbol, 'mode': mode, 'pnl_pct': p_pnl,
                                       'size_usd': p_size, 'win': True, 'exit_reason': 'TP2',
                                       'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                       'entry_price': pos['entry_price'], 'exit_price': p_price})

                if exit_reason:
                    exit_price = current_close * (1 - cost)
                    pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                    pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                    capital += pos['size_usd'] + pnl_usd
                    record_realized(pnl_usd, year)
                    trades.append({'symbol': symbol, 'mode': mode, 'pnl_pct': pnl_pct,
                                   'size_usd': pos['size_usd'], 'win': pnl_pct > 0,
                                   'exit_reason': exit_reason,
                                   'entry_time': pos['entry_time'], 'exit_time': row['time'],
                                   'entry_price': pos['entry_price'], 'exit_price': exit_price})
                    keys_to_close.append(key)

        for k in keys_to_close:
            del positions[k]

        # ============================================================
        # 2. PYRAMIDING (longs, bull filter)
        # ============================================================
        if is_bull:
            for key, pos in list(positions.items()):
                mode = pos['mode']
                if mode == 'SHORT' or pos.get('pyramided'):
                    continue
                symbol = pos['symbol']
                cost = get_cost(mode)
                lookups = all_lookups[mode]
                prev_lk = all_prev[mode]

                row = lookups.get(symbol, {}).get(date)
                prev_row = prev_lk.get(symbol, {}).get(date)
                if row is None or prev_row is None:
                    continue

                current_close = float(row['close'])
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                new_high = pd.notna(prev_row['donchian_high']) and current_close > float(prev_row['donchian_high'])

                if gain_pct >= PYRAMID_GAIN_PCT and new_high:
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    add_risk = total_equity * PYRAMID_RISK_PCT
                    atr_val = float(row['atr'])
                    stop_pct = (LONG_BASE['atr_mult'] * atr_val) / current_close
                    add_size = (add_risk / stop_pct) if stop_pct > 0 else total_equity * 0.05
                    add_size = min(add_size, capital * 0.50)
                    if add_size >= 100:
                        capital -= add_size
                        pos['size_usd'] += add_size
                        pos['pyramided'] = True
                        pyramid_adds += 1

        # ============================================================
        # 3. NEW ENTRIES
        # ============================================================

        # Spot long entries (or all-futures longs for slot 1)
        if len(positions) < cur_max_pos and is_bull:
            for symbol in SPOT_LONG_COINS:
                if symbol not in sl_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= cur_max_pos:
                    break
                row = sl_lookups.get(symbol, {}).get(date)
                prev_row = sl_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row['donchian_high']):
                    continue
                cc = float(row['close'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                if cc > float(prev_row['donchian_high']) and float(row['volume']) > LONG_BASE['volume_mult'] * vol_sma and cc > float(row['ema_21']):
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    cost = sl_cost
                    entry_price = cc * (1 + cost)
                    stop_pct = (LONG_BASE['atr_mult'] * float(row['atr'])) / entry_price
                    if stop_pct <= 0:
                        continue
                    size = (total_equity * cur_risk) / stop_pct
                    size = min(size, capital * 0.95)
                    if size < 100:
                        continue
                    capital -= size
                    positions[f'SPOT_LONG:{symbol}'] = {
                        'mode': 'SPOT_LONG', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0, 'size_usd': size, 'pyramided': False,
                    }

        # Futures long entries
        if len(positions) < cur_max_pos and is_bull:
            for symbol in FUTURES_LONG_COINS:
                if symbol not in fl_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= cur_max_pos:
                    break
                row = fl_lookups.get(symbol, {}).get(date)
                prev_row = fl_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row['donchian_high']):
                    continue
                cc = float(row['close'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                if cc > float(prev_row['donchian_high']) and float(row['volume']) > LONG_BASE['volume_mult'] * vol_sma and cc > float(row['ema_21']):
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    entry_price = cc * (1 + fl_cost)
                    stop_pct = (LONG_BASE['atr_mult'] * float(row['atr'])) / entry_price
                    if stop_pct <= 0:
                        continue
                    size = (total_equity * cur_risk) / stop_pct
                    size *= cur_leverage  # LEVERAGE
                    size = min(size, capital * 0.95)
                    if size < 100:
                        continue
                    capital -= size
                    positions[f'FUTURES_LONG:{symbol}'] = {
                        'mode': 'FUTURES_LONG', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0, 'size_usd': size, 'pyramided': False,
                    }

        # Short entries
        if len(positions) < cur_max_pos and is_bear:
            for symbol in SHORT_COINS_LIST:
                if symbol not in sh_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= cur_max_pos:
                    break
                row = sh_lookups.get(symbol, {}).get(date)
                prev_row = sh_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row.get('donchian_low')):
                    continue
                cc = float(row['close'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                if cc < float(prev_row['donchian_low']) and float(row['volume']) > SHORT_BASE['volume_mult'] * vol_sma and cc < float(row['ema_21']):
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    entry_price = cc * (1 - sh_cost)
                    stop_pct = (SHORT_BASE['atr_mult'] * float(row['atr'])) / entry_price
                    if stop_pct <= 0:
                        continue
                    size = (total_equity * cur_risk) / stop_pct
                    size = min(size, capital * 0.95)
                    if size < 100:
                        continue
                    capital -= size
                    positions[f'SHORT:{symbol}'] = {
                        'mode': 'SHORT', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'low_watermark': float(row['low']),
                        'partials_taken': 0, 'size_usd': size,
                        'pyramided': False, 'hold_days': 0,
                    }

        # Equity tracking
        total_equity = compute_total_equity(capital, positions, all_lookups, date)
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining
    for key, pos in list(positions.items()):
        mode = pos['mode']
        symbol = pos['symbol']
        cost = get_cost(mode)
        prep = sh_prepared if mode == 'SHORT' else fl_prepared if mode == 'FUTURES_LONG' else sl_prepared
        if symbol not in prep:
            continue
        last = prep[symbol].iloc[-1]
        if mode == 'SHORT':
            exit_price = float(last['close']) * (1 + cost)
            total_pnl = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100 + FUNDING_DAILY * pos['hold_days'] * 100
        else:
            exit_price = float(last['close']) * (1 - cost)
            total_pnl = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (total_pnl / 100)
        capital += pos['size_usd'] + pnl_usd
        record_realized(pnl_usd, current_year)
        trades.append({'symbol': symbol, 'mode': mode, 'pnl_pct': total_pnl,
                       'size_usd': pos['size_usd'], 'win': total_pnl > 0,
                       'exit_reason': 'End of backtest',
                       'entry_time': pos['entry_time'], 'exit_time': last['time'],
                       'entry_price': pos['entry_price'], 'exit_price': exit_price})

    # Final year taxes
    if enable_taxes and current_year is not None:
        gains = yearly_realized.get(current_year, 0.0)
        fed, state, total = compute_total_tax(gains)
        capital -= total
        yearly_taxes[current_year] = {
            'gross_gains': gains, 'federal': fed, 'state': state, 'total': total,
        }

    # Max drawdown
    equities = [pt['equity'] for pt in equity_curve]
    max_dd = 0
    if equities:
        peak = equities[0]
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

    real_trades = [t for t in trades if t['pnl_pct'] != 0]
    stats = compute_stats_from_trades(real_trades)

    return {
        'trades': trades,
        'real_trades': len(real_trades),
        'equity_curve': equity_curve,
        'final_capital': capital,
        'yearly_taxes': yearly_taxes,
        'yearly_realized': yearly_realized,
        'max_dd': max_dd,
        'wr': stats['wr'],
        'pf': stats['pf'],
        'gross_pnl': stats['gross_pnl'],
        'total_tax': sum(tx['total'] for tx in yearly_taxes.values()),
        'net_pnl': stats['gross_pnl'] - sum(tx['total'] for tx in yearly_taxes.values()),
        'pyramid_adds': pyramid_adds,
    }


# ============================================================================
# MAIN — SWEEP + REPORT
# ============================================================================

def main():
    print("=" * 120)
    print("RETURN OPTIMIZATION RESEARCH — PATH TO $120K/YEAR")
    print(f"  Starting capital: ${STARTING_CAPITAL:,.0f} | 4-year backtest | Tax-aware | All combinations")
    print("=" * 120)

    # Fetch data
    print(f"\nFetching daily data for {len(ALL_COINS)} coins...")
    coin_data = fetch_all_coins(coins=ALL_COINS, years=4)
    print(f"Loaded {len(coin_data)} coins\n")

    btc_df = coin_data.get('BTC-USD')
    bull_filter = compute_btc_bull_filter(btc_df)
    bear_filter = compute_btc_bear_filter(btc_df, require_death_cross=True)

    # Prepare data once (reused by all configs)
    print("Preparing indicators...")
    sl_params = {**LONG_BASE, 'fee_pct': 0.45, 'slippage_pct': 0.05}
    fl_params = {**LONG_BASE, 'fee_pct': 0.06, 'slippage_pct': 0.05}

    sl_prepared, sl_lookups, sl_prev = prepare_mode_data(coin_data, SPOT_LONG_COINS, sl_params, is_short=False)
    fl_prepared, fl_lookups, fl_prev = prepare_mode_data(coin_data, FUTURES_LONG_COINS, fl_params, is_short=False)
    sh_prepared, sh_lookups, sh_prev = prepare_mode_data(coin_data, SHORT_COINS_LIST, SHORT_BASE, is_short=True)

    all_dates = set()
    for mode_prep in [sl_prepared, fl_prepared, sh_prepared]:
        for df in mode_prep.values():
            all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    common_args = dict(
        coin_data=coin_data, bull_filter=bull_filter, bear_filter=bear_filter,
        sl_prepared=sl_prepared, sl_lookups=sl_lookups, sl_prev=sl_prev,
        fl_prepared=fl_prepared, fl_lookups=fl_lookups, fl_prev=fl_prev,
        sh_prepared=sh_prepared, sh_lookups=sh_lookups, sh_prev=sh_prev,
        all_dates=all_dates, starting_capital=STARTING_CAPITAL,
    )

    # ================================================================
    # PHASE 1: STATIC PARAMETER SWEEP
    # ================================================================
    print_section("PHASE 1: STATIC PARAMETER SWEEP")
    print("  Sweeping: positions [4,6,8] x risk [2%,3%,4%,5%] x leverage [1x,1.5x,2x,3x] x fees [mixed,all-futures]")

    results = []
    total_combos = 3 * 4 * 4 * 2
    combo_num = 0

    for max_pos in [4, 6, 8]:
        for risk in [0.02, 0.03, 0.04, 0.05]:
            for lev in [1.0, 1.5, 2.0, 3.0]:
                for all_fut in [False, True]:
                    combo_num += 1
                    if combo_num % 16 == 0:
                        print(f"    [{combo_num}/{total_combos}]...", flush=True)

                    r = backtest_parametric(
                        **common_args,
                        risk_pct=risk, max_positions=max_pos,
                        leverage=lev, all_futures=all_fut,
                    )
                    fee_label = 'all-fut' if all_fut else 'mixed'
                    label = f"P{max_pos}_R{risk*100:.0f}%_L{lev}x_{fee_label}"
                    results.append({
                        'label': label,
                        'max_pos': max_pos,
                        'risk': risk * 100,
                        'leverage': lev,
                        'fees': fee_label,
                        'trades': r['real_trades'],
                        'wr': r['wr'],
                        'pf': r['pf'],
                        'gross': r['gross_pnl'],
                        'tax': r['total_tax'],
                        'net': r['net_pnl'],
                        'net_pct': r['net_pnl'] / STARTING_CAPITAL * 100,
                        'max_dd': r['max_dd'],
                        'final': r['final_capital'],
                        'yearly_taxes': r['yearly_taxes'],
                        'equity_curve': r['equity_curve'],
                    })

    # Sort by net return
    results.sort(key=lambda x: x['net'], reverse=True)

    print(f"\n  TOP 20 STATIC CONFIGS (by net return after tax)")
    print(f"  {'#':>3} {'Config':<30} {'Trades':>7} {'WR':>6} {'PF':>6} {'Gross':>11} {'Tax':>9} {'Net':>11} {'Net%':>7} {'MaxDD':>7} {'Final':>12}")
    print(f"  {'---':>3} {'-----':.<30} {'------':>7} {'--':>6} {'--':>6} {'-----':>11} {'---':>9} {'---':>11} {'----':>7} {'-----':>7} {'-----':>12}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} {r['label']:<30} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['gross']:>+10,.0f} {r['tax']:>9,.0f} {r['net']:>+10,.0f} {r['net_pct']:>+6.1f}% {r['max_dd']:>6.1f}% ${r['final']:>11,.0f}")

    # Top 10 risk-adjusted
    risk_adj = sorted([r for r in results if r['max_dd'] > 0],
                      key=lambda x: x['net'] / x['max_dd'], reverse=True)
    print(f"\n  TOP 10 RISK-ADJUSTED (net return / max DD)")
    print(f"  {'#':>3} {'Config':<30} {'Net':>11} {'MaxDD':>7} {'Ratio':>7} {'Final':>12}")
    print(f"  {'---':>3} {'-----':.<30} {'---':>11} {'-----':>7} {'-----':>7} {'-----':>12}")
    for i, r in enumerate(risk_adj[:10]):
        ratio = r['net'] / r['max_dd']
        print(f"  {i+1:>3} {r['label']:<30} {r['net']:>+10,.0f} {r['max_dd']:>6.1f}% {ratio:>6.1f}x ${r['final']:>11,.0f}")

    # ================================================================
    # PHASE 2: PROGRESSIVE RISK PROFILES
    # ================================================================
    print_section("PHASE 2: PROGRESSIVE RISK PROFILES")
    print("  Testing dynamic risk scaling based on equity growth")

    prog_results = []
    for profile in PROGRESSIVE_PROFILES:
        print(f"    Running {profile}...", flush=True)
        r = backtest_parametric(
            **common_args,
            progressive_profile=profile,
        )
        prog_results.append({
            'profile': profile,
            'trades': r['real_trades'],
            'wr': r['wr'],
            'pf': r['pf'],
            'gross': r['gross_pnl'],
            'tax': r['total_tax'],
            'net': r['net_pnl'],
            'net_pct': r['net_pnl'] / STARTING_CAPITAL * 100,
            'max_dd': r['max_dd'],
            'final': r['final_capital'],
            'yearly_taxes': r['yearly_taxes'],
            'equity_curve': r['equity_curve'],
        })

    prog_results.sort(key=lambda x: x['net'], reverse=True)

    print(f"\n  PROGRESSIVE PROFILES (ranked by net return)")
    print(f"  {'Profile':<25} {'Trades':>7} {'WR':>6} {'PF':>6} {'Gross':>11} {'Tax':>9} {'Net':>11} {'Net%':>7} {'MaxDD':>7} {'Final':>12}")
    print(f"  {'-------':<25} {'------':>7} {'--':>6} {'--':>6} {'-----':>11} {'---':>9} {'---':>11} {'----':>7} {'-----':>7} {'-----':>12}")
    for r in prog_results:
        print(f"  {r['profile']:<25} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['gross']:>+10,.0f} {r['tax']:>9,.0f} {r['net']:>+10,.0f} {r['net_pct']:>+6.1f}% {r['max_dd']:>6.1f}% ${r['final']:>11,.0f}")

    # ================================================================
    # PHASE 3: YEAR-BY-YEAR FOR TOP 5 OVERALL
    # ================================================================
    print_section("PHASE 3: YEAR-BY-YEAR FOR TOP 5 CONFIGS")

    # Combine static + progressive, pick top 5
    all_ranked = []
    for r in results:
        all_ranked.append({**r, 'name': r['label']})
    for r in prog_results:
        all_ranked.append({**r, 'name': f"PROG:{r['profile']}"})
    all_ranked.sort(key=lambda x: x['net'], reverse=True)

    for rank, cfg in enumerate(all_ranked[:5], 1):
        print(f"\n  #{rank}: {cfg['name']} — Net: +${cfg['net']:,.0f} ({cfg['net_pct']:+.1f}%), DD: {cfg['max_dd']:.1f}%, Final: ${cfg['final']:,.0f}")
        yt = cfg['yearly_taxes']
        eq = cfg['equity_curve']

        # EOY equity from equity curve
        eoy = {}
        for pt in eq:
            eoy[pt['date'].year] = pt['equity']

        years = sorted(yt.keys())
        print(f"    {'Year':<6} {'Gross':>10} {'Tax':>9} {'Net':>10} {'EOY Equity':>13}")
        for yr in years:
            tx = yt[yr]
            net_yr = tx['gross_gains'] - tx['total']
            print(f"    {yr:<6} {tx['gross_gains']:>+9,.0f} {tx['total']:>9,.0f} {net_yr:>+9,.0f} ${eoy.get(yr, 0):>12,.0f}")

    # ================================================================
    # PHASE 4: PATH TO $120K/YEAR
    # ================================================================
    print_section("PHASE 4: PATH TO $120K/YEAR INCOME")

    best = all_ranked[0]
    print(f"  Best config: {best['name']}")
    print(f"  4-year net return: +${best['net']:,.0f} ({best['net_pct']:+.1f}%)")
    print(f"  Annualized: ~{(best['net_pct'] / 4):+.1f}% per year")
    print(f"  Final equity: ${best['final']:,.0f}")

    # Project forward: if we compound at the avg annual rate, when do we hit $120K/year
    avg_annual_rate = ((best['final'] / STARTING_CAPITAL) ** (1/4)) - 1
    print(f"  CAGR: {avg_annual_rate * 100:.1f}%")

    equity = STARTING_CAPITAL
    print(f"\n  PROJECTION (compounding at {avg_annual_rate * 100:.1f}% CAGR):")
    print(f"  {'Year':>6} {'Start Equity':>14} {'Gross Gain':>12} {'Tax (~15%)':>12} {'Net Gain':>12} {'End Equity':>14}")
    for yr in range(1, 11):
        gross_gain = equity * avg_annual_rate
        tax_est = gross_gain * 0.15 if gross_gain > 15000 else 0
        net_gain = gross_gain - tax_est
        end_eq = equity + net_gain
        marker = ""
        if net_gain >= 120000:
            marker = " <-- TARGET"
        print(f"  {yr:>6} ${equity:>13,.0f} ${gross_gain:>11,.0f} ${tax_est:>11,.0f} ${net_gain:>11,.0f} ${end_eq:>13,.0f}{marker}")
        if net_gain >= 120000 and yr > 1:
            break
        equity = end_eq

    print()


if __name__ == '__main__':
    main()
