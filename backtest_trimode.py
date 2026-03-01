"""Tri-Mode Combined Backtest — Spot Long + Futures Long + Short

Shared equity pool, compounding, progressive tax deductions.

Runs 3 modes simultaneously on a single daily bar loop:
  - Spot Long: Donchian breakout, 4x ATR trailing, pyramiding (bull filter)
  - Futures Long: Same signals, lower fees (bull filter)
  - Futures Short: Inverted Donchian, 2x ATR, death cross filter

Shared constraints:
  - Max 4 positions total across all modes
  - No same-coin conflicts (can't hold BTC spot long + BTC futures long)
  - Progressive US federal + CA state taxes deducted from equity at year-end

Usage:
  cd C:\\ResearchAgent && venv/Scripts/python backtest_trimode.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

from backtest_donchian_daily import (
    fetch_all_coins, calculate_indicators, DEFAULT_PARAMS, COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_shorts import (
    compute_btc_bear_filter, calculate_short_indicators,
    SHORT_DEFAULT_PARAMS, SHORT_COINS,
)
from backtest_walkforward import (
    compute_sharpe, compute_max_drawdown, print_section,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SPOT_LONG_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Spot Long',
    'atr_mult': 4.0,          # Phase 3 production
    'fee_pct': 0.45,          # spot Coinbase taker
    'slippage_pct': 0.05,
}

FUTURES_LONG_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Futures Long',
    'atr_mult': 4.0,
    'fee_pct': 0.06,          # CFM perps
    'slippage_pct': 0.05,
}

SHORT_PARAMS = {
    **SHORT_DEFAULT_PARAMS,
    'label': 'Short',
    'donchian_period': 10,     # death cross best: L10_A2.0_E15_V2.0
    'exit_period': 15,
    'atr_mult': 2.0,
    'volume_mult': 2.0,
    'fee_pct': 0.06,
    'slippage_pct': 0.05,
    'funding_rate_daily': -0.03,
    'emergency_stop_pct': 15.0,
    'max_hold_days': 30,
}

# Coin universes (match production bot)
SPOT_LONG_COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
                   'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD']
FUTURES_LONG_COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
                      'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD']
SHORT_COINS_LIST = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
                    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD']
ALL_COINS = sorted(set(SPOT_LONG_COINS + FUTURES_LONG_COINS + SHORT_COINS_LIST))

# Pyramiding
PYRAMID_GAIN_PCT = 15.0
PYRAMID_RISK_PCT = 0.01

MAX_POSITIONS = 4
RISK_PCT = 0.02  # 2% risk per trade

STARTING_CAPITALS = [10_000, 20_000, 30_000]


# ============================================================================
# TAX COMPUTATION — Progressive US Federal + CA State
# ============================================================================

# Federal brackets (2025/2026 single filer)
FEDERAL_STD_DEDUCTION = 15_000
FEDERAL_BRACKETS = [
    (11_925,  0.10),
    (48_475,  0.12),
    (103_350, 0.22),
    (197_300, 0.24),
    (250_525, 0.32),
    (626_350, 0.35),
    (float('inf'), 0.37),
]

# California brackets (single filer)
CA_EXEMPTION_CREDIT = 144
CA_BRACKETS = [
    (10_412,  0.01),
    (24_684,  0.02),
    (38_959,  0.04),
    (54_081,  0.06),
    (68_350,  0.08),
    (349_137, 0.093),
    (418_961, 0.103),
    (698_271, 0.113),
    (float('inf'), 0.123),
]


def _apply_brackets(income, brackets):
    """Apply progressive tax brackets to taxable income."""
    if income <= 0:
        return 0.0
    tax = 0.0
    prev_limit = 0.0
    for limit, rate in brackets:
        taxable_in_bracket = min(income, limit) - prev_limit
        if taxable_in_bracket <= 0:
            break
        tax += taxable_in_bracket * rate
        prev_limit = limit
    return tax


def compute_federal_tax(gross_income):
    """Federal income tax for single filer with standard deduction."""
    taxable = gross_income - FEDERAL_STD_DEDUCTION
    return _apply_brackets(taxable, FEDERAL_BRACKETS)


def compute_ca_tax(gross_income):
    """California state income tax (no standard deduction, exemption credit)."""
    tax = _apply_brackets(gross_income, CA_BRACKETS)
    return max(0.0, tax - CA_EXEMPTION_CREDIT)


def compute_total_tax(realized_gains):
    """Combined federal + CA tax on realized trading gains."""
    if realized_gains <= 0:
        return 0.0, 0.0, 0.0
    federal = compute_federal_tax(realized_gains)
    state = compute_ca_tax(realized_gains)
    return federal, state, federal + state


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_mode_data(coin_data, coins, params, is_short=False):
    """Prepare indicators and lookups for one trading mode."""
    prepared = {}
    for symbol in coins:
        df = coin_data.get(symbol)
        if df is None:
            continue
        if is_short:
            df_ind = calculate_short_indicators(df, params)
            drop_cols = ['donchian_high', 'donchian_low', 'atr', 'volume_sma', 'ema_21', 'rsi', 'exit_high']
        else:
            df_ind = calculate_indicators(df, params)
            drop_cols = ['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi']
        df_ind = df_ind.dropna(subset=drop_cols).reset_index(drop=True)
        if len(df_ind) > 30:
            prepared[symbol] = df_ind

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

    return prepared, lookups, prev_lookups


# ============================================================================
# CORE TRI-MODE BACKTEST ENGINE
# ============================================================================

def has_coin_conflict(positions, symbol):
    """Check if any position already holds this coin in any mode."""
    for key in positions:
        if key.endswith(':' + symbol):
            return True
    return False


def compute_total_equity(capital, positions, all_lookups, date):
    """Mark-to-market all positions + cash."""
    equity = capital
    for key, pos in positions.items():
        mode = pos['mode']
        symbol = pos['symbol']
        lookups = all_lookups[mode]
        row = lookups.get(symbol, {}).get(date)
        if row is not None:
            current_price = float(row['close'])
            if mode == 'SHORT':
                price_change = (pos['entry_price'] - current_price) / pos['entry_price']
                current_val = pos['size_usd'] * (1 + price_change)
                equity += max(current_val, 0)  # isolated margin floor
            else:
                current_val = pos['size_usd'] * (current_price / pos['entry_price'])
                equity += current_val
        else:
            equity += pos['size_usd']
    return equity


def backtest_trimode(coin_data, starting_capital=10000.0, enable_taxes=True):
    """Unified tri-mode backtest with shared equity pool and tax deductions.

    Returns dict with:
      trades, equity_curve, final_capital, pyramid_adds,
      yearly_summary, per_mode_trades
    """
    # Prepare data for all 3 modes
    sl_prepared, sl_lookups, sl_prev = prepare_mode_data(
        coin_data, SPOT_LONG_COINS, SPOT_LONG_PARAMS, is_short=False)
    fl_prepared, fl_lookups, fl_prev = prepare_mode_data(
        coin_data, FUTURES_LONG_COINS, FUTURES_LONG_PARAMS, is_short=False)
    sh_prepared, sh_lookups, sh_prev = prepare_mode_data(
        coin_data, SHORT_COINS_LIST, SHORT_PARAMS, is_short=True)

    all_lookups = {'SPOT_LONG': sl_lookups, 'FUTURES_LONG': fl_lookups, 'SHORT': sh_lookups}
    all_prev = {'SPOT_LONG': sl_prev, 'FUTURES_LONG': fl_prev, 'SHORT': sh_prev}

    # Unified date list
    all_dates = set()
    for mode_prep in [sl_prepared, fl_prepared, sh_prepared]:
        for df in mode_prep.values():
            all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Cost per side for each mode
    sl_cost = (SPOT_LONG_PARAMS['fee_pct'] + SPOT_LONG_PARAMS['slippage_pct']) / 100
    fl_cost = (FUTURES_LONG_PARAMS['fee_pct'] + FUTURES_LONG_PARAMS['slippage_pct']) / 100
    sh_cost = (SHORT_PARAMS['fee_pct'] + SHORT_PARAMS['slippage_pct']) / 100
    funding_daily = SHORT_PARAMS.get('funding_rate_daily', 0) / 100

    # State
    capital = starting_capital
    positions = {}  # key: "MODE:SYMBOL"
    trades = []
    equity_curve = []
    pyramid_adds = 0
    yearly_realized = {}  # year -> cumulative realized $ gains
    yearly_taxes = {}     # year -> {federal, state, total}
    current_year = None

    def record_realized(pnl_usd, year):
        yearly_realized[year] = yearly_realized.get(year, 0.0) + pnl_usd

    def get_cost(mode):
        if mode == 'SPOT_LONG':
            return sl_cost
        elif mode == 'FUTURES_LONG':
            return fl_cost
        else:
            return sh_cost

    for date in all_dates:
        year = date.year

        # === YEAR-END TAX DEDUCTION ===
        if current_year is not None and year != current_year and enable_taxes:
            gains = yearly_realized.get(current_year, 0.0)
            fed, state, total = compute_total_tax(gains)
            capital -= total
            yearly_taxes[current_year] = {
                'gross_gains': gains,
                'federal': fed,
                'state': state,
                'total': total,
            }
        current_year = year

        is_bull = bull_filter.get(date, False)
        is_bear = bear_filter.get(date, False)

        # ============================================================
        # 1. EXITS — process all open positions
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
                # --- Short exit logic ---
                pos['low_watermark'] = min(pos['low_watermark'], float(row['low']))
                pos['hold_days'] += 1

                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ratio = float(row['volume']) / vol_sma
                is_bounce = (volume_ratio > SHORT_PARAMS['volume_blowoff']
                             and float(row['rsi']) < SHORT_PARAMS['rsi_blowoff'])
                stop_mult = SHORT_PARAMS['atr_mult_tight'] if is_bounce else SHORT_PARAMS['atr_mult']

                trailing_stop = pos['low_watermark'] + (stop_mult * current_atr)
                if current_close >= trailing_stop:
                    exit_reason = f'Trailing stop ({stop_mult}x ATR)'

                prev_row = prev_lk.get(symbol, {}).get(date)
                if not exit_reason and prev_row is not None and pd.notna(prev_row.get('exit_high')):
                    if current_close > float(prev_row['exit_high']):
                        exit_reason = f'Donchian exit ({SHORT_PARAMS["exit_period"]}-day high)'

                emergency = pos['entry_price'] * (1 + SHORT_PARAMS['emergency_stop_pct'] / 100)
                if not exit_reason and current_close >= emergency:
                    exit_reason = f'Emergency stop (+{SHORT_PARAMS["emergency_stop_pct"]:.0f}%)'

                if not exit_reason and pos['hold_days'] >= SHORT_PARAMS['max_hold_days']:
                    exit_reason = f'Max hold ({SHORT_PARAMS["max_hold_days"]}d)'

                # Partial profit taking (shorts)
                if not exit_reason:
                    gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= SHORT_PARAMS['tp1_pct']:
                        partial_exit = current_close * (1 + cost)
                        partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                        partial_pnl += funding_daily * pos['hold_days'] * 100
                        partial_size = pos['size_usd'] * SHORT_PARAMS['tp1_fraction']
                        partial_gain = partial_size * (partial_pnl / 100)
                        capital += partial_size + partial_gain
                        record_realized(partial_gain, year)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 1
                        trades.append({
                            'symbol': symbol, 'mode': 'SHORT',
                            'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                            'exit_time': row['time'], 'exit_price': partial_exit,
                            'pnl_pct': partial_pnl, 'size_usd': partial_size,
                            'exit_reason': f'Partial TP1 (-{SHORT_PARAMS["tp1_pct"]:.0f}%)',
                            'win': True, 'hold_days': pos['hold_days'],
                        })
                    elif pos['partials_taken'] == 1 and gain_pct >= SHORT_PARAMS['tp2_pct']:
                        partial_exit = current_close * (1 + cost)
                        partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                        partial_pnl += funding_daily * pos['hold_days'] * 100
                        partial_size = pos['size_usd'] * SHORT_PARAMS['tp2_fraction']
                        partial_gain = partial_size * (partial_pnl / 100)
                        capital += partial_size + partial_gain
                        record_realized(partial_gain, year)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 2
                        trades.append({
                            'symbol': symbol, 'mode': 'SHORT',
                            'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                            'exit_time': row['time'], 'exit_price': partial_exit,
                            'pnl_pct': partial_pnl, 'size_usd': partial_size,
                            'exit_reason': f'Partial TP2 (-{SHORT_PARAMS["tp2_pct"]:.0f}%)',
                            'win': True, 'hold_days': pos['hold_days'],
                        })

                # Full short exit
                if exit_reason:
                    exit_price = current_close * (1 + cost)
                    price_pnl = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
                    fund_pnl = funding_daily * pos['hold_days'] * 100
                    total_pnl = price_pnl + fund_pnl
                    pnl_usd = pos['size_usd'] * (total_pnl / 100)
                    capital += pos['size_usd'] + pnl_usd
                    record_realized(pnl_usd, year)
                    trades.append({
                        'symbol': symbol, 'mode': 'SHORT',
                        'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                        'exit_time': row['time'], 'exit_price': exit_price,
                        'pnl_pct': total_pnl, 'size_usd': pos['size_usd'],
                        'exit_reason': exit_reason, 'win': total_pnl > 0,
                        'hold_days': pos['hold_days'],
                    })
                    keys_to_close.append(key)

            else:
                # --- Long exit logic (spot + futures) ---
                params = SPOT_LONG_PARAMS if mode == 'SPOT_LONG' else FUTURES_LONG_PARAMS
                pos['high_watermark'] = max(pos['high_watermark'], float(row['high']))

                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ratio = float(row['volume']) / vol_sma
                is_blowoff = (volume_ratio > params['volume_blowoff']
                              and float(row['rsi']) > params['rsi_blowoff'])
                stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

                trailing_stop = pos['high_watermark'] - (stop_mult * current_atr)
                if current_close <= trailing_stop:
                    exit_reason = f'Trailing stop ({stop_mult}x ATR)'

                prev_row = prev_lk.get(symbol, {}).get(date)
                if not exit_reason and prev_row is not None and pd.notna(prev_row['exit_low']):
                    if current_close < float(prev_row['exit_low']):
                        exit_reason = 'Donchian exit (10-day low)'

                if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                    exit_reason = 'Emergency stop (15%)'

                # Partial profit taking (longs)
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                        partial_price = current_close * (1 - cost)
                        partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params['tp1_fraction']
                        partial_gain = partial_size * (partial_pnl / 100)
                        capital += partial_size + partial_gain
                        record_realized(partial_gain, year)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 1
                        trades.append({
                            'symbol': symbol, 'mode': mode,
                            'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                            'exit_time': row['time'], 'exit_price': partial_price,
                            'pnl_pct': partial_pnl, 'size_usd': partial_size,
                            'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                            'win': True,
                        })
                    elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                        partial_price = current_close * (1 - cost)
                        partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params['tp2_fraction']
                        partial_gain = partial_size * (partial_pnl / 100)
                        capital += partial_size + partial_gain
                        record_realized(partial_gain, year)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 2
                        trades.append({
                            'symbol': symbol, 'mode': mode,
                            'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                            'exit_time': row['time'], 'exit_price': partial_price,
                            'pnl_pct': partial_pnl, 'size_usd': partial_size,
                            'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                            'win': True,
                        })

                # Full long exit
                if exit_reason:
                    exit_price = current_close * (1 - cost)
                    pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                    pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                    capital += pos['size_usd'] + pnl_usd
                    record_realized(pnl_usd, year)
                    trades.append({
                        'symbol': symbol, 'mode': mode,
                        'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
                        'exit_time': row['time'], 'exit_price': exit_price,
                        'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
                        'exit_reason': exit_reason, 'win': pnl_pct > 0,
                    })
                    keys_to_close.append(key)

        for k in keys_to_close:
            del positions[k]

        # ============================================================
        # 2. PYRAMIDING — longs only, gated by bull filter
        # ============================================================
        if is_bull:
            for key, pos in list(positions.items()):
                mode = pos['mode']
                if mode == 'SHORT' or pos.get('pyramided'):
                    continue
                symbol = pos['symbol']
                params = SPOT_LONG_PARAMS if mode == 'SPOT_LONG' else FUTURES_LONG_PARAMS
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
                    stop_distance = params['atr_mult'] * atr_val
                    stop_pct = stop_distance / current_close
                    if stop_pct > 0:
                        add_size = add_risk / stop_pct
                    else:
                        add_size = total_equity * 0.05
                    add_size = min(add_size, capital * 0.50)
                    if add_size >= 100:
                        capital -= add_size
                        pos['size_usd'] += add_size
                        pos['pyramided'] = True
                        pyramid_adds += 1
                        trades.append({
                            'symbol': symbol, 'mode': mode,
                            'entry_time': row['time'],
                            'entry_price': current_close * (1 + cost),
                            'exit_time': row['time'], 'exit_price': current_close,
                            'pnl_pct': 0, 'size_usd': add_size,
                            'exit_reason': f'Pyramid add (+{gain_pct:.0f}%)',
                            'win': True,
                        })

        # ============================================================
        # 3. NEW ENTRIES — priority: spot long > futures long > short
        # ============================================================

        # 3a. Spot long entries (bull filter gate)
        if len(positions) < MAX_POSITIONS and is_bull:
            for symbol in SPOT_LONG_COINS:
                if symbol not in sl_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                row = sl_lookups.get(symbol, {}).get(date)
                prev_row = sl_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row['donchian_high']):
                    continue

                current_close = float(row['close'])
                breakout = current_close > float(prev_row['donchian_high'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ok = float(row['volume']) > SPOT_LONG_PARAMS['volume_mult'] * vol_sma
                trend_ok = current_close > float(row['ema_21'])

                if breakout and volume_ok and trend_ok:
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    risk_amount = total_equity * RISK_PCT
                    entry_price = current_close * (1 + sl_cost)
                    atr_val = float(row['atr'])
                    stop_pct = (SPOT_LONG_PARAMS['atr_mult'] * atr_val) / entry_price
                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / MAX_POSITIONS
                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue
                    capital -= position_size
                    positions[f'SPOT_LONG:{symbol}'] = {
                        'mode': 'SPOT_LONG', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0, 'size_usd': position_size, 'pyramided': False,
                    }

        # 3b. Futures long entries (bull filter gate, no same-coin conflict)
        if len(positions) < MAX_POSITIONS and is_bull:
            for symbol in FUTURES_LONG_COINS:
                if symbol not in fl_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                row = fl_lookups.get(symbol, {}).get(date)
                prev_row = fl_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row['donchian_high']):
                    continue

                current_close = float(row['close'])
                breakout = current_close > float(prev_row['donchian_high'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ok = float(row['volume']) > FUTURES_LONG_PARAMS['volume_mult'] * vol_sma
                trend_ok = current_close > float(row['ema_21'])

                if breakout and volume_ok and trend_ok:
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    risk_amount = total_equity * RISK_PCT
                    entry_price = current_close * (1 + fl_cost)
                    atr_val = float(row['atr'])
                    stop_pct = (FUTURES_LONG_PARAMS['atr_mult'] * atr_val) / entry_price
                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / MAX_POSITIONS
                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue
                    capital -= position_size
                    positions[f'FUTURES_LONG:{symbol}'] = {
                        'mode': 'FUTURES_LONG', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'high_watermark': float(row['high']),
                        'partials_taken': 0, 'size_usd': position_size, 'pyramided': False,
                    }

        # 3c. Short entries (bear filter gate, no same-coin conflict)
        if len(positions) < MAX_POSITIONS and is_bear:
            for symbol in SHORT_COINS_LIST:
                if symbol not in sh_prepared or has_coin_conflict(positions, symbol):
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                row = sh_lookups.get(symbol, {}).get(date)
                prev_row = sh_prev.get(symbol, {}).get(date)
                if row is None or prev_row is None or pd.isna(prev_row.get('donchian_low')):
                    continue

                current_close = float(row['close'])
                breakdown = current_close < float(prev_row['donchian_low'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ok = float(row['volume']) > SHORT_PARAMS['volume_mult'] * vol_sma
                trend_ok = current_close < float(row['ema_21'])

                if breakdown and volume_ok and trend_ok:
                    total_equity = compute_total_equity(capital, positions, all_lookups, date)
                    risk_amount = total_equity * RISK_PCT
                    entry_price = current_close * (1 - sh_cost)
                    atr_val = float(row['atr'])
                    stop_pct = (SHORT_PARAMS['atr_mult'] * atr_val) / entry_price
                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / MAX_POSITIONS
                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue
                    capital -= position_size
                    positions[f'SHORT:{symbol}'] = {
                        'mode': 'SHORT', 'symbol': symbol,
                        'entry_price': entry_price, 'entry_time': row['time'],
                        'low_watermark': float(row['low']),
                        'partials_taken': 0, 'size_usd': position_size,
                        'pyramided': False, 'hold_days': 0,
                    }

        # ============================================================
        # 4. EQUITY TRACKING
        # ============================================================
        total_equity = compute_total_equity(capital, positions, all_lookups, date)
        equity_curve.append({'date': date, 'equity': total_equity})

    # === Close remaining positions ===
    for key, pos in list(positions.items()):
        mode = pos['mode']
        symbol = pos['symbol']
        cost = get_cost(mode)
        lookups = all_lookups[mode]
        if mode == 'SHORT':
            prep = sh_prepared
        elif mode == 'FUTURES_LONG':
            prep = fl_prepared
        else:
            prep = sl_prepared

        last = prep[symbol].iloc[-1]
        if mode == 'SHORT':
            exit_price = float(last['close']) * (1 + cost)
            price_pnl = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
            fund_pnl = funding_daily * pos['hold_days'] * 100
            total_pnl = price_pnl + fund_pnl
        else:
            exit_price = float(last['close']) * (1 - cost)
            total_pnl = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (total_pnl / 100)
        capital += pos['size_usd'] + pnl_usd
        record_realized(pnl_usd, current_year)
        trades.append({
            'symbol': symbol, 'mode': mode,
            'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'],
            'exit_time': last['time'], 'exit_price': exit_price,
            'pnl_pct': total_pnl, 'size_usd': pos['size_usd'],
            'exit_reason': 'End of backtest', 'win': total_pnl > 0,
        })

    # Process final year's taxes
    if enable_taxes and current_year is not None:
        gains = yearly_realized.get(current_year, 0.0)
        fed, state, total = compute_total_tax(gains)
        capital -= total
        yearly_taxes[current_year] = {
            'gross_gains': gains, 'federal': fed, 'state': state, 'total': total,
        }

    # Build per-mode breakdown
    per_mode = {'SPOT_LONG': [], 'FUTURES_LONG': [], 'SHORT': []}
    for t in trades:
        per_mode[t['mode']].append(t)

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': capital,
        'pyramid_adds': pyramid_adds,
        'yearly_taxes': yearly_taxes,
        'yearly_realized': yearly_realized,
        'per_mode_trades': per_mode,
        'starting_capital': starting_capital,
    }


# ============================================================================
# REPORTING
# ============================================================================

def compute_stats_from_trades(trades):
    """Compute WR, PF, total return from a trades list."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'gross_pnl': 0}
    wins = [t for t in trades if t['win'] and t['pnl_pct'] != 0]
    losses = [t for t in trades if not t['win'] and t['pnl_pct'] != 0]
    total_win = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in wins)
    total_loss = sum(abs(t['size_usd'] * t['pnl_pct'] / 100) for t in losses)
    pf = total_win / total_loss if total_loss > 0 else float('inf')
    wr = len(wins) / len([t for t in trades if t['pnl_pct'] != 0]) * 100 if trades else 0
    gross_pnl = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in trades)
    return {'trades': len(trades), 'wr': wr, 'pf': pf, 'gross_pnl': gross_pnl}


def print_yearly_breakdown(results):
    """Print year-by-year breakdown with taxes and ending equity."""
    yearly_taxes = results['yearly_taxes']
    trades = results['trades']
    starting = results['starting_capital']

    # Build year-by-year equity from equity curve
    eq_by_year = {}
    for pt in results['equity_curve']:
        eq_by_year[pt['date'].year] = pt['equity']

    years = sorted(yearly_taxes.keys())
    print(f"\n  {'Year':<6} {'Trades':>7} {'Gross P&L':>12} {'Fed Tax':>10} {'CA Tax':>10} {'Total Tax':>11} {'Net P&L':>12} {'EOY Equity':>13}")
    print(f"  {'----':<6} {'------':>7} {'---------':>12} {'-------':>10} {'------':>10} {'---------':>11} {'-------':>12} {'----------':>13}")

    running_equity = starting
    for yr in years:
        tx = yearly_taxes[yr]
        yr_trades = [t for t in trades if hasattr(t['exit_time'], 'year') and t['exit_time'].year == yr]
        if not yr_trades:
            yr_trades = [t for t in trades
                         if hasattr(t.get('exit_time'), 'date') and t['exit_time'].date().year == yr]
        n_trades = len(yr_trades)
        gross = tx['gross_gains']
        fed = tx['federal']
        state = tx['state']
        total = tx['total']
        net = gross - total
        running_equity += net

        eff_rate = (total / gross * 100) if gross > 0 else 0

        print(f"  {yr:<6} {n_trades:>7} {'+' if gross >= 0 else ''}{gross:>11,.0f} {fed:>10,.0f} {state:>10,.0f} {total:>11,.0f} {'+' if net >= 0 else ''}{net:>11,.0f} ${running_equity:>12,.0f}")

    # Totals
    total_gross = sum(tx['gross_gains'] for tx in yearly_taxes.values())
    total_fed = sum(tx['federal'] for tx in yearly_taxes.values())
    total_state = sum(tx['state'] for tx in yearly_taxes.values())
    total_tax = sum(tx['total'] for tx in yearly_taxes.values())
    total_net = total_gross - total_tax
    eff = (total_tax / total_gross * 100) if total_gross > 0 else 0

    print(f"  {'----':<6} {'------':>7} {'---------':>12} {'-------':>10} {'------':>10} {'---------':>11} {'-------':>12} {'----------':>13}")
    print(f"  {'TOTAL':<6} {len([t for t in trades if t['pnl_pct'] != 0]):>7} {'+' if total_gross >= 0 else ''}{total_gross:>11,.0f} {total_fed:>10,.0f} {total_state:>10,.0f} {total_tax:>11,.0f} {'+' if total_net >= 0 else ''}{total_net:>11,.0f} ${results['final_capital']:>12,.0f}")
    print(f"\n  Effective tax rate: {eff:.1f}%")


def print_mode_breakdown(results):
    """Print per-mode breakdown."""
    per_mode = results['per_mode_trades']
    starting = results['starting_capital']

    print(f"\n  {'Mode':<16} {'Trades':>7} {'WR':>7} {'PF':>7} {'Gross P&L':>12} {'Avg P&L%':>9}")
    print(f"  {'----':<16} {'------':>7} {'--':>7} {'--':>7} {'---------':>12} {'--------':>9}")

    for mode in ['SPOT_LONG', 'FUTURES_LONG', 'SHORT']:
        mode_trades = per_mode[mode]
        real_trades = [t for t in mode_trades if t['pnl_pct'] != 0]
        if not real_trades:
            print(f"  {mode:<16} {'0':>7} {'—':>7} {'—':>7} {'$0':>12} {'—':>9}")
            continue
        stats = compute_stats_from_trades(real_trades)
        avg_pnl = sum(t['pnl_pct'] for t in real_trades) / len(real_trades)
        print(f"  {mode:<16} {stats['trades']:>7} {stats['wr']:>6.1f}% {stats['pf']:>6.2f} {'+' if stats['gross_pnl'] >= 0 else ''}{stats['gross_pnl']:>11,.0f} {avg_pnl:>+8.1f}%")


def print_comparison_table(all_results):
    """Print final comparison across all capital levels."""
    print(f"\n  {'Starting':>10} {'Trades':>7} {'WR':>7} {'PF':>7} {'Gross Return':>14} {'Total Tax':>11} {'Net Return':>12} {'Net %':>7} {'Max DD':>8} {'Final Equity':>14}")
    print(f"  {'--------':>10} {'------':>7} {'--':>7} {'--':>7} {'------------':>14} {'---------':>11} {'----------':>12} {'-----':>7} {'------':>8} {'------------':>14}")

    for cap, results in all_results:
        trades = [t for t in results['trades'] if t['pnl_pct'] != 0]
        stats = compute_stats_from_trades(trades)
        total_tax = sum(tx['total'] for tx in results['yearly_taxes'].values())
        gross_ret = stats['gross_pnl']
        net_ret = gross_ret - total_tax
        net_pct = (net_ret / cap) * 100

        # Max drawdown from equity curve
        equities = [pt['equity'] for pt in results['equity_curve']]
        if equities:
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        print(f"  ${cap:>9,} {stats['trades']:>7} {stats['wr']:>6.1f}% {stats['pf']:>6.2f} {'+' if gross_ret >= 0 else ''}{gross_ret:>13,.0f} {total_tax:>11,.0f} {'+' if net_ret >= 0 else ''}{net_ret:>11,.0f} {net_pct:>+6.1f}% {max_dd:>7.1f}% ${results['final_capital']:>13,.0f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 110)
    print("TRI-MODE COMBINED BACKTEST -- SPOT LONG + FUTURES LONG + SHORT")
    print("  Shared equity pool | Max 4 positions | Compounding | Tax-aware")
    print(f"  Bull filter: BTC > SMA(200) | Bear filter: Death cross")
    print(f"  Starting capitals: ${STARTING_CAPITALS[0]:,}, ${STARTING_CAPITALS[1]:,}, ${STARTING_CAPITALS[2]:,}")
    print("=" * 110)

    # 1. Fetch data
    print(f"\nFetching daily data for {len(ALL_COINS)} coins (4 years)...")
    coin_data = fetch_all_coins(coins=ALL_COINS, years=4)
    print(f"Loaded {len(coin_data)} coins successfully\n")

    # 2. Compute filters
    btc_df = coin_data.get('BTC-USD')
    global bull_filter, bear_filter
    bull_filter = compute_btc_bull_filter(btc_df)
    bear_filter = compute_btc_bear_filter(btc_df, require_death_cross=True)

    # 3. Run at all capital levels
    all_results = []
    for cap in STARTING_CAPITALS:
        print(f"\n  Running ${cap:,} backtest...", end=' ', flush=True)
        results = backtest_trimode(coin_data, starting_capital=float(cap))
        real_trades = [t for t in results['trades'] if t['pnl_pct'] != 0]
        print(f"done — {len(real_trades)} trades, final equity ${results['final_capital']:,.0f}")
        all_results.append((cap, results))

    # 4. Detailed results per capital level
    for cap, results in all_results:
        print_section(f"${cap:,} STARTING CAPITAL")

        print("  YEAR-BY-YEAR BREAKDOWN")
        print_yearly_breakdown(results)

        print(f"\n  MODE BREAKDOWN")
        print_mode_breakdown(results)

        print(f"\n  Pyramid adds: {results['pyramid_adds']}")

    # 5. Comparison table
    print_section("COMPARISON ACROSS CAPITAL LEVELS")
    print_comparison_table(all_results)

    # 6. Verdict
    print_section("SUMMARY")
    # Pick the $10K result for commentary
    r10 = all_results[0][1]
    total_gross = sum(tx['gross_gains'] for tx in r10['yearly_taxes'].values())
    total_tax = sum(tx['total'] for tx in r10['yearly_taxes'].values())
    total_net = total_gross - total_tax
    eff_rate = (total_tax / total_gross * 100) if total_gross > 0 else 0

    sl_stats = compute_stats_from_trades([t for t in r10['per_mode_trades']['SPOT_LONG'] if t['pnl_pct'] != 0])
    fl_stats = compute_stats_from_trades([t for t in r10['per_mode_trades']['FUTURES_LONG'] if t['pnl_pct'] != 0])
    sh_stats = compute_stats_from_trades([t for t in r10['per_mode_trades']['SHORT'] if t['pnl_pct'] != 0])

    print(f"  On $10K starting capital (4 years, compounding, after taxes):")
    print(f"    Gross return:     +${total_gross:,.0f} ({total_gross / 10000 * 100:+.1f}%)")
    print(f"    Total taxes paid: ${total_tax:,.0f} ({eff_rate:.1f}% effective rate)")
    print(f"    Net return:       +${total_net:,.0f} ({total_net / 10000 * 100:+.1f}%)")
    print(f"    Final equity:     ${r10['final_capital']:,.0f}")
    print(f"\n  Mode contributions (gross P&L):")
    print(f"    Spot Long:     +${sl_stats['gross_pnl']:,.0f} ({sl_stats['trades']} trades)")
    print(f"    Futures Long:  +${fl_stats['gross_pnl']:,.0f} ({fl_stats['trades']} trades)")
    print(f"    Short:         {'+' if sh_stats['gross_pnl'] >= 0 else ''}${sh_stats['gross_pnl']:,.0f} ({sh_stats['trades']} trades)")
    print()


if __name__ == '__main__':
    main()
