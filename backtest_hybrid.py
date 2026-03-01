"""Hybrid Core/Satellite Backtest — Smart Accumulation + Cycle Riding

Compares 3 approaches on 2022-2026 data at $30K starting capital:
  1. CURRENT BOT: Donchian + bull filter + 4x ATR trailing + pyramiding
  2. BUY & HOLD: Equal-weight top coins, buy at start, hold to end
  3. HYBRID: 65% core (cycle-ride, no bull filter, weekly stops, composite exit)
           + 35% satellite (existing Donchian bot)

Core portfolio:
  - Coins: BTC (20%), ETH (15%), SOL (15%), XRP (15%) — deployed via signal-enhanced DCA
  - Entry: Donchian breakouts allowed in ALL markets (no bull filter gate)
  - Stops: Weekly close below 20-week SMA only (survives 30-40% corrections)
  - Exit: Composite top indicator (Pi Cycle + monthly RSI + weekly EMA + halving timing)
  - No partial profit-taking, no daily trailing stops

Satellite portfolio:
  - Existing Donchian bot with bull filter, 4x ATR trailing, pyramiding
  - Runs on 35% of capital

Usage:
  cd C:\\ResearchAgent && venv/Scripts/python backtest_hybrid.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as date_type
from collections import defaultdict

from backtest_donchian_daily import (
    fetch_all_coins, calculate_indicators, compute_stats,
    DEFAULT_PARAMS, COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_phase3 import backtest_portfolio_phase3
from backtest_walkforward import print_section
from backtest_trimode import compute_total_tax


# ============================================================================
# CONFIGURATION
# ============================================================================

STARTING_CAPITAL = 30_000.0
CORE_PCT = 0.65      # 65% core
SATELLITE_PCT = 0.35  # 35% satellite

# Core coin allocations (% of core capital)
CORE_ALLOCATIONS = {
    'BTC-USD': 0.20 / CORE_PCT,   # 20% of total = 30.8% of core
    'ETH-USD': 0.15 / CORE_PCT,   # 15% of total = 23.1% of core
    'SOL-USD': 0.15 / CORE_PCT,   # 15% of total = 23.1% of core
    'XRP-USD': 0.15 / CORE_PCT,   # 15% of total = 23.1% of core
}

ALL_COINS = sorted(set(list(CORE_ALLOCATIONS.keys()) + list(COIN_UNIVERSE)))

# Satellite (existing Donchian bot)
SATELLITE_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Satellite (Donchian)',
    'atr_mult': 4.0,
    'max_positions': 4,
}

# BTC halving dates
HALVING_DATES = {
    'halving_2020': date_type(2020, 5, 11),
    'halving_2024': date_type(2024, 4, 19),
}

# Composite exit thresholds
COMPOSITE_EXIT = {
    'monthly_rsi_threshold': 90,
    'halving_months_threshold': 18,  # months after halving
    'sell_at_score': 2,              # start selling at 2/4
    'full_exit_score': 3,            # fully exit at 3/4
}

# Core entry signal params
CORE_ENTRY = {
    'donchian_period': 20,
    'volume_mult': 1.5,
    'ema_period': 21,
    'rsi_oversold': 30,      # extra buy signal
    'dca_interval_days': 14,  # DCA every 2 weeks during accumulation
}

# Weekly stop
WEEKLY_STOP_PERIOD = 20  # 20-week SMA


# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================

def compute_weekly_sma(daily_df, period=20):
    """Compute weekly closing prices and N-week SMA.

    Returns dict: date -> {'weekly_close': float, 'weekly_sma': float}
    Weekly close = Friday close (or last trading day of the week).
    """
    df = daily_df.copy()
    df['date'] = df['time'].dt.date
    df['week'] = df['time'].dt.isocalendar().week.astype(int)
    df['year'] = df['time'].dt.year

    # Group by year+week, take last close as weekly close
    weekly = df.groupby(['year', 'week']).agg(
        weekly_close=('close', 'last'),
        last_date=('date', 'max'),
    ).reset_index()
    weekly = weekly.sort_values('last_date').reset_index(drop=True)
    weekly['weekly_sma'] = weekly['weekly_close'].rolling(window=period).mean()

    # Build lookup: every daily date maps to its most recent weekly SMA
    result = {}
    weekly_list = weekly.dropna(subset=['weekly_sma']).to_dict('records')
    wi = 0
    for _, row in df.iterrows():
        d = row['date']
        # Advance weekly pointer to most recent completed week
        while wi < len(weekly_list) - 1 and weekly_list[wi + 1]['last_date'] <= d:
            wi += 1
        if wi < len(weekly_list) and weekly_list[wi]['last_date'] <= d:
            result[d] = {
                'weekly_close': float(weekly_list[wi]['weekly_close']),
                'weekly_sma': float(weekly_list[wi]['weekly_sma']),
            }
    return result


def compute_pi_cycle_top(btc_df):
    """Compute Pi Cycle Top indicator: 111-day MA crosses above 2x 350-day MA.

    Returns dict: date -> bool (True = top signal firing)
    """
    df = btc_df.copy()
    df['ma_111'] = df['close'].rolling(window=111).mean()
    df['ma_350x2'] = df['close'].rolling(window=350).mean() * 2

    result = {}
    prev_below = True
    for _, row in df.iterrows():
        d = row['time'].date()
        if pd.isna(row['ma_111']) or pd.isna(row['ma_350x2']):
            result[d] = False
            continue
        currently_above = float(row['ma_111']) > float(row['ma_350x2'])
        # Signal fires on the crossover (transition from below to above)
        result[d] = currently_above and prev_below
        prev_below = not currently_above
    return result


def compute_monthly_rsi(btc_df, period=14):
    """Compute monthly RSI from daily data.

    Returns dict: date -> float (RSI value, updated monthly)
    """
    df = btc_df.copy()
    df['date'] = df['time'].dt.date
    df['month'] = df['time'].dt.to_period('M')

    # Get monthly closes
    monthly = df.groupby('month').agg(
        close=('close', 'last'),
        last_date=('date', 'max'),
    ).reset_index()
    monthly = monthly.sort_values('last_date').reset_index(drop=True)

    # Compute RSI on monthly closes
    delta = monthly['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    monthly['rsi'] = 100 - (100 / (1 + rs))

    # Map each daily date to its most recent monthly RSI
    result = {}
    monthly_list = monthly.dropna(subset=['rsi']).to_dict('records')
    mi = 0
    for _, row in df.iterrows():
        d = row['date']
        while mi < len(monthly_list) - 1 and monthly_list[mi + 1]['last_date'] <= d:
            mi += 1
        if mi < len(monthly_list) and monthly_list[mi]['last_date'] <= d:
            result[d] = float(monthly_list[mi]['rsi'])
    return result


def compute_weekly_ema_break(btc_df, period=21):
    """Check if weekly close is below 21-week EMA after being above for 6+ months.

    Returns dict: date -> bool (True = bearish break)
    """
    df = btc_df.copy()
    df['date'] = df['time'].dt.date
    df['week'] = df['time'].dt.isocalendar().week.astype(int)
    df['year'] = df['time'].dt.year

    weekly = df.groupby(['year', 'week']).agg(
        close=('close', 'last'),
        last_date=('date', 'max'),
    ).reset_index()
    weekly = weekly.sort_values('last_date').reset_index(drop=True)
    weekly['ema'] = weekly['close'].ewm(span=period, adjust=False).mean()

    # Track weeks above EMA
    weeks_above = 0
    weekly_signals = {}
    for _, row in weekly.iterrows():
        if pd.isna(row['ema']):
            weekly_signals[row['last_date']] = False
            continue
        if float(row['close']) > float(row['ema']):
            weeks_above += 1
            weekly_signals[row['last_date']] = False
        else:
            # Break below after being above for 26+ weeks (~6 months)
            weekly_signals[row['last_date']] = weeks_above >= 26
            weeks_above = 0

    # Map daily dates to weekly signals
    result = {}
    sorted_weeks = sorted(weekly_signals.keys())
    wi = 0
    for _, row in df.iterrows():
        d = row['date']
        while wi < len(sorted_weeks) - 1 and sorted_weeks[wi + 1] <= d:
            wi += 1
        if wi < len(sorted_weeks) and sorted_weeks[wi] <= d:
            result[d] = weekly_signals[sorted_weeks[wi]]
        else:
            result[d] = False
    return result


def get_halving_months(d):
    """Return months since most recent halving."""
    # Find most recent halving
    recent = None
    for name, hd in HALVING_DATES.items():
        if hd <= d:
            if recent is None or hd > recent:
                recent = hd
    if recent is None:
        return 0
    return (d.year - recent.year) * 12 + (d.month - recent.month)


def compute_composite_exit_score(d, pi_cycle, monthly_rsi, weekly_ema_break):
    """Compute composite top indicator score (0-4).

    Signals:
    1. Pi Cycle Top crossover (or recently fired within 30 days)
    2. Monthly RSI > 90
    3. Weekly close < 21-week EMA (after 6+ months above)
    4. > 18 months post-halving
    """
    score = 0

    # 1. Pi Cycle (check if fired recently — within 30 days)
    for offset in range(31):
        check = d - timedelta(days=offset)
        if pi_cycle.get(check, False):
            score += 1
            break

    # 2. Monthly RSI > 90
    if monthly_rsi.get(d, 0) > COMPOSITE_EXIT['monthly_rsi_threshold']:
        score += 1

    # 3. Weekly EMA break
    if weekly_ema_break.get(d, False):
        score += 1

    # 4. Halving timing
    if get_halving_months(d) > COMPOSITE_EXIT['halving_months_threshold']:
        score += 1

    return score


# ============================================================================
# CORE PORTFOLIO BACKTEST
# ============================================================================

def backtest_core(coin_data, core_capital, all_dates, indicators_cache):
    """Backtest the core cycle-riding portfolio.

    Entry logic:
    - Donchian breakout (20-day high) on any coin — NO bull filter
    - OR DCA buy every 2 weeks during accumulation (RSI < 30 bonus)
    - Each coin has a target allocation; entries build toward that target

    Exit logic:
    - Weekly close below 20-week SMA for that specific coin
    - OR composite top indicator score >= threshold
    - No partial profit-taking, no daily trailing stops

    Position sizing:
    - Each coin targets its allocation % of core capital
    - Initial entry: full allocation. Re-entry after stop: full allocation.
    """
    cost_spot = (0.45 + 0.05) / 100   # spot fees
    cost_futures = (0.06 + 0.05) / 100  # futures fees (for all-futures mode)
    cost = cost_futures  # use futures fees for core

    capital = core_capital
    positions = {}  # symbol -> {entry_price, size_usd, entry_date, high_watermark}
    trades = []
    equity_curve = []
    composite_scores = []

    # Prepare indicators for core coins
    prepared = {}
    lookups = {}
    for symbol in CORE_ALLOCATIONS:
        if symbol not in coin_data:
            continue
        df = calculate_indicators(coin_data[symbol], {
            **DEFAULT_PARAMS,
            'donchian_period': CORE_ENTRY['donchian_period'],
            'volume_mult': CORE_ENTRY['volume_mult'],
            'ema_period': CORE_ENTRY['ema_period'],
        })
        df = df.dropna(subset=['donchian_high', 'atr', 'ema_21', 'rsi'])
        df = df.reset_index(drop=True)
        if len(df) > 30:
            prepared[symbol] = df
            lookup = {}
            for _, row in df.iterrows():
                lookup[row['time'].date()] = row
            lookups[symbol] = lookup

    # Previous day lookups
    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # Weekly SMA for each core coin (for stop logic)
    weekly_sma_data = {}
    for symbol in CORE_ALLOCATIONS:
        if symbol in coin_data:
            weekly_sma_data[symbol] = compute_weekly_sma(
                coin_data[symbol], period=WEEKLY_STOP_PERIOD)

    # Composite exit indicators (BTC-based)
    pi_cycle = indicators_cache['pi_cycle']
    monthly_rsi = indicators_cache['monthly_rsi']
    weekly_ema_break = indicators_cache['weekly_ema_break']

    last_dca = {}  # symbol -> last DCA date
    exited_via_composite = False  # once composite fires, stay out

    for date in all_dates:
        score = compute_composite_exit_score(date, pi_cycle, monthly_rsi, weekly_ema_break)
        composite_scores.append({'date': date, 'score': score})

        # === EXITS ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            row = lookups.get(symbol, {}).get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            exit_reason = None

            # 1. Composite top indicator
            if score >= COMPOSITE_EXIT['full_exit_score']:
                exit_reason = f'Composite top (score {score}/4)'
                exited_via_composite = True

            # 2. Weekly 20-week SMA stop
            if not exit_reason:
                ws = weekly_sma_data.get(symbol, {}).get(date)
                if ws and ws['weekly_close'] < ws['weekly_sma']:
                    # Only trigger if we've been in the trade > 30 days
                    hold_days = (date - pos['entry_date']).days
                    if hold_days > 30:
                        exit_reason = f'Weekly < 20w SMA (${ws["weekly_sma"]:,.0f})'

            # 3. Scale out at lower composite score
            if not exit_reason and score >= COMPOSITE_EXIT['sell_at_score']:
                # Sell 50% of position
                half_size = pos['size_usd'] * 0.5
                exit_price = current_close * (1 - cost)
                pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = half_size * (pnl_pct / 100)
                capital += half_size + pnl_usd
                pos['size_usd'] -= half_size
                trades.append({
                    'symbol': symbol, 'mode': 'CORE',
                    'entry_time': pos['entry_date'], 'exit_time': date,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct, 'size_usd': half_size,
                    'exit_reason': f'Composite scale-out (score {score}/4)',
                    'win': pnl_pct > 0,
                })
                if pos['size_usd'] < 50:
                    del positions[symbol]
                continue

            if exit_reason:
                exit_price = current_close * (1 - cost)
                pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol, 'mode': 'CORE',
                    'entry_time': pos['entry_date'], 'exit_time': date,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
                    'exit_reason': exit_reason, 'win': pnl_pct > 0,
                })
                del positions[symbol]

        # === ENTRIES (no bull filter!) ===
        # Don't re-enter if composite exit fired recently
        if exited_via_composite:
            # Allow re-entry only if score drops back to 0
            if score == 0:
                exited_via_composite = False
            else:
                # Track equity and continue
                total_eq = capital
                for sym, pos in positions.items():
                    r = lookups.get(sym, {}).get(date)
                    if r is not None:
                        unrealized = pos['size_usd'] * ((float(r['close']) - pos['entry_price']) / pos['entry_price'])
                        total_eq += pos['size_usd'] + unrealized
                    else:
                        total_eq += pos['size_usd']
                equity_curve.append({'date': date, 'equity': total_eq})
                continue

        for symbol, alloc_pct in CORE_ALLOCATIONS.items():
            if symbol in positions:
                continue
            if symbol not in prepared:
                continue

            row = lookups[symbol].get(date)
            prev_row = prev_lookups.get(symbol, {}).get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            should_enter = False
            entry_reason = ''

            # Signal 1: Donchian breakout (works in any market regime)
            if prev_row is not None and not pd.isna(prev_row.get('donchian_high')):
                breakout = current_close > float(prev_row['donchian_high'])
                vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                volume_ok = float(row['volume']) > CORE_ENTRY['volume_mult'] * vol_sma
                trend_ok = current_close > float(row['ema_21'])
                if breakout and volume_ok and trend_ok:
                    should_enter = True
                    entry_reason = 'Donchian breakout'

            # Signal 2: RSI oversold bounce (accumulation signal)
            if not should_enter and not pd.isna(row.get('rsi')):
                if float(row['rsi']) < CORE_ENTRY['rsi_oversold']:
                    # DCA cooldown: only buy every N days per coin
                    last = last_dca.get(symbol)
                    if last is None or (date - last).days >= CORE_ENTRY['dca_interval_days']:
                        should_enter = True
                        entry_reason = f'RSI oversold ({float(row["rsi"]):.0f})'

            if should_enter:
                # Position size: target allocation of total equity
                total_eq = capital + sum(p['size_usd'] for p in positions.values())
                target_size = total_eq * alloc_pct
                target_size = min(target_size, capital * 0.90)
                if target_size < 100:
                    continue

                entry_price = current_close * (1 + cost)
                capital -= target_size
                positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_date': date,
                    'size_usd': target_size,
                    'entry_reason': entry_reason,
                }
                last_dca[symbol] = date

        # Equity tracking
        total_eq = capital
        for sym, pos in positions.items():
            r = lookups.get(sym, {}).get(date)
            if r is not None:
                unrealized = pos['size_usd'] * ((float(r['close']) - pos['entry_price']) / pos['entry_price'])
                total_eq += pos['size_usd'] + unrealized
            else:
                total_eq += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_eq})

    # Close remaining
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        if symbol in prepared:
            last = prepared[symbol].iloc[-1]
            exit_price = float(last['close']) * (1 - cost)
            pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        else:
            exit_price = pos['entry_price']
            pnl_pct = 0
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'mode': 'CORE',
            'entry_time': pos['entry_date'], 'exit_time': all_dates[-1],
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
            'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
            'exit_reason': 'End of backtest', 'win': pnl_pct > 0,
        })

    return trades, equity_curve, capital, composite_scores


# ============================================================================
# BUY & HOLD BENCHMARK
# ============================================================================

def backtest_buy_and_hold(coin_data, starting_capital, all_dates):
    """Simple equal-weight buy-and-hold benchmark.

    Buys all core coins on first available date, holds to end.
    """
    cost = (0.45 + 0.05) / 100
    coins = list(CORE_ALLOCATIONS.keys())
    per_coin = starting_capital / len(coins)

    positions = {}
    equity_curve = []
    trades = []

    # Build price lookups
    price_lookups = {}
    for symbol in coins:
        if symbol not in coin_data:
            continue
        lookup = {}
        for _, row in coin_data[symbol].iterrows():
            lookup[row['time'].date()] = float(row['close'])
        price_lookups[symbol] = lookup

    # Buy on first date each coin has data
    capital = starting_capital
    for symbol in coins:
        if symbol not in price_lookups:
            continue
        for date in all_dates:
            price = price_lookups[symbol].get(date)
            if price:
                entry_price = price * (1 + cost)
                positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_date': date,
                    'size_usd': per_coin,
                }
                capital -= per_coin
                break

    # Track equity daily
    for date in all_dates:
        total_eq = capital
        for symbol, pos in positions.items():
            price = price_lookups.get(symbol, {}).get(date)
            if price:
                unrealized = pos['size_usd'] * ((price - pos['entry_price']) / pos['entry_price'])
                total_eq += pos['size_usd'] + unrealized
            else:
                total_eq += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_eq})

    # Close at end
    final_capital = capital
    for symbol, pos in positions.items():
        last_price = None
        for date in reversed(all_dates):
            last_price = price_lookups.get(symbol, {}).get(date)
            if last_price:
                break
        if last_price:
            exit_price = last_price * (1 - cost)
            pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
            pnl_usd = pos['size_usd'] * (pnl_pct / 100)
            final_capital += pos['size_usd'] + pnl_usd
            trades.append({
                'symbol': symbol, 'mode': 'BUY_HOLD',
                'entry_time': pos['entry_date'], 'exit_time': all_dates[-1],
                'entry_price': pos['entry_price'], 'exit_price': exit_price,
                'pnl_pct': pnl_pct, 'size_usd': pos['size_usd'],
                'exit_reason': 'End of backtest', 'win': pnl_pct > 0,
            })

    return trades, equity_curve, final_capital


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def compute_year_returns(equity_curve):
    """Per-year returns from equity curve."""
    if not equity_curve:
        return {}
    yearly = {}
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
            'start': prev_eq, 'end': end_eq,
            'return_pct': (end_eq / prev_eq - 1) * 100,
            'return_usd': end_eq - prev_eq,
        }
        prev_eq = end_eq
    return yearly


def compute_max_dd(equity_curve):
    """Max drawdown from equity curve."""
    if not equity_curve:
        return 0
    peak = equity_curve[0]['equity']
    max_dd = 0
    for pt in equity_curve:
        peak = max(peak, pt['equity'])
        dd = (peak - pt['equity']) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def print_report(label, trades, equity_curve, starting_capital):
    """Print strategy summary."""
    real = [t for t in trades if abs(t.get('pnl_pct', 0)) > 0.001]
    if not real:
        print(f"  {label}: No trades")
        return {}

    wins = [t for t in real if t['win']]
    losses = [t for t in real if not t['win']]
    wr = len(wins) / len(real) * 100

    gross_w = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in wins)
    gross_l = abs(sum(t['size_usd'] * t['pnl_pct'] / 100 for t in losses))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')
    total_pnl = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in real)
    final = equity_curve[-1]['equity'] if equity_curve else starting_capital
    dd = compute_max_dd(equity_curve)

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    print(f"  {label}")
    print(f"    Trades: {len(real)}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"    Avg Win: {avg_win:+.1f}%  |  Avg Loss: {avg_loss:+.1f}%")
    print(f"    Net P&L: ${total_pnl:+,.0f}  |  Return: {total_pnl/starting_capital*100:+.1f}%")
    print(f"    Max DD: {dd:.1f}%  |  Final: ${final:,.0f}")

    return {
        'trades': len(real), 'wr': wr, 'pf': pf, 'dd': dd,
        'pnl': total_pnl, 'final': final, 'return_pct': total_pnl/starting_capital*100,
    }


def print_year_table(label, eq):
    """Print year-by-year returns."""
    yearly = compute_year_returns(eq)
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
    print("=" * 110)
    print("HYBRID CORE/SATELLITE BACKTEST — SMART ACCUMULATION + CYCLE RIDING")
    print(f"  Capital: ${STARTING_CAPITAL:,.0f} | Core: {CORE_PCT*100:.0f}% (${STARTING_CAPITAL*CORE_PCT:,.0f}) | Satellite: {SATELLITE_PCT*100:.0f}% (${STARTING_CAPITAL*SATELLITE_PCT:,.0f})")
    print(f"  Core coins: {', '.join(CORE_ALLOCATIONS.keys())}")
    print("=" * 110)

    # Fetch data
    print(f"\nFetching daily data for {len(ALL_COINS)} coins...")
    coin_data = fetch_all_coins(coins=ALL_COINS, years=4)
    print(f"Loaded {len(coin_data)} coins")

    btc_df = coin_data.get('BTC-USD')
    bull_filter = compute_btc_bull_filter(btc_df)

    # Build unified timeline
    all_dates = set()
    for df in coin_data.values():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Pre-compute composite exit indicators
    print("\nComputing composite exit indicators...")
    pi_cycle = compute_pi_cycle_top(btc_df)
    monthly_rsi = compute_monthly_rsi(btc_df)
    weekly_ema_break = compute_weekly_ema_break(btc_df)

    pi_fires = sum(1 for v in pi_cycle.values() if v)
    rsi_above_90 = sum(1 for v in monthly_rsi.values() if v > 90)
    ema_breaks = sum(1 for v in weekly_ema_break.values() if v)
    print(f"  Pi Cycle Top fires: {pi_fires}")
    print(f"  Monthly RSI > 90 days: {rsi_above_90}")
    print(f"  Weekly 21-EMA breaks: {ema_breaks}")
    print(f"  Halving dates: {[str(d) for d in HALVING_DATES.values()]}")

    indicators_cache = {
        'pi_cycle': pi_cycle,
        'monthly_rsi': monthly_rsi,
        'weekly_ema_break': weekly_ema_break,
    }

    # ================================================================
    # APPROACH 1: CURRENT BOT (Donchian + bull filter)
    # ================================================================
    print_section("APPROACH 1: CURRENT BOT (Donchian + Bull Filter + 4x ATR)")

    donch_coins = {k: v for k, v in coin_data.items() if k in COIN_UNIVERSE}
    donch_params = {**SATELLITE_PARAMS, 'starting_capital': STARTING_CAPITAL}
    donch_trades, donch_eq, _, _ = backtest_portfolio_phase3(
        donch_coins, donch_params, bull_filter, pyramiding=True)
    donch_stats = print_report('Current Bot (full $30K)', donch_trades, donch_eq, STARTING_CAPITAL)
    print_year_table('Current Bot', donch_eq)

    # ================================================================
    # APPROACH 2: BUY & HOLD BENCHMARK
    # ================================================================
    print_section("APPROACH 2: BUY & HOLD (Equal Weight, 4 Core Coins)")

    bh_trades, bh_eq, bh_cap = backtest_buy_and_hold(coin_data, STARTING_CAPITAL, all_dates)
    bh_stats = print_report('Buy & Hold', bh_trades, bh_eq, STARTING_CAPITAL)
    print_year_table('Buy & Hold', bh_eq)

    # Per-coin breakdown
    print(f"\n    Per-Coin:")
    for t in bh_trades:
        print(f"    {t['symbol']:<10} Entry: ${t['entry_price']:>10,.2f}  Exit: ${t['exit_price']:>10,.2f}  P&L: {t['pnl_pct']:>+7.1f}%  (${t['size_usd']*t['pnl_pct']/100:>+,.0f})")

    # ================================================================
    # APPROACH 3: HYBRID (Core + Satellite)
    # ================================================================
    print_section("APPROACH 3: HYBRID (65% Core Cycle-Ride + 35% Satellite Donchian)")

    core_capital = STARTING_CAPITAL * CORE_PCT
    sat_capital = STARTING_CAPITAL * SATELLITE_PCT

    # Run core
    print(f"\n  --- CORE PORTFOLIO (${core_capital:,.0f}) ---")
    print(f"  No bull filter | Weekly 20w SMA stop | Composite exit | No partials")
    core_trades, core_eq, core_final, composite_scores = backtest_core(
        coin_data, core_capital, all_dates, indicators_cache)
    core_stats = print_report('Core (Cycle-Ride)', core_trades, core_eq, core_capital)
    print_year_table('Core', core_eq)

    # Per-coin trades
    core_by_coin = defaultdict(list)
    for t in core_trades:
        core_by_coin[t['symbol']].append(t)
    print(f"\n    Core Per-Coin:")
    print(f"    {'Coin':<10} {'Trades':>7} {'WR':>6} {'Avg P&L':>8} {'Total P&L':>11}")
    for sym in sorted(core_by_coin.keys()):
        ct = [t for t in core_by_coin[sym] if abs(t['pnl_pct']) > 0.001]
        if not ct:
            continue
        w = sum(1 for t in ct if t['win'])
        wr = w / len(ct) * 100
        avg = np.mean([t['pnl_pct'] for t in ct])
        total = sum(t['size_usd'] * t['pnl_pct'] / 100 for t in ct)
        print(f"    {sym:<10} {len(ct):>7} {wr:>5.1f}% {avg:>+7.1f}% ${total:>+10,.0f}")

    # Show composite exit signals
    print(f"\n    Composite Exit Scores (when > 0):")
    shown = 0
    for cs in composite_scores:
        if cs['score'] > 0 and shown < 20:
            print(f"      {cs['date']}: score {cs['score']}/4")
            shown += 1

    # Run satellite
    print(f"\n  --- SATELLITE PORTFOLIO (${sat_capital:,.0f}) ---")
    print(f"  Standard Donchian + bull filter + 4x ATR + pyramid")
    sat_params = {**SATELLITE_PARAMS, 'starting_capital': sat_capital}
    sat_trades, sat_eq, _, _ = backtest_portfolio_phase3(
        donch_coins, sat_params, bull_filter, pyramiding=True)
    sat_stats = print_report('Satellite (Donchian)', sat_trades, sat_eq, sat_capital)
    print_year_table('Satellite', sat_eq)

    # Combined hybrid equity curve
    print(f"\n  --- COMBINED HYBRID ---")
    hybrid_eq = []
    core_lookup = {pt['date']: pt['equity'] for pt in core_eq}
    sat_lookup = {pt['date']: pt['equity'] for pt in sat_eq}
    for date in all_dates:
        ce = core_lookup.get(date, core_capital)
        se = sat_lookup.get(date, sat_capital)
        hybrid_eq.append({'date': date, 'equity': ce + se})

    hybrid_final = hybrid_eq[-1]['equity'] if hybrid_eq else STARTING_CAPITAL
    hybrid_pnl = hybrid_final - STARTING_CAPITAL
    hybrid_dd = compute_max_dd(hybrid_eq)
    print(f"  Combined Final: ${hybrid_final:,.0f}  |  Net: ${hybrid_pnl:+,.0f}  |  Return: {hybrid_pnl/STARTING_CAPITAL*100:+.1f}%  |  Max DD: {hybrid_dd:.1f}%")
    print_year_table('Hybrid Combined', hybrid_eq)

    # ================================================================
    # HEAD-TO-HEAD COMPARISON
    # ================================================================
    print_section("HEAD-TO-HEAD: ALL 3 APPROACHES")

    donch_yearly = compute_year_returns(donch_eq)
    bh_yearly = compute_year_returns(bh_eq)
    hybrid_yearly = compute_year_returns(hybrid_eq)

    approaches = [
        ('Current Bot', donch_yearly, donch_stats),
        ('Buy & Hold', bh_yearly, bh_stats),
        ('Hybrid', hybrid_yearly, {'dd': hybrid_dd, 'pnl': hybrid_pnl, 'final': hybrid_final, 'return_pct': hybrid_pnl/STARTING_CAPITAL*100}),
    ]

    print(f"\n  SUMMARY:")
    print(f"  {'Approach':<16} {'Net P&L':>12} {'Return%':>9} {'Max DD':>8} {'Final':>14}")
    print(f"  {'--------':<16} {'-------':>12} {'-------':>9} {'------':>8} {'-----':>14}")
    for name, _, stats in approaches:
        if stats:
            print(f"  {name:<16} ${stats.get('pnl',0):>+11,.0f} {stats.get('return_pct',0):>+8.1f}% {stats.get('dd',0):>7.1f}% ${stats.get('final',0):>13,.0f}")

    years = sorted(set(list(donch_yearly.keys()) + list(bh_yearly.keys()) + list(hybrid_yearly.keys())))
    print(f"\n  YEAR-BY-YEAR:")
    print(f"  {'Year':<6} {'Current Bot':>14} {'Buy & Hold':>14} {'Hybrid':>14}")
    for yr in years:
        d_val = donch_yearly.get(yr, {}).get('return_usd', 0)
        b_val = bh_yearly.get(yr, {}).get('return_usd', 0)
        h_val = hybrid_yearly.get(yr, {}).get('return_usd', 0)
        marker = ' <-- BEAR' if yr in [2022, 2025] else ''
        print(f"  {yr:<6} ${d_val:>+13,.0f} ${b_val:>+13,.0f} ${h_val:>+13,.0f}{marker}")

    # ================================================================
    # PATH TO $120K/YEAR
    # ================================================================
    print_section("PATH TO $120K/YEAR INCOME")

    for name, _, stats in approaches:
        if not stats or stats.get('pnl', 0) <= 0:
            continue
        total_pnl = stats['pnl']
        final = stats['final']
        n_years = len(years)
        cagr = ((final / STARTING_CAPITAL) ** (1 / n_years)) - 1 if final > STARTING_CAPITAL else 0

        print(f"\n  {name}:")
        print(f"    {n_years}-year return: ${total_pnl:+,.0f} ({stats['return_pct']:+.1f}%)")
        print(f"    CAGR: {cagr * 100:.1f}%")

        if cagr > 0:
            equity = STARTING_CAPITAL
            print(f"    Projection at {cagr*100:.1f}% CAGR:")
            print(f"    {'Year':>6} {'Capital':>14} {'Gross':>12} {'Tax':>10} {'Net':>12}")
            for yr in range(1, 16):
                gross = equity * cagr
                tax = gross * 0.15 if gross > 15000 else 0
                net = gross - tax
                marker = " <-- TARGET" if net >= 120000 else ""
                print(f"    {yr:>6} ${equity:>13,.0f} ${gross:>11,.0f} ${tax:>9,.0f} ${net:>11,.0f}{marker}")
                if net >= 120000:
                    break
                equity += net

            needed = 120000 / 0.85 / cagr if cagr > 0 else float('inf')
            print(f"    Year-1 capital needed for $120K net: ${needed:,.0f}")

    # ================================================================
    # SENSITIVITY: WHAT IF WE STARTED AT THE BEAR BOTTOM?
    # ================================================================
    print_section("SENSITIVITY: WHAT IF STARTED DCA AT NOV 2022 BEAR BOTTOM?")
    print("  The 4-year backtest starts Feb 2022 — already into the bear but not at the bottom.")
    print("  BTC bottomed Nov 2022 at $15,760. SOL at $9.64.")
    print("  Our cached data covers this period — the hybrid core accumulates during this phase")
    print("  while the current bot sits idle waiting for BTC > SMA(200).")

    # Check when first core entries happened
    first_entries = {}
    for t in core_trades:
        sym = t['symbol']
        if sym not in first_entries or t['entry_time'] < first_entries[sym]:
            first_entries[sym] = t['entry_time']
    if first_entries:
        print(f"\n  Core first entries:")
        for sym in sorted(first_entries.keys()):
            d = first_entries[sym]
            print(f"    {sym}: {d}")

    # Check when first satellite entries happened
    first_sat = {}
    for t in donch_trades:
        sym = t['symbol']
        if sym not in first_sat or t['entry_time'] < first_sat[sym]:
            first_sat[sym] = t['entry_time']
    if first_sat:
        print(f"\n  Current Bot first entries:")
        for sym in sorted(first_sat.keys()):
            d = first_sat[sym]
            d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
            print(f"    {sym}: {d_str}")

    print("\n" + "=" * 110)
    print("BACKTEST COMPLETE")
    print("=" * 110)


if __name__ == '__main__':
    main()
