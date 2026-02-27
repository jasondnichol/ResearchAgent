"""Short-Side Donchian Backtest — Futures (CFM Perps, 1x Isolated Margin)

Tests the inverse Donchian strategy for shorting during bear markets:
  - Entry: Close < N-day Donchian low (breakdown) + volume + EMA filter
  - Exit: ATR trailing stop (inverted) + N-day high exit + max hold + emergency
  - Bear filter: BTC < SMA(200) (inverse of production bull filter)
  - Funding rate: modeled as flat daily rate on position value
  - Fees: Coinbase CFM 0.06% taker + 0.05% slippage = 0.11% per side

Run structure:
  1. Base cases (Turtle Mirror, Recommended, Tight, Wide)
  2. Full parameter sweep — SMA(200) filter (144 combos)
  3. Bear filter comparison (SMA(200) vs death cross, top 5)
  4. Death cross deep dive — full sweep (144 combos)
  5. Death cross walk-forward (top 5, train/test)
  6. Death cross per-coin & win/loss breakdown
  7. Funding rate sensitivity (death cross)
  8. Combined long+short vs long-only (death cross shorts)
  9. Final verdict
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
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Perp-eligible coins (NEAR has no perp, replaced with DOGE)
SHORT_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

# Default short parameters
SHORT_DEFAULT_PARAMS = {
    'label': 'Short Base (Turtle Mirror)',
    'donchian_period': 20,
    'exit_period': 10,
    'atr_period': 14,
    'atr_mult': 2.0,
    'volume_mult': 1.5,
    'ema_period': 21,
    'rsi_blowoff': 20,            # inverse: tighten if RSI < 20 (oversold bounce risk)
    'volume_blowoff': 3.0,
    'atr_mult_tight': 1.0,        # tightened stop for bounce risk
    'emergency_stop_pct': 15.0,
    'max_hold_days': 30,
    # Coinbase CFM fees (much lower than spot)
    'fee_pct': 0.06,
    'slippage_pct': 0.05,
    # Funding rate: negative = shorts receive
    'funding_rate_daily': -0.03,
    # Portfolio
    'risk_per_trade_pct': 2.0,
    'max_positions': 4,
    'starting_capital': 10000.0,
    # Partial profit taking (on price drops)
    'tp1_pct': 10.0,
    'tp2_pct': 20.0,
    'tp1_fraction': 0.25,
    'tp2_fraction': 0.25,
}

# Named variants
SHORT_RECOMMENDED = {
    **SHORT_DEFAULT_PARAMS,
    'label': 'Recommended (15d/10d/2.5x)',
    'donchian_period': 15,
    'atr_mult': 2.5,
}

SHORT_TIGHT = {
    **SHORT_DEFAULT_PARAMS,
    'label': 'Tight (15d/10d/3x)',
    'donchian_period': 15,
    'atr_mult': 3.0,
}

SHORT_WIDE = {
    **SHORT_DEFAULT_PARAMS,
    'label': 'Wide (20d/15d/3.5x)',
    'exit_period': 15,
    'atr_mult': 3.5,
}

# Pyramiding constants
PYRAMID_GAIN_PCT = 15.0
PYRAMID_RISK_PCT = 0.01

# Phase 3 production long params (for combined test)
LONG_PROD_PARAMS = {
    **DEFAULT_PARAMS,
    'label': 'Long-Only (Phase 3 Production)',
    'atr_mult': 4.0,
    'tp1_fraction': 0.20,
    'tp2_fraction': 0.20,
}


# ============================================================================
# SHORT-SPECIFIC INDICATORS
# ============================================================================

def calculate_short_indicators(df, params):
    """Calculate indicators for short-side trading.

    Reuses base calculate_indicators for donchian_high/low, ATR, volume_sma,
    ema_21, and RSI. Adds exit_high (N-day rolling HIGH for short cover).
    """
    df = calculate_indicators(df, params)
    df['exit_high'] = df['high'].rolling(window=params['exit_period']).max()
    return df


# ============================================================================
# BEAR FILTER
# ============================================================================

def compute_btc_bear_filter(btc_df, require_death_cross=False):
    """Compute daily bear signal from BTC price data.

    Simple: Bear = BTC close < SMA(200)
    Strict: Bear = BTC close < SMA(200) AND SMA(50) < SMA(200) (death cross)

    Returns dict: date -> bool (True = bear confirmed)
    """
    df = btc_df.copy()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    bear_filter = {}
    for _, row in df.iterrows():
        if pd.isna(row['sma_200']) or pd.isna(row['sma_50']):
            bear_filter[row['time'].date()] = False
            continue

        below_200 = row['close'] < row['sma_200']
        if require_death_cross:
            death_cross = row['sma_50'] < row['sma_200']
            bear_filter[row['time'].date()] = bool(below_200 and death_cross)
        else:
            bear_filter[row['time'].date()] = bool(below_200)

    total = len(bear_filter)
    bear_days = sum(1 for v in bear_filter.values() if v)
    bull_days = total - bear_days
    mode = "death cross" if require_death_cross else "SMA(200) only"
    print(f"  BTC Bear Filter ({mode}): {bear_days} bear days ({bear_days/total*100:.1f}%), "
          f"{bull_days} non-bear days ({bull_days/total*100:.1f}%)")

    return bear_filter


# ============================================================================
# SHORT-SIDE PORTFOLIO BACKTEST
# ============================================================================

def backtest_portfolio_short(coin_data, params, bear_filter, pyramiding=False):
    """Portfolio backtest for short positions with bear filter.

    Mirrors backtest_portfolio_phase3 with inverted entry/exit logic:
    - Entry: close < Donchian LOW + volume + EMA below + bear filter
    - Exit: trailing stop (inverted), Donchian high exit, emergency, max hold
    - P&L: profit when price drops, loss when price rises
    - Funding: daily rate applied to hold duration

    Returns: (trades, equity_curve, capital, pyramid_adds)
    """
    label = params['label']
    if pyramiding:
        label += ' + pyramid'
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']
    funding_daily = params.get('funding_rate_daily', 0) / 100  # convert % to decimal

    # Pre-calculate indicators (with exit_high for short cover)
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_short_indicators(df, params)
        df_ind = df_ind.dropna(subset=['donchian_high', 'donchian_low', 'atr',
                                        'volume_sma', 'ema_21', 'rsi', 'exit_high'])
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

    # Previous day lookups
    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # State
    positions = {}
    trades = []
    equity_curve = []
    pyramid_adds = 0

    for date in all_dates:
        is_bear = bear_filter.get(date, False)

        # === EXITS (check all open short positions) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_low = float(row['low'])
            current_atr = float(row['atr'])

            # Update low watermark (inverse of high watermark for longs)
            pos['low_watermark'] = min(pos['low_watermark'], current_low)
            pos['hold_days'] += 1

            exit_reason = None

            # Inverse blow-off: tighten stop if RSI < 20 AND vol > 3x (bounce risk)
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            volume_ratio = float(row['volume']) / vol_sma
            is_bounce_risk = (volume_ratio > params['volume_blowoff']
                              and float(row['rsi']) < params['rsi_blowoff'])
            stop_mult = params['atr_mult_tight'] if is_bounce_risk else params['atr_mult']

            # 1. ATR trailing stop (inverted): low_watermark + N*ATR
            trailing_stop = pos['low_watermark'] + (stop_mult * current_atr)
            if current_close >= trailing_stop:
                exit_reason = f'Trailing stop ({stop_mult}x ATR)'

            # 2. Donchian exit: close > N-day HIGH (upward reversal)
            prev_row = prev_lookups[symbol].get(date)
            if not exit_reason and prev_row is not None and pd.notna(prev_row.get('exit_high')):
                if current_close > float(prev_row['exit_high']):
                    exit_reason = f'Donchian exit ({params["exit_period"]}-day high)'

            # 3. Emergency stop: price rose X% above entry
            emergency_level = pos['entry_price'] * (1 + params['emergency_stop_pct'] / 100)
            if not exit_reason and current_close >= emergency_level:
                exit_reason = f'Emergency stop (+{params["emergency_stop_pct"]:.0f}%)'

            # 4. Max hold time
            if not exit_reason and pos['hold_days'] >= params['max_hold_days']:
                exit_reason = f'Max hold ({params["max_hold_days"]}d)'

            # Partial profit taking (inverted: profit when price drops)
            if not exit_reason:
                # For shorts: gain when price drops below entry
                gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                    # Cover 25% — buy back at current price + fee
                    partial_exit = current_close * (1 + cost_per_side)
                    partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                    # Add funding for partial
                    partial_funding = funding_daily * pos['hold_days'] * 100
                    partial_pnl += partial_funding
                    partial_size = pos['size_usd'] * params['tp1_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 1
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_exit,
                        'pnl_pct': partial_pnl,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP1 (-{params["tp1_pct"]:.0f}% drop)',
                        'size_usd': partial_size,
                        'side': 'SHORT',
                        'win': True,
                    })
                elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                    partial_exit = current_close * (1 + cost_per_side)
                    partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                    partial_funding = funding_daily * pos['hold_days'] * 100
                    partial_pnl += partial_funding
                    partial_size = pos['size_usd'] * params['tp2_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 2
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_exit,
                        'pnl_pct': partial_pnl,
                        'hold_days': pos['hold_days'],
                        'exit_reason': f'Partial TP2 (-{params["tp2_pct"]:.0f}% drop)',
                        'size_usd': partial_size,
                        'side': 'SHORT',
                        'win': True,
                    })

            # Full exit
            if exit_reason:
                exit_price_adj = current_close * (1 + cost_per_side)  # buy back + fee
                price_pnl_pct = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
                funding_pnl_pct = funding_daily * pos['hold_days'] * 100
                total_pnl_pct = price_pnl_pct + funding_pnl_pct
                pnl_usd = pos['size_usd'] * (total_pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd

                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'],
                    'exit_time': row['time'],
                    'exit_price': exit_price_adj,
                    'pnl_pct': total_pnl_pct,
                    'price_pnl_pct': price_pnl_pct,
                    'funding_pnl_pct': funding_pnl_pct,
                    'hold_days': pos['hold_days'],
                    'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'],
                    'side': 'SHORT',
                    'win': total_pnl_pct > 0,
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === PYRAMIDING (add to winning shorts when price makes new low) ===
        if pyramiding and is_bear:
            for symbol, pos in list(positions.items()):
                if pos.get('pyramided'):
                    continue

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue

                current_close = float(row['close'])
                gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100

                # Pyramid: up +15% profit AND making new N-day low
                new_low = (pd.notna(prev_row.get('donchian_low'))
                           and current_close < float(prev_row['donchian_low']))

                if gain_pct >= PYRAMID_GAIN_PCT and new_low:
                    total_equity = capital + sum(
                        p['size_usd'] * (1 + (p['entry_price'] - current_close) / p['entry_price'])
                        if lookups.get(s, {}).get(date) is not None else p['size_usd']
                        for s, p in positions.items()
                    )
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
                            'symbol': symbol,
                            'entry_time': row['time'],
                            'entry_price': current_close * (1 - cost_per_side),
                            'exit_time': row['time'],
                            'exit_price': current_close,
                            'pnl_pct': 0,
                            'hold_days': 0,
                            'exit_reason': f'Pyramid add (+{gain_pct:.0f}%, new low)',
                            'size_usd': add_size,
                            'side': 'SHORT',
                            'win': True,
                        })

        # === NEW SHORT ENTRIES (gated by bear filter) ===
        if len(positions) < max_positions and is_bear:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row.get('donchian_low')):
                    continue

                current_close = float(row['close'])

                # Short entry: breakdown below Donchian LOW
                breakdown = current_close < float(prev_row['donchian_low'])

                # Volume confirmation
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True

                # Trend filter: price BELOW EMA(21) (downtrend)
                trend_ok = current_close < float(row['ema_21'])

                if breakdown and volume_ok and trend_ok:
                    total_equity = capital + sum(
                        p['size_usd'] * (1 + (p['entry_price'] - float(lookups[s].get(date, row)['close'])) / p['entry_price'])
                        if lookups.get(s, {}).get(date) is not None else p['size_usd']
                        for s, p in positions.items()
                    )
                    risk_amount = total_equity * risk_pct
                    entry_price = current_close * (1 - cost_per_side)  # sell: receive less
                    atr_val = float(row['atr'])
                    stop_distance = params['atr_mult'] * atr_val
                    stop_pct = stop_distance / entry_price

                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / max_positions

                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue

                    capital -= position_size  # margin locked
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

        # Equity tracking (short P&L: gains when price falls)
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                price_change = (pos['entry_price'] - float(row['close'])) / pos['entry_price']
                current_val = pos['size_usd'] * (1 + price_change)
                total_equity += max(current_val, 0)  # can't go below 0 with isolated margin
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining positions at end of backtest
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
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'],
            'exit_time': last['time'],
            'exit_price': exit_price_adj,
            'pnl_pct': total_pnl_pct,
            'price_pnl_pct': price_pnl_pct,
            'funding_pnl_pct': funding_pnl_pct,
            'hold_days': pos['hold_days'],
            'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'],
            'side': 'SHORT',
            'win': total_pnl_pct > 0,
        })

    return trades, equity_curve, capital, pyramid_adds


# ============================================================================
# COMBINED LONG+SHORT PORTFOLIO
# ============================================================================

def backtest_combined_portfolio(coin_data, long_params, short_params,
                                bull_filter, bear_filter):
    """Run long-only (Phase 3) and short-only independently on same starting capital.

    Returns long stats, short stats, and a merged equity curve showing
    what combined performance would look like if both ran on the same
    capital pool (sequential: longs during bull, shorts during bear).

    For simplicity, runs both independently and merges equity curves additively
    (net PnL from both added to starting capital).
    """
    # Run long-only with Phase 3 params
    long_trades, long_eq, long_cap, long_pyrs = backtest_portfolio_phase3(
        coin_data, long_params, bull_filter, pyramiding=True)

    # Run short-only
    short_trades, short_eq, short_cap, short_pyrs = backtest_portfolio_short(
        coin_data, short_params, bear_filter, pyramiding=False)

    starting = long_params['starting_capital']

    # Merge equity curves by date
    long_by_date = {e['date']: e['equity'] for e in long_eq}
    short_by_date = {e['date']: e['equity'] for e in short_eq}
    all_dates = sorted(set(long_by_date.keys()) | set(short_by_date.keys()))

    combined_eq = []
    for date in all_dates:
        long_val = long_by_date.get(date, starting)
        short_val = short_by_date.get(date, starting)
        # Combined = starting + (long_pnl) + (short_pnl)
        combined = starting + (long_val - starting) + (short_val - starting)
        combined_eq.append({'date': date, 'equity': combined})

    return long_trades, short_trades, long_eq, short_eq, combined_eq


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep(coin_data, bear_filter):
    """Run parameter sweep across all short strategy variants.

    Returns list of result dicts sorted by total_return_pct.
    """
    sweep_configs = []
    for lookback in [10, 15, 20, 25]:
        for atr_mult in [2.0, 2.5, 3.0, 3.5]:
            for exit_period in [5, 10, 15]:
                for vol_mult in [1.0, 1.5, 2.0]:
                    label = f"L{lookback}_A{atr_mult}_E{exit_period}_V{vol_mult}"
                    p = {
                        **SHORT_DEFAULT_PARAMS,
                        'label': label,
                        'donchian_period': lookback,
                        'atr_mult': atr_mult,
                        'exit_period': exit_period,
                        'volume_mult': vol_mult,
                    }
                    sweep_configs.append(p)

    results = []
    total = len(sweep_configs)
    for i, params in enumerate(sweep_configs):
        trades, eq, cap, _ = backtest_portfolio_short(
            coin_data, params, bear_filter, pyramiding=False)
        stats = compute_stats(trades, params['label'])
        results.append({
            'label': params['label'],
            'params': params,
            'stats': stats,
            'eq': eq,
        })
        if (i + 1) % 50 == 0:
            print(f"    Sweep progress: {i+1}/{total}")

    results.sort(key=lambda r: -r['stats'].get('total_return_pct', -999))
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("SHORT-SIDE DONCHIAN BACKTEST - FUTURES (CFM PERPS, 1x ISOLATED)")
    print(f"  Coins: {', '.join(c.replace('-USD','') for c in SHORT_COINS)}")
    print(f"  Fees: {SHORT_DEFAULT_PARAMS['fee_pct']}% taker + "
          f"{SHORT_DEFAULT_PARAMS['slippage_pct']}% slippage = "
          f"{SHORT_DEFAULT_PARAMS['fee_pct'] + SHORT_DEFAULT_PARAMS['slippage_pct']}% per side")
    print(f"  Funding: {SHORT_DEFAULT_PARAMS['funding_rate_daily']}% per day (shorts receive)")
    print(f"  Max hold: {SHORT_DEFAULT_PARAMS['max_hold_days']} days")
    print("=" * 100)

    # Fetch data
    print("\n  Fetching historical data...")
    coin_data = fetch_all_coins(coins=SHORT_COINS, years=4)
    btc_df = coin_data.get('BTC-USD')
    if btc_df is None:
        print("ERROR: No BTC data")
        return

    # Compute filters
    print("\n  Computing bear filters...")
    bear_simple = compute_btc_bear_filter(btc_df, require_death_cross=False)
    bear_strict = compute_btc_bear_filter(btc_df, require_death_cross=True)
    print("\n  Computing bull filter (for combined test)...")
    bull_filter = compute_btc_bull_filter(btc_df)

    # ==================================================================
    # SECTION 1: BASE CASES
    # ==================================================================
    print_section("SECTION 1: BASE CASES (SMA(200) bear filter)")

    base_variants = [
        ('Turtle Mirror (20d/10d/2x)',    SHORT_DEFAULT_PARAMS, False),
        ('Recommended (15d/10d/2.5x)',    SHORT_RECOMMENDED,    False),
        ('Tight (15d/10d/3x)',            SHORT_TIGHT,          False),
        ('Wide (20d/15d/3.5x)',           SHORT_WIDE,           False),
        ('Turtle + pyramid',              SHORT_DEFAULT_PARAMS, True),
        ('Recommended + pyramid',         SHORT_RECOMMENDED,    True),
    ]

    base_results = []
    for name, params, pyramid in base_variants:
        trades, eq, cap, p_adds = backtest_portfolio_short(
            coin_data, params, bear_simple, pyramiding=pyramid)
        stats = compute_stats(trades, name)
        base_results.append((name, stats, eq, trades, p_adds))

    print(f"\n  {'='*130}")
    print(f"  BASE CASE RESULTS (Full Period)")
    print(f"  {'='*130}")
    for name, stats, eq, _, p_adds in base_results:
        print_stats_row(name, stats, eq)
        if p_adds > 0:
            print(f"    (pyramid adds: {p_adds})")

    # Per-coin for best base case
    best_base_idx = max(range(len(base_results)),
                        key=lambda i: base_results[i][1].get('total_return_pct', -999))
    best_base_name, best_base_stats, _, best_base_trades, _ = base_results[best_base_idx]
    print(f"\n  Best base case: {best_base_name}")
    if best_base_trades:
        per_coin = compute_per_coin_stats(best_base_trades)
        print(f"\n  Per-coin breakdown ({best_base_name}):")
        print(f"    {'Coin':<10s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'PF':>6s} {'Sum P&L%':>10s}")
        for coin, cs in sorted(per_coin.items()):
            print(f"    {coin:<10s} {cs['trades']:>6d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>5.1f}% {cs['profit_factor']:>5.2f} "
                  f"{cs['sum_pnl_pct']:>+9.1f}%")

    # Exit reason breakdown
    if best_base_trades:
        full_exits = [t for t in best_base_trades if 'Partial' not in t['exit_reason']
                      and 'Pyramid' not in t['exit_reason']]
        reasons = {}
        for t in full_exits:
            r = t['exit_reason'].split('(')[0].strip()
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  Exit reasons ({best_base_name}):")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30s} {count:>4d} ({count/len(full_exits)*100:.1f}%)")

    # ==================================================================
    # SECTION 2: PARAMETER SWEEP
    # ==================================================================
    print_section("SECTION 2: PARAMETER SWEEP (144 variants)")

    print("  Running sweep...")
    sweep_results = run_parameter_sweep(coin_data, bear_simple)

    print(f"\n  {'='*130}")
    print(f"  TOP 20 VARIANTS (by return)")
    print(f"  {'='*130}")
    for r in sweep_results[:20]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    print(f"\n  BOTTOM 5 VARIANTS:")
    for r in sweep_results[-5:]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    # ==================================================================
    # SECTION 3: BEAR FILTER COMPARISON
    # ==================================================================
    print_section("SECTION 3: BEAR FILTER COMPARISON (top 5 variants)")

    top5_params = [r['params'] for r in sweep_results[:5]]

    print(f"\n  {'='*130}")
    print(f"  SMA(200) ONLY vs DEATH CROSS")
    print(f"  {'='*130}")
    for params in top5_params:
        # SMA(200) only
        trades_s, eq_s, _, _ = backtest_portfolio_short(coin_data, params, bear_simple)
        stats_s = compute_stats(trades_s, f"{params['label']} [SMA200]")

        # Death cross
        trades_d, eq_d, _, _ = backtest_portfolio_short(coin_data, params, bear_strict)
        stats_d = compute_stats(trades_d, f"{params['label']} [DeathX]")

        print_stats_row(stats_s['label'], stats_s, eq_s)
        print_stats_row(stats_d['label'], stats_d, eq_d)
        print()

    # ==================================================================
    # SECTION 4: DEATH CROSS DEEP DIVE — FULL PARAMETER SWEEP
    # ==================================================================
    print_section("SECTION 4: DEATH CROSS DEEP DIVE — FULL SWEEP (144 variants)")

    print("  The death cross filter (SMA50 < SMA200 + price < SMA200) showed")
    print("  dramatic improvements in Section 3. Running full sweep with death cross...")
    dc_sweep_results = run_parameter_sweep(coin_data, bear_strict)

    print(f"\n  {'='*130}")
    print(f"  TOP 20 DEATH CROSS VARIANTS (by return)")
    print(f"  {'='*130}")
    for r in dc_sweep_results[:20]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    print(f"\n  BOTTOM 5 DEATH CROSS VARIANTS:")
    for r in dc_sweep_results[-5:]:
        print_stats_row(r['label'], r['stats'], r['eq'])

    # Side-by-side SMA200 vs death cross for top 10 DC variants
    print(f"\n  {'='*130}")
    print(f"  SMA(200) vs DEATH CROSS — Top 10 Side-by-Side")
    print(f"  {'='*130}")
    print(f"  {'Variant':<25s}  {'--- SMA(200) ---':^35s}  {'--- Death Cross ---':^35s}  {'Delta':>7s}")
    for r in dc_sweep_results[:10]:
        params = r['params']
        sma_match = next((s for s in sweep_results if s['label'] == params['label']), None)
        sma_ret = sma_match['stats'].get('total_return_pct', -999) if sma_match else -999
        sma_pf = sma_match['stats'].get('profit_factor', 0) if sma_match else 0
        sma_t = sma_match['stats'].get('total_trades', 0) if sma_match else 0
        sma_dd = compute_max_drawdown(sma_match['eq'])[0] if sma_match else 0

        dc_ret = r['stats'].get('total_return_pct', -999)
        dc_pf = r['stats'].get('profit_factor', 0)
        dc_t = r['stats'].get('total_trades', 0)
        dc_dd = compute_max_drawdown(r['eq'])[0]

        print(f"  {params['label']:<25s}  "
              f"{sma_t:>3d}t {sma_pf:>5.2f}PF {sma_ret:>+7.1f}% {sma_dd:>5.1f}%DD  "
              f"{dc_t:>3d}t {dc_pf:>5.2f}PF {dc_ret:>+7.1f}% {dc_dd:>5.1f}%DD  "
              f"{dc_ret - sma_ret:>+7.1f}%")

    # Pattern analysis: what parameters dominate top 20?
    print(f"\n  PARAMETER PATTERN ANALYSIS (Top 20 death cross variants):")
    top20_dc = dc_sweep_results[:20]
    for param_name, key in [('Lookback', 'donchian_period'),
                             ('ATR mult', 'atr_mult'),
                             ('Exit period', 'exit_period'),
                             ('Volume mult', 'volume_mult')]:
        vals = [r['params'][key] for r in top20_dc]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        print(f"    {param_name:<16s}: {dict(sorted(counts.items()))}")

    # ==================================================================
    # SECTION 5: DEATH CROSS WALK-FORWARD
    # ==================================================================
    print_section("SECTION 5: DEATH CROSS WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')
    train_data, test_data = split_coin_data(coin_data, cutoff)
    btc_train = train_data.get('BTC-USD')
    btc_test = test_data.get('BTC-USD')

    # Period-specific filters
    print("  Computing period-specific filters...")
    train_bear_sma = compute_btc_bear_filter(btc_train, require_death_cross=False) if btc_train is not None else {}
    test_bear_sma = compute_btc_bear_filter(btc_test, require_death_cross=False) if btc_test is not None else {}
    train_bear_dc = compute_btc_bear_filter(btc_train, require_death_cross=True) if btc_train is not None else {}
    test_bear_dc = compute_btc_bear_filter(btc_test, require_death_cross=True) if btc_test is not None else {}

    # Death cross walk-forward on top 5
    dc_wf_results = []
    for r in dc_sweep_results[:5]:
        params = r['params']
        train_trades, train_eq, _, _ = backtest_portfolio_short(
            train_data, params, train_bear_dc)
        train_stats = compute_stats(train_trades, f"{params['label']} [train]")

        test_trades, test_eq, _, _ = backtest_portfolio_short(
            test_data, params, test_bear_dc)
        test_stats = compute_stats(test_trades, f"{params['label']} [OOS]")

        dc_wf_results.append((params['label'], train_stats, train_eq,
                               test_stats, test_eq, params))

    print(f"\n  {'='*130}")
    print(f"  TRAIN (2022-2024) — Death Cross")
    print(f"  {'='*130}")
    for label, train_stats, train_eq, _, _, _ in dc_wf_results:
        print_stats_row(label, train_stats, train_eq)

    print(f"\n  {'='*130}")
    print(f"  OUT-OF-SAMPLE (2025-2026) — Death Cross")
    print(f"  {'='*130}")
    for label, _, _, test_stats, test_eq, _ in dc_wf_results:
        print_stats_row(label, test_stats, test_eq)

    # SMA200 walk-forward for comparison
    sma_wf_results = []
    for r in sweep_results[:5]:
        params = r['params']
        test_trades, test_eq, _, _ = backtest_portfolio_short(
            test_data, params, test_bear_sma)
        test_stats = compute_stats(test_trades, f"{params['label']} [OOS]")
        sma_wf_results.append((params['label'], test_stats, test_eq))

    print(f"\n  {'='*130}")
    print(f"  OOS COMPARISON: SMA(200) vs Death Cross")
    print(f"  {'='*130}")
    print(f"\n  SMA(200) top 5 OOS:")
    for label, test_stats, test_eq in sma_wf_results:
        print_stats_row(label, test_stats, test_eq)
    print(f"\n  Death Cross top 5 OOS:")
    for label, _, _, test_stats, test_eq, _ in dc_wf_results:
        print_stats_row(label, test_stats, test_eq)

    # ==================================================================
    # SECTION 6: DEATH CROSS PER-COIN & WIN/LOSS ANALYSIS
    # ==================================================================
    print_section("SECTION 6: DEATH CROSS — PER-COIN & TRADE ANALYSIS")

    best_dc_params = dc_sweep_results[0]['params']
    dc_full_trades, dc_full_eq, _, _ = backtest_portfolio_short(
        coin_data, best_dc_params, bear_strict)

    print(f"  Best death cross variant: {best_dc_params['label']}")

    if dc_full_trades:
        per_coin_dc = compute_per_coin_stats(dc_full_trades)
        print(f"\n  {'Coin':<10s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'PF':>6s} "
              f"{'Sum P&L%':>10s} {'Avg Hold':>9s}")
        for coin, cs in sorted(per_coin_dc.items()):
            coin_exits = [t for t in dc_full_trades
                          if t['symbol'] == coin
                          and 'Partial' not in t['exit_reason']
                          and 'Pyramid' not in t['exit_reason']]
            avg_hold = (sum(t['hold_days'] for t in coin_exits) / len(coin_exits)
                        if coin_exits else 0)
            print(f"  {coin:<10s} {cs['trades']:>6d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>5.1f}% {cs['profit_factor']:>5.2f} "
                  f"{cs['sum_pnl_pct']:>+9.1f}% {avg_hold:>7.1f}d")

        # Exit reason breakdown
        full_exits = [t for t in dc_full_trades
                      if 'Partial' not in t['exit_reason']
                      and 'Pyramid' not in t['exit_reason']]
        reasons = {}
        for t in full_exits:
            r = t['exit_reason'].split('(')[0].strip()
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  Exit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(full_exits) * 100 if full_exits else 0
            print(f"    {reason:<30s} {count:>4d} ({pct:.1f}%)")

        # Win/loss distribution
        wins = [t for t in dc_full_trades
                if t.get('win', False)
                and 'Partial' not in t['exit_reason']
                and 'Pyramid' not in t['exit_reason']]
        losses = [t for t in dc_full_trades
                  if not t.get('win', False)
                  and 'Partial' not in t['exit_reason']
                  and 'Pyramid' not in t['exit_reason']]
        if wins:
            avg_win = sum(t['pnl_pct'] for t in wins) / len(wins)
            max_win = max(t['pnl_pct'] for t in wins)
            med_win = sorted(t['pnl_pct'] for t in wins)[len(wins) // 2]
            print(f"\n  Winning trades: {len(wins)}")
            print(f"    Avg: {avg_win:+.1f}%  Median: {med_win:+.1f}%  Max: {max_win:+.1f}%")
        if losses:
            avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses)
            max_loss = min(t['pnl_pct'] for t in losses)
            med_loss = sorted(t['pnl_pct'] for t in losses)[len(losses) // 2]
            print(f"  Losing trades:  {len(losses)}")
            print(f"    Avg: {avg_loss:+.1f}%  Median: {med_loss:+.1f}%  Max: {max_loss:+.1f}%")

        # Hold time distribution
        all_exits = [t for t in dc_full_trades
                     if 'Partial' not in t['exit_reason']
                     and 'Pyramid' not in t['exit_reason']]
        if all_exits:
            hold_days = [t['hold_days'] for t in all_exits]
            avg_hold = sum(hold_days) / len(hold_days)
            min_hold = min(hold_days)
            max_hold = max(hold_days)
            print(f"\n  Hold time: avg {avg_hold:.1f}d, min {min_hold}d, max {max_hold}d")

    # ==================================================================
    # SECTION 7: FUNDING RATE SENSITIVITY (DEATH CROSS)
    # ==================================================================
    print_section("SECTION 7: FUNDING RATE SENSITIVITY (Death Cross)")

    funding_rates = [-0.05, -0.03, 0.0, 0.03, 0.05]
    print(f"  Variant: {best_dc_params['label']}")
    print(f"\n  {'='*130}")
    for rate in funding_rates:
        p = {**best_dc_params, 'funding_rate_daily': rate}
        trades_f, eq_f, _, _ = backtest_portfolio_short(coin_data, p, bear_strict)
        stats_f = compute_stats(trades_f, f"Funding {rate:+.2f}%/day")
        print_stats_row(stats_f['label'], stats_f, eq_f)

    # ==================================================================
    # SECTION 8: COMBINED LONG+SHORT (DEATH CROSS SHORTS)
    # ==================================================================
    print_section("SECTION 8: COMBINED LONG+SHORT vs LONG-ONLY")

    # SMA200 combined
    l_trades, s_trades, l_eq, s_eq, comb_eq = backtest_combined_portfolio(
        coin_data, LONG_PROD_PARAMS, sweep_results[0]['params'],
        bull_filter, bear_simple)
    long_stats = compute_stats(l_trades, 'Long-Only (Phase 3)')
    short_stats_sma = compute_stats(s_trades, 'Short (SMA200)')
    combined_stats_sma = compute_stats(l_trades + s_trades, 'Combined (SMA200 shorts)')

    # Death cross combined
    l_trades_dc, s_trades_dc, l_eq_dc, s_eq_dc, comb_eq_dc = backtest_combined_portfolio(
        coin_data, LONG_PROD_PARAMS, best_dc_params,
        bull_filter, bear_strict)
    long_stats_dc = compute_stats(l_trades_dc, 'Long-Only (Phase 3)')
    short_stats_dc = compute_stats(s_trades_dc, 'Short (DeathX)')
    combined_stats_dc = compute_stats(l_trades_dc + s_trades_dc, 'Combined (DeathX shorts)')

    print(f"\n  {'='*130}")
    print(f"  FULL COMPARISON")
    print(f"  {'='*130}")
    print_stats_row('Long-Only (Phase 3 Prod)', long_stats, l_eq)
    print_stats_row(f'Short SMA200 ({sweep_results[0]["label"][:20]})',
                    short_stats_sma, s_eq)
    print_stats_row(f'Short DeathX ({best_dc_params["label"][:20]})',
                    short_stats_dc, s_eq_dc)
    print_stats_row('Combined (SMA200 shorts)', combined_stats_sma, comb_eq)
    print_stats_row('Combined (DeathX shorts)', combined_stats_dc, comb_eq_dc)

    # Time-in-market analysis
    total_days = len(bear_simple)
    bear_days_sma = sum(1 for v in bear_simple.values() if v)
    bear_days_dc = sum(1 for v in bear_strict.values() if v)
    bull_days = sum(1 for v in bull_filter.values() if v)

    print(f"\n  Time-in-market:")
    print(f"    Long (bull filter):      {bull_days:>4d} days ({bull_days/total_days*100:.1f}%)")
    print(f"    Short (SMA200 bear):     {bear_days_sma:>4d} days ({bear_days_sma/total_days*100:.1f}%)")
    print(f"    Short (death cross):     {bear_days_dc:>4d} days ({bear_days_dc/total_days*100:.1f}%)")
    print(f"    Death cross is {bear_days_sma - bear_days_dc} fewer days = more selective filter")

    # ==================================================================
    # SECTION 9: FINAL VERDICT
    # ==================================================================
    print_section("FINAL VERDICT — SHORT-SIDE VIABILITY")

    best_dc = dc_sweep_results[0] if dc_sweep_results else None
    best_sma = sweep_results[0] if sweep_results else None

    print(f"\n  {'='*90}")
    print(f"  BEAR FILTER COMPARISON — FULL PERIOD")
    print(f"  {'='*90}")

    if best_sma:
        ss = best_sma['stats']
        sd, _ = compute_max_drawdown(best_sma['eq'])
        print(f"  SMA(200) best:    {best_sma['label']}")
        print(f"    {ss.get('total_trades',0):>3d}t  "
              f"WR:{ss.get('win_rate',0)*100:.1f}%  "
              f"PF:{ss.get('profit_factor',0):.2f}  "
              f"Ret:{ss.get('total_return_pct',0):+.1f}%  "
              f"DD:{sd:.1f}%")

    if best_dc:
        ds = best_dc['stats']
        dd_val, _ = compute_max_drawdown(best_dc['eq'])
        print(f"\n  Death Cross best: {best_dc['label']}")
        print(f"    {ds.get('total_trades',0):>3d}t  "
              f"WR:{ds.get('win_rate',0)*100:.1f}%  "
              f"PF:{ds.get('profit_factor',0):.2f}  "
              f"Ret:{ds.get('total_return_pct',0):+.1f}%  "
              f"DD:{dd_val:.1f}%")

    # OOS summary
    if dc_wf_results:
        best_dc_oos = max(dc_wf_results, key=lambda r: r[3].get('total_return_pct', -999))
        oos_s = best_dc_oos[3]
        oos_dd, _ = compute_max_drawdown(best_dc_oos[4])
        print(f"\n  Death Cross OOS:  {best_dc_oos[0]}")
        print(f"    {oos_s.get('total_trades',0):>3d}t  "
              f"WR:{oos_s.get('win_rate',0)*100:.1f}%  "
              f"PF:{oos_s.get('profit_factor',0):.2f}  "
              f"Ret:{oos_s.get('total_return_pct',0):+.1f}%  "
              f"DD:{oos_dd:.1f}%")

    # Viability
    print(f"\n  {'='*90}")
    print(f"  VIABILITY ASSESSMENT")
    print(f"  {'='*90}")

    if best_dc:
        dc_pf = ds.get('profit_factor', 0)
        dc_trades = ds.get('total_trades', 0)
        dc_return = ds.get('total_return_pct', 0)

        if dc_trades < 10:
            print(f"  INSUFFICIENT DATA: Only {dc_trades} trades.")
        elif dc_pf >= 1.5 and dc_return > 0:
            print(f"  VIABLE: Death cross short strategy meets thresholds.")
            print(f"    PF {dc_pf:.2f} >= 1.5, return {dc_return:+.1f}%")
        elif dc_pf >= 1.0 and dc_return > 0:
            print(f"  PROMISING: PF {dc_pf:.2f} >= 1.0, return {dc_return:+.1f}%")
            print(f"    Needs OOS confirmation before proceeding.")
        else:
            print(f"  WEAK: PF {dc_pf:.2f}, return {dc_return:+.1f}%")
            print(f"    Short strategy may not be viable.")

    # Combined benefit
    if combined_stats_dc.get('total_trades', 0) > 0:
        long_ret = long_stats_dc.get('total_return_pct', 0)
        comb_ret = combined_stats_dc.get('total_return_pct', 0)
        long_dd_val, _ = compute_max_drawdown(l_eq_dc) if l_eq_dc else (0, 0)
        comb_dd_val, _ = compute_max_drawdown(comb_eq_dc) if comb_eq_dc else (0, 0)

        print(f"\n  COMBINED PORTFOLIO:")
        print(f"    Long-only:   {long_ret:>+7.1f}%  MaxDD: {long_dd_val:.1f}%")
        print(f"    Combined:    {comb_ret:>+7.1f}%  MaxDD: {comb_dd_val:.1f}%")
        delta = comb_ret - long_ret
        print(f"    Shorts add:  {delta:>+7.1f}% during bear markets")

    # Recommended params
    print(f"\n  RECOMMENDED SHORT PARAMETERS:")
    if best_dc:
        bp = best_dc['params']
        print(f"    Bear filter:       Death cross (SMA50 < SMA200 + price < SMA200)")
        print(f"    Donchian lookback: {bp['donchian_period']} days")
        print(f"    Exit period:       {bp['exit_period']} days")
        print(f"    ATR multiplier:    {bp['atr_mult']}x")
        vol_str = 'OFF' if bp['volume_mult'] == 0 else f"{bp['volume_mult']}x avg"
        print(f"    Volume filter:     {vol_str}")
        print(f"    Max hold:          {bp['max_hold_days']} days")
        print(f"    Emergency stop:    {bp['emergency_stop_pct']}%")

    print(f"\n  NEXT STEPS:")
    print(f"    1. If viable -> Phase F2: Coinbase CFM API integration")
    print(f"    2. Paper trade shorts 30-60 days alongside long bot")
    print(f"    3. If OOS confirms -> Phase F3: dual-mode bot")

    print(f"\n{'='*100}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*100}")



if __name__ == "__main__":
    main()
